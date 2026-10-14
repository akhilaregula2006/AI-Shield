"""
Self-healing decision engine.

decide():  drift report (+ optional live performance) -> NONE | MONITOR | RETRAIN
heal():    champion/challenger retrain on fresh labeled data; promote only if the
           challenger beats the champion on a held-out slice of that fresh data,
           otherwise production is left untouched (safe by default).
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.settings import settings
from src.deployment import registry
from src.evaluation.metrics import compute_metrics
from src.feature_engineering.features import FEATURE_COLUMNS, TARGET
from src.feature_store import store
from src.monitoring.drift import DriftReport
from src.monitoring.metrics import HEALING_ACTIONS
from src.training.train import train_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Action(str, Enum):
    NONE = "none"
    MONITOR = "monitor"
    RETRAIN = "retrain"


@dataclass
class HealingOutcome:
    action: str
    executed: bool
    promoted: bool
    reason: str
    champion_version: Optional[int] = None
    challenger_version: Optional[int] = None
    champion_metrics: Optional[Dict[str, float]] = None
    challenger_metrics: Optional[Dict[str, float]] = None
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def decide(report: DriftReport, live_metrics: Optional[Dict[str, float]] = None) -> Action:
    """Pure decision rule (easy to unit test and to explain in an interview)."""
    if live_metrics and live_metrics.get("pr_auc", 1.0) < settings.min_pr_auc:
        return Action.RETRAIN
    if report.insufficient_data:
        return Action.NONE
    if report.dataset_drift:
        return Action.RETRAIN
    if report.drifted_features:
        return Action.MONITOR
    return Action.NONE


def _log_path():
    return settings.data_drift_dir / "healing_log.jsonl"


def read_healing_log(limit: int = 10) -> list:
    path = _log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]


def _record(outcome: HealingOutcome) -> HealingOutcome:
    outcome.created_at = datetime.now(timezone.utc).isoformat()
    settings.data_drift_dir.mkdir(parents=True, exist_ok=True)
    with open(_log_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(outcome.to_dict()) + "\n")
    HEALING_ACTIONS.labels(
        action=outcome.action, outcome="promoted" if outcome.promoted else "kept_champion"
    ).inc()
    return outcome


def heal(recent_labeled: pd.DataFrame) -> HealingOutcome:
    """Retrain a challenger on stored + recent labeled data and compare with the champion."""
    champion_version = registry.get_production_version()
    if champion_version is None:
        raise RuntimeError("No champion model to heal; run scripts/run_pipeline.py first")
    if recent_labeled[TARGET].nunique() < 2 or len(recent_labeled) < settings.min_feedback_rows:
        return _record(
            HealingOutcome(
                action=Action.RETRAIN.value,
                executed=False,
                promoted=False,
                reason=(
                    f"Insufficient labeled data ({len(recent_labeled)} rows; need "
                    f">={settings.min_feedback_rows} with both classes)"
                ),
                champion_version=champion_version,
            )
        )

    fresh_train, fresh_eval = train_test_split(
        recent_labeled,
        test_size=0.4,
        stratify=recent_labeled[TARGET],
        random_state=settings.random_seed,
    )
    base = store.read_features()
    challenger = train_model(
        pd.concat([base, fresh_train], ignore_index=True), register=True, tags={"trigger": "drift"}
    )
    challenger_version = challenger["version"]

    x_eval, y_eval = fresh_eval[FEATURE_COLUMNS], fresh_eval[TARGET]
    champ = registry.load_model(champion_version)
    champ_metrics = compute_metrics(
        y_eval, champ.model.predict_proba(x_eval)[:, 1], champ.threshold
    )
    chall_metrics = compute_metrics(
        y_eval, challenger["model"].predict_proba(x_eval)[:, 1], challenger["threshold"]
    )

    promote = chall_metrics["pr_auc"] >= champ_metrics["pr_auc"] + settings.promotion_min_gain
    if promote:
        registry.promote(challenger_version, reason="challenger beat champion on fresh data")
        store.append_features(recent_labeled, source="production-feedback")
        reason = "Challenger beat champion on held-out fresh data; promoted"
    else:
        registry.reject(challenger_version, "did not beat champion on fresh data")
        reason = "Challenger did not beat champion; production unchanged"
    logger.info("Healing result: %s", reason)
    return _record(
        HealingOutcome(
            action=Action.RETRAIN.value,
            executed=True,
            promoted=promote,
            reason=reason,
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_metrics=champ_metrics,
            challenger_metrics=chall_metrics,
        )
    )
