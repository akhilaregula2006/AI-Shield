"""Statistical drift detection: Population Stability Index + two-sample KS test per feature."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from configs.settings import settings
from src.feature_engineering.features import FEATURE_COLUMNS

_EPS = 1e-6


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    feature_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    drifted_features: List[str] = field(default_factory=list)
    drift_share: float = 0.0
    max_psi: float = 0.0
    dataset_drift: bool = False
    insufficient_data: bool = False
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins from the reference. <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, _EPS, None), np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[List[str]] = None,
    psi_threshold: Optional[float] = None,
    ks_alpha: Optional[float] = None,
    share_threshold: Optional[float] = None,
    min_samples: Optional[int] = None,
) -> DriftReport:
    """A feature drifts when PSI >= threshold AND the KS test rejects at ``ks_alpha``."""
    psi_threshold = settings.drift_psi_threshold if psi_threshold is None else psi_threshold
    ks_alpha = settings.drift_ks_alpha if ks_alpha is None else ks_alpha
    share_threshold = (
        settings.drift_feature_share_threshold if share_threshold is None else share_threshold
    )
    min_samples = settings.drift_min_samples if min_samples is None else min_samples
    features = features or FEATURE_COLUMNS

    report = DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    if len(current) < min_samples:
        report.insufficient_data = True
        return report

    for name in features:
        ref_values, cur_values = reference[name].to_numpy(), current[name].to_numpy()
        score = psi(ref_values, cur_values)
        p_value = float(ks_2samp(ref_values, cur_values).pvalue)
        report.feature_scores[name] = {"psi": score, "ks_p_value": p_value}
        if score >= psi_threshold and p_value < ks_alpha:
            report.drifted_features.append(name)

    report.max_psi = max(v["psi"] for v in report.feature_scores.values())
    report.drift_share = len(report.drifted_features) / len(features)
    report.dataset_drift = report.drift_share >= share_threshold
    return report


def _report_path():
    return settings.data_drift_dir / "latest_report.json"


def save_report(report: DriftReport) -> None:
    settings.data_drift_dir.mkdir(parents=True, exist_ok=True)
    _report_path().write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def load_latest_report() -> Optional[dict]:
    path = _report_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
