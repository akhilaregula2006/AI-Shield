import numpy as np
import pytest

from configs.settings import settings
from src.deployment import registry
from src.feature_engineering.features import engineer_features
from src.ingestion.loader import generate_synthetic_data
from src.monitoring import feedback as feedback_store
from src.monitoring.drift import DriftReport, detect_drift, load_latest_report, psi, save_report
from src.self_healing import engine
from src.self_healing.engine import Action, decide, heal, read_healing_log


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(2, 1, 5000)) > 0.5


def test_psi_constant_feature_is_zero():
    assert psi(np.ones(100), np.ones(100)) == 0.0


def test_no_drift_on_same_regime(trained_env):
    reference = registry.load_model().reference
    current = engineer_features(generate_synthetic_data(n_rows=2000, seed=11))
    report = detect_drift(reference, current)
    assert not report.dataset_drift and report.drift_share < 0.1


def test_drift_detected_on_shifted_regime(trained_env, drifted_raw):
    reference = registry.load_model().reference
    report = detect_drift(reference, engineer_features(drifted_raw))
    assert report.dataset_drift
    assert "V11" in report.drifted_features and report.max_psi > 0.25


def test_insufficient_data_is_reported_not_flagged(trained_env, drifted_raw):
    reference = registry.load_model().reference
    report = detect_drift(reference, engineer_features(drifted_raw).head(50))
    assert report.insufficient_data and not report.dataset_drift


def test_report_persistence_roundtrip():
    assert load_latest_report() is None
    save_report(DriftReport(n_reference=1, n_current=2, created_at="now"))
    report = load_latest_report()
    assert report is not None and report["n_current"] == 2


@pytest.mark.parametrize(
    "report,live,expected",
    [
        (DriftReport(1, 1, insufficient_data=True), None, Action.NONE),
        (DriftReport(1, 1, dataset_drift=True), None, Action.RETRAIN),
        (DriftReport(1, 1, drifted_features=["V1"]), None, Action.MONITOR),
        (DriftReport(1, 1), None, Action.NONE),
        (DriftReport(1, 1), {"pr_auc": 0.1}, Action.RETRAIN),
        (DriftReport(1, 1), {"pr_auc": 0.99}, Action.NONE),
    ],
)
def test_decision_rules(report, live, expected):
    assert decide(report, live) == expected


def test_heal_promotes_better_challenger(trained_env, drifted_raw):
    outcome = heal(engineer_features(drifted_raw))
    assert outcome.executed and outcome.promoted
    assert outcome.challenger_metrics is not None and outcome.champion_metrics is not None
    assert outcome.challenger_metrics["pr_auc"] > outcome.champion_metrics["pr_auc"] + 0.5
    assert registry.get_production_version() == outcome.challenger_version
    assert read_healing_log()[-1]["promoted"] is True


def test_heal_keeps_champion_when_challenger_not_better(trained_env, drifted_raw, monkeypatch):
    monkeypatch.setattr(settings, "promotion_min_gain", 5.0)  # impossible bar
    outcome = heal(engineer_features(drifted_raw))
    assert outcome.executed and not outcome.promoted
    assert registry.get_production_version() == 1
    statuses = {v["version"]: v["status"] for v in registry.list_versions()}
    assert statuses[outcome.challenger_version] == "rejected"


def test_heal_refuses_with_too_little_labeled_data(trained_env, drifted_raw):
    outcome = heal(engineer_features(drifted_raw).head(50))
    assert not outcome.executed and "Insufficient" in outcome.reason
    assert len(registry.list_versions()) == 1  # nothing was trained


def test_heal_requires_champion():
    with pytest.raises(RuntimeError):
        heal(engineer_features(generate_synthetic_data(n_rows=500, seed=1)))


def test_feedback_store_roundtrip():
    row = engineer_features(generate_synthetic_data(n_rows=5, seed=1)).iloc[0].to_dict()
    assert feedback_store.add_feedback(row, 1) == 1
    assert feedback_store.add_feedback(row, 0) == 2
    assert feedback_store.load_feedback()["Class"].tolist() == [1, 0]
    feedback_store.clear_feedback()
    assert feedback_store.load_feedback().empty


def test_healing_log_empty_by_default():
    assert engine.read_healing_log() == []
