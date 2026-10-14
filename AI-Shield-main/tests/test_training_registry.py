import numpy as np
import pytest

from src.deployment import registry
from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.feature_engineering.features import FEATURE_COLUMNS, engineer_features
from src.ingestion.loader import generate_synthetic_data
from src.training.train import train_model


def test_metrics_perfect_and_single_class():
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert perfect["pr_auc"] == 1.0 and perfect["recall"] == 1.0 and perfect["precision"] == 1.0
    assert compute_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))["pr_auc"] == 0.0


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1])
    threshold = find_best_threshold(y, np.array([0.1, 0.2, 0.3, 0.7, 0.9]))
    assert 0.3 < threshold <= 0.7


def test_trained_model_quality(trained_env):
    loaded = registry.load_model()
    assert loaded.metrics["pr_auc"] > 0.9
    assert loaded.metrics["recall"] > 0.7
    assert list(loaded.reference.columns) == FEATURE_COLUMNS


def test_training_does_not_register_when_disabled(trained_env):
    feats = engineer_features(generate_synthetic_data(n_rows=1500, seed=5, fraud_rate=0.05))
    result = train_model(feats, register=False)
    assert result["version"] is None
    assert len(registry.list_versions()) == 1  # only the snapshot model


def test_promote_and_rollback(trained_env):
    feats = engineer_features(generate_synthetic_data(n_rows=1500, seed=5, fraud_rate=0.05))
    v2 = train_model(feats)["version"]
    assert registry.get_production_version() == 1
    registry.promote(v2, reason="test")
    assert registry.get_production_version() == v2
    assert registry.rollback() == 1
    assert registry.get_production_version() == 1
    statuses = {v["version"]: v["status"] for v in registry.list_versions()}
    assert statuses == {1: "production", v2: "rolled_back"}


def test_rollback_without_history_raises(trained_env):
    with pytest.raises(RuntimeError):
        registry.rollback()


def test_promote_unknown_version_raises(trained_env):
    with pytest.raises(KeyError):
        registry.promote(99)


def test_load_without_production_raises():
    with pytest.raises(LookupError):
        registry.load_model()


def test_registry_detects_tampered_artifact(trained_env):
    artifact = registry.settings.models_dir / "versions" / "v1" / "model.joblib"
    with open(artifact, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="Integrity"):
        registry.load_model()
