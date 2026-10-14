import shutil
from pathlib import Path

import pytest

from api import main as api_main
from configs.settings import settings
from src.deployment import registry
from src.deployment.service import model_service
from src.feature_engineering.features import engineer_features
from src.feature_store import store
from src.ingestion.loader import generate_synthetic_data
from src.security import rate_limit
from src.training.train import train_model

_PATHS = {
    "models_dir": "models",
    "data_drift_dir": "drift",
    "feature_store_path": "feature_store/features.parquet",
}


def _point_settings_at(monkeypatch, root: Path) -> None:
    for attr, rel in _PATHS.items():
        monkeypatch.setattr(settings, attr, root / rel)
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"file://{root / 'mlruns'}")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own models/drift/feature-store dirs and clean singletons."""
    _point_settings_at(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "rate_limit_requests", 100_000)
    rate_limit.reset_limiter()
    model_service._loaded, model_service._explainer = None, None  # pylint: disable=protected-access
    api_main.buffer.clear()
    yield


@pytest.fixture(scope="session")
def trained_snapshot(tmp_path_factory):
    """Train once per session (regime-A data, promoted); tests copy the result."""
    root = tmp_path_factory.mktemp("snapshot")
    mp = pytest.MonkeyPatch()
    _point_settings_at(mp, root)
    features = engineer_features(generate_synthetic_data(n_rows=6000, seed=1))
    store.write_features(features, source="test")
    result = train_model(features)
    registry.promote(result["version"], reason="test bootstrap")
    mp.undo()
    return root


@pytest.fixture
def trained_env(trained_snapshot, tmp_path):
    """Function-scoped copy of the pre-trained environment (already pointed at tmp_path)."""
    for rel in ("models", "feature_store"):
        shutil.copytree(trained_snapshot / rel, tmp_path / rel)
    return tmp_path


@pytest.fixture
def drifted_raw():
    return generate_synthetic_data(n_rows=3000, seed=2, regime="B", covariate_shift=1.0)
