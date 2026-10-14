"""Labeled feedback store: ground-truth outcomes that arrive after predictions were served."""

import threading

import pandas as pd

from configs.settings import settings
from src.feature_engineering.features import FEATURE_COLUMNS, TARGET

_LOCK = threading.Lock()


def _path():
    return settings.data_drift_dir / "feedback.parquet"


def add_feedback(features: dict, label: int) -> int:
    """Append one labeled row; returns the number of stored feedback rows."""
    row = {name: float(features[name]) for name in FEATURE_COLUMNS}
    row[TARGET] = int(label)
    with _LOCK:
        settings.data_drift_dir.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame([row])
        combined = (
            pd.concat([pd.read_parquet(_path()), new], ignore_index=True)
            if _path().exists()
            else new
        )
        combined.to_parquet(_path(), index=False)
        return len(combined)


def load_feedback() -> pd.DataFrame:
    with _LOCK:
        return (
            pd.read_parquet(_path())
            if _path().exists()
            else pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET])
        )


def clear_feedback() -> None:
    with _LOCK:
        if _path().exists():
            _path().unlink()
