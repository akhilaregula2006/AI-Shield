"""
File-based model registry with versioning, promotion, rollback, and integrity checks.

Layout (under ``settings.models_dir``)::

    registry.json                      # production pointer + per-version metadata
    versions/v<N>/model.joblib         # the fitted model (SHA-256 recorded in registry.json)
    versions/v<N>/reference.parquet    # training feature sample used as the drift baseline
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from configs.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_LOCK = threading.RLock()


@dataclass
class LoadedModel:
    model: Any
    version: int
    threshold: float
    metrics: Dict[str, float]
    reference: pd.DataFrame


def _registry_file() -> Path:
    return settings.models_dir / "registry.json"


def _version_dir(version: int) -> Path:
    return settings.models_dir / "versions" / f"v{version}"


def _sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _read() -> Dict[str, Any]:
    path = _registry_file()
    if not path.exists():
        return {"next_version": 1, "production": None, "history": [], "versions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(state: Dict[str, Any]) -> None:
    path = _registry_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX


def register_model(
    model: Any,
    metrics: Dict[str, float],
    threshold: float,
    reference: pd.DataFrame,
    run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> int:
    """Store a new model version (status ``staged``); returns the version number."""
    with _LOCK:
        state = _read()
        version = state["next_version"]
        vdir = _version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib")
        reference.to_parquet(vdir / "reference.parquet", index=False)
        state["versions"][str(version)] = {
            "version": version,
            "status": "staged",
            "metrics": metrics,
            "threshold": threshold,
            "run_id": run_id,
            "tags": tags or {},
            "sha256": _sha256(vdir / "model.joblib"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        state["next_version"] = version + 1
        _write(state)
    logger.info("Registered model v%d (pr_auc=%.4f)", version, metrics.get("pr_auc", 0.0))
    return version


def get_production_version() -> Optional[int]:
    return _read()["production"]


def list_versions() -> List[Dict[str, Any]]:
    return list(_read()["versions"].values())


def promote(version: int, reason: str = "manual") -> None:
    with _LOCK:
        state = _read()
        if str(version) not in state["versions"]:
            raise KeyError(f"Model v{version} is not registered")
        current = state["production"]
        if current is not None and current != version:
            state["history"].append(current)
            state["versions"][str(current)]["status"] = "archived"
        state["production"] = version
        entry = state["versions"][str(version)]
        entry["status"] = "production"
        entry["promotion_reason"] = reason
        _write(state)
    logger.info("Promoted model v%d to production (%s)", version, reason)


def reject(version: int, reason: str) -> None:
    with _LOCK:
        state = _read()
        entry = state["versions"].get(str(version))
        if entry and entry["status"] == "staged":
            entry["status"] = "rejected"
            entry["rejection_reason"] = reason
            _write(state)


def rollback() -> int:
    """Re-point production at the previous production version."""
    with _LOCK:
        state = _read()
        if not state["history"]:
            raise RuntimeError("No previous production version to roll back to")
        current = state["production"]
        previous = state["history"].pop()
        if current is not None:
            state["versions"][str(current)]["status"] = "rolled_back"
        state["production"] = previous
        state["versions"][str(previous)]["status"] = "production"
        _write(state)
    logger.warning("Rolled back production from v%s to v%d", current, previous)
    return previous


def load_model(version: Optional[int] = None) -> LoadedModel:
    """Load a version (default: production), verifying the artifact checksum first."""
    state = _read()
    version = version if version is not None else state["production"]
    if version is None:
        raise LookupError("No production model registered yet")
    entry = state["versions"][str(version)]
    path = _version_dir(version) / "model.joblib"
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"Integrity check failed for model v{version}; artifact was modified")
    return LoadedModel(
        model=joblib.load(path),
        version=version,
        threshold=entry["threshold"],
        metrics=entry["metrics"],
        reference=pd.read_parquet(_version_dir(version) / "reference.parquet"),
    )
