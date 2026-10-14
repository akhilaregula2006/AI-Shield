"""
Lightweight file-based feature store: one parquet file + a JSON manifest carrying
version, row count, and a SHA-256 checksum verified on every read.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from configs.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_MAX_HISTORY = 20


def manifest_path() -> Path:
    return settings.feature_store_path.parent / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> Dict[str, Any]:
    path = manifest_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_features(df: pd.DataFrame, source: str = "unknown") -> Dict[str, Any]:
    """Persist features and bump the manifest version."""
    path = settings.feature_store_path
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = read_manifest()
    df.to_parquet(path, index=False)
    history = previous.get("history", [])
    if previous:
        history.append({k: previous[k] for k in ("version", "rows", "sha256", "created_at")})
    manifest = {
        "version": previous.get("version", 0) + 1,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "fraud_rate": float(df["Class"].mean()) if "Class" in df.columns else None,
        "sha256": _sha256(path),
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": history[-_MAX_HISTORY:],
    }
    manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Feature store v%d written (%d rows)", manifest["version"], manifest["rows"])
    return manifest


def read_features(verify: bool = True) -> pd.DataFrame:
    path = settings.feature_store_path
    if not path.exists():
        raise FileNotFoundError(f"Feature store empty: {path}. Run scripts/run_pipeline.py")
    if verify:
        expected = read_manifest().get("sha256")
        if expected and expected != _sha256(path):
            raise ValueError("Feature store checksum mismatch; file changed outside the pipeline")
    return pd.read_parquet(path)


def append_features(new_df: pd.DataFrame, source: str = "append") -> Dict[str, Any]:
    """Append labeled rows (e.g. production feedback) and write a new version."""
    combined = pd.concat([read_features(), new_df], ignore_index=True)
    return write_features(combined, source=source)
