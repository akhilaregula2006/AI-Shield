"""Minimal local feature store: versioned Parquet files + a JSON manifest.

Deliberately lightweight (no Feast/Redis) to match the M1-local-dev
constraint noted for this project. A "feature group" is a named,
versioned table of engineered features. Each write creates a new
version so downstream training/serving can pin to a specific one.
"""

import json
import time
from pathlib import Path

import pandas as pd

DEFAULT_STORE_ROOT = Path("feature_store/data")


class FeatureStore:
    """Local, file-based feature store backed by versioned Parquet files."""

    def __init__(self, root: Path = DEFAULT_STORE_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            self._write_manifest({})

    def _read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

    def write_feature_group(self, name: str, rows: list[dict]) -> str:
        """Write a new version of a feature group. Returns the version id."""
        version = str(int(time.time() * 1000))
        group_dir = self.root / name
        group_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows)
        out_path = group_dir / f"{version}.parquet"
        df.to_parquet(out_path, index=False)

        manifest = self._read_manifest()
        manifest.setdefault(name, {})
        manifest[name]["latest"] = version
        manifest[name].setdefault("versions", []).append(
            {"version": version, "row_count": len(df), "columns": list(df.columns)}
        )
        self._write_manifest(manifest)
        return version

    def read_feature_group(self, name: str, version: str | None = None) -> pd.DataFrame:
        """Read a feature group. Defaults to the latest version."""
        manifest = self._read_manifest()
        if name not in manifest:
            raise KeyError(f"No feature group named '{name}' in store at {self.root}")

        version = version or manifest[name]["latest"]
        path = self.root / name / f"{version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Version {version} of '{name}' not found at {path}")
        return pd.read_parquet(path)

    def list_versions(self, name: str) -> list[dict]:
        manifest = self._read_manifest()
        return manifest.get(name, {}).get("versions", [])
