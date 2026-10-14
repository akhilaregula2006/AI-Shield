"""
Feature engineering shared by training and serving (single source of truth, so
there is no train/serve skew).
"""

import numpy as np
import pandas as pd

from src.ingestion.loader import V_COLUMNS

TARGET = "Class"
FEATURE_COLUMNS = [*V_COLUMNS, "log_amount", "hour_sin", "hour_cos"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw frame (Time, V1-V28, Amount[, Class]) -> model features[, Class]."""
    out = df[V_COLUMNS].astype(float).copy()
    out["log_amount"] = np.log1p(df["Amount"].astype(float).clip(lower=0))
    hour = (df["Time"].astype(float) // 3600) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out = out[FEATURE_COLUMNS]
    if TARGET in df.columns:
        out[TARGET] = df[TARGET].astype(int)
    return out
