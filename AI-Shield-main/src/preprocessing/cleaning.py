"""Deterministic cleaning applied after validation."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates and reset the index."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        logger.info("Dropped %d duplicate rows", before - len(out))
    return out
