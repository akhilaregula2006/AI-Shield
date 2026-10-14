"""
Data ingestion.

Loads the Kaggle ULB Credit Card Fraud dataset from ``data/raw/creditcard.csv``.
If the file is absent (fresh clone, CI), a synthetic generator with the same schema
(Time, V1-V28, Amount, Class) is used so the full pipeline stays runnable.
The generator also supports a second "regime" to simulate concept + covariate drift.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from configs.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *V_COLUMNS, "Amount", "Class"]

# Which PCA components separate fraud from legit, per regime. Regime B uses different
# components, so a model trained on regime A degrades on B (concept drift).
REGIME_SIGNAL = {
    "A": {"V1": -2.5, "V3": -2.0, "V4": 2.0, "V7": -1.5, "V10": -2.0, "V12": -2.5, "V14": -3.0},
    "B": {"V2": 2.5, "V5": -2.0, "V6": 2.0, "V8": -1.5, "V9": -2.0, "V18": 2.5, "V19": -2.0},
}
_COVARIATE_SHIFT_COLUMNS = ["V11", "V13", "V15", "V22", "V24"]


def generate_synthetic_data(
    n_rows: Optional[int] = None,
    fraud_rate: Optional[float] = None,
    seed: Optional[int] = None,
    regime: str = "A",
    covariate_shift: float = 0.0,
) -> pd.DataFrame:
    """Generate ULB-shaped data. ``regime='B'`` and ``covariate_shift>0`` simulate drift."""
    if regime not in REGIME_SIGNAL:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {list(REGIME_SIGNAL)}")
    n_rows = n_rows or settings.synthetic_rows
    fraud_rate = settings.synthetic_fraud_rate if fraud_rate is None else fraud_rate
    rng = np.random.default_rng(settings.random_seed if seed is None else seed)

    n_fraud = min(max(int(n_rows * fraud_rate), 20), n_rows // 2)
    labels = np.zeros(n_rows, dtype=int)
    labels[rng.choice(n_rows, size=n_fraud, replace=False)] = 1

    data = rng.normal(0.0, 1.0, size=(n_rows, len(V_COLUMNS)))
    df = pd.DataFrame(data, columns=V_COLUMNS)
    fraud_mask = labels == 1
    for col, shift in REGIME_SIGNAL[regime].items():
        df.loc[fraud_mask, col] += shift
    for col in _COVARIATE_SHIFT_COLUMNS:
        df[col] += covariate_shift

    amount = rng.lognormal(mean=3.5, sigma=1.2, size=n_rows)
    amount[fraud_mask] = rng.lognormal(mean=4.2, sigma=1.5, size=n_fraud)
    df["Amount"] = np.round(amount * (1.0 + covariate_shift), 2).tolist()
    df["Time"] = np.round(rng.uniform(0, 172_800, size=n_rows), 0).tolist()
    df["Class"] = labels.tolist()
    return df[RAW_COLUMNS]


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw dataset, falling back to synthetic data when the CSV is missing."""
    csv_path = Path(path) if path else settings.data_raw_dir / settings.raw_data_filename
    if csv_path.exists():
        logger.info("Loading raw data from %s", csv_path)
        return pd.read_csv(csv_path)
    logger.warning("%s not found; using synthetic data with the same schema", csv_path)
    return generate_synthetic_data()
