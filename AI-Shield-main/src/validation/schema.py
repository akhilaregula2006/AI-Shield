"""Schema and quality validation for raw transaction data (pandera)."""

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from src.ingestion.loader import V_COLUMNS


def _build_schema() -> DataFrameSchema:
    columns = {
        "Time": Column(float, Check.ge(0), coerce=True, nullable=False),
        "Amount": Column(float, Check.ge(0), coerce=True, nullable=False),
        "Class": Column(int, Check.isin([0, 1]), coerce=True, nullable=False),
    }
    for name in V_COLUMNS:
        columns[name] = Column(float, coerce=True, nullable=False)
    return DataFrameSchema(columns, strict=False)


RAW_SCHEMA = _build_schema()


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Return the coerced frame or raise ``pandera.errors.SchemaError``."""
    return RAW_SCHEMA.validate(df)
