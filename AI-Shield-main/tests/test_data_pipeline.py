import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.feature_engineering.features import FEATURE_COLUMNS, engineer_features
from src.feature_store import store
from src.ingestion.loader import RAW_COLUMNS, generate_synthetic_data, load_raw_data
from src.preprocessing.cleaning import clean
from src.validation.schema import validate_raw


def test_synthetic_data_matches_ulb_schema():
    df = generate_synthetic_data(n_rows=2000, fraud_rate=0.05, seed=0)
    assert list(df.columns) == RAW_COLUMNS
    assert 80 <= df["Class"].sum() <= 120


def test_synthetic_data_is_deterministic():
    a = generate_synthetic_data(n_rows=500, seed=7)
    b = generate_synthetic_data(n_rows=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_unknown_regime_rejected():
    with pytest.raises(ValueError):
        generate_synthetic_data(n_rows=100, regime="Z")


def test_load_raw_data_falls_back_to_synthetic(tmp_path):
    df = load_raw_data(tmp_path / "missing.csv")
    assert list(df.columns) == RAW_COLUMNS


def test_load_raw_data_reads_csv(tmp_path):
    path = tmp_path / "creditcard.csv"
    generate_synthetic_data(n_rows=300, seed=1).to_csv(path, index=False)
    assert len(load_raw_data(path)) == 300


def test_validation_accepts_good_data():
    assert len(validate_raw(generate_synthetic_data(n_rows=300, seed=1))) == 300


def test_validation_rejects_negative_amount():
    df = generate_synthetic_data(n_rows=300, seed=1)
    df.loc[0, "Amount"] = -5.0
    with pytest.raises(SchemaError):
        validate_raw(df)


def test_validation_rejects_missing_column_and_bad_label():
    df = generate_synthetic_data(n_rows=300, seed=1)
    with pytest.raises(SchemaError):
        validate_raw(df.drop(columns=["V3"]))
    df.loc[0, "Class"] = 2
    with pytest.raises(SchemaError):
        validate_raw(df)


def test_clean_drops_duplicates():
    df = generate_synthetic_data(n_rows=100, seed=1)
    assert len(clean(pd.concat([df, df.iloc[:10]]))) == 100


def test_engineer_features_shape_and_values():
    raw = generate_synthetic_data(n_rows=200, seed=1)
    feats = engineer_features(raw)
    assert list(feats.columns) == [*FEATURE_COLUMNS, "Class"]
    assert np.allclose(feats["log_amount"], np.log1p(raw["Amount"]))
    assert np.allclose(feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2, 1.0)


def test_engineer_features_without_label_for_serving():
    raw = generate_synthetic_data(n_rows=50, seed=1).drop(columns=["Class"])
    assert "Class" not in engineer_features(raw).columns


def test_feature_store_roundtrip_and_versioning():
    feats = engineer_features(generate_synthetic_data(n_rows=400, seed=1))
    m1 = store.write_features(feats, source="test")
    m2 = store.append_features(feats.iloc[:50], source="append")
    assert (m1["version"], m2["version"]) == (1, 2)
    assert m2["rows"] == 450 and len(m2["history"]) == 1
    assert len(store.read_features()) == 450


def test_feature_store_detects_tampering():
    feats = engineer_features(generate_synthetic_data(n_rows=400, seed=1))
    store.write_features(feats)
    with open(store.settings.feature_store_path, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        store.read_features()


def test_feature_store_missing_file_message():
    with pytest.raises(FileNotFoundError):
        store.read_features()
