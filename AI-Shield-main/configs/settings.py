"""
Central configuration for AIShield.

Every module pulls config from here instead of reading os.environ directly.
This is the single source of truth for paths, model params, and service settings.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (configs/settings.py -> AIShield/)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App metadata ---
    app_name: str = "AIShield"
    environment: str = "dev"  # dev | staging | prod
    log_level: str = "INFO"

    # --- Paths ---
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    data_validated_dir: Path = ROOT_DIR / "data" / "validated"
    data_features_dir: Path = ROOT_DIR / "data" / "features"
    data_drift_dir: Path = ROOT_DIR / "data" / "drift"
    models_dir: Path = ROOT_DIR / "models"

    # --- Feature store (Phase 1: flat parquet file; swapped for Feast later) ---
    feature_store_path: Path = ROOT_DIR / "feature_store" / "features.parquet"

    # --- MLflow ---
    mlflow_tracking_uri: str = f"file://{ROOT_DIR / 'mlflow' / 'mlruns'}"
    mlflow_experiment_name: str = "aishield-fraud-detection"
    mlflow_registered_model_name: str = "aishield-fraud-model"

    # --- API ---
    api_host: str = "0.0.0.0"  # nosec B104
    api_port: int = 8000

    # --- Auth ---
    jwt_secret_key: str = "change-me-in-.env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    admin_username: str = "admin"
    admin_password: str = "change-me-admin"
    analyst_username: str = "analyst"
    analyst_password: str = "change-me-analyst"

    # --- Rate limiting ---
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # --- Data ---
    raw_data_filename: str = "creditcard.csv"
    synthetic_rows: int = 20000
    synthetic_fraud_rate: float = 0.02

    # --- Training ---
    random_seed: int = 42
    test_size: float = 0.2
    decision_threshold: float = 0.5
    promotion_min_gain: float = 0.0
    min_pr_auc: float = 0.5
    explanation_top_k: int = 5

    # --- Monitoring / drift / self-healing ---
    monitoring_buffer_size: int = 5000
    drift_psi_threshold: float = 0.2
    drift_ks_alpha: float = 0.01
    drift_feature_share_threshold: float = 0.15
    drift_min_samples: int = 500
    min_feedback_rows: int = 200
    drift_check_interval_seconds: int = 300  # 0 disables the background monitor
    auto_heal: bool = False  # if True the monitor also retrains/promotes without a human

    # --- LLM copilot (optional; falls back to rule-based answers when unset) ---
    anthropic_api_key: str = ""
    copilot_model: str = "claude-sonnet-4-6"


settings = Settings()
