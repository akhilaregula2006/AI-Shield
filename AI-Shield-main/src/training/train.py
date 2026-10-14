"""
Model training: XGBoost with class-imbalance weighting, MLflow experiment tracking,
and registration into the model registry (staged; promotion is a separate decision).
"""

from typing import Any, Dict, Optional

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from configs.settings import settings
from src.deployment import registry
from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.feature_engineering.features import FEATURE_COLUMNS, TARGET
from src.utils.logger import get_logger

logger = get_logger(__name__)
_REFERENCE_SAMPLE = 5000


def build_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=2,
        random_state=settings.random_seed,
    )


def train_model(
    features: pd.DataFrame, register: bool = True, tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Train on a train/val/test split; pick the threshold on val, report metrics on test."""
    x_all, y_all = features[FEATURE_COLUMNS], features[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=settings.test_size,
        stratify=y_all,
        random_state=settings.random_seed,
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train, random_state=settings.random_seed
    )

    positives = max(int(y_fit.sum()), 1)
    model = build_model(scale_pos_weight=(len(y_fit) - positives) / positives)
    model.fit(x_fit, y_fit)

    threshold = find_best_threshold(y_val, model.predict_proba(x_val)[:, 1])
    metrics = compute_metrics(y_test, model.predict_proba(x_test)[:, 1], threshold)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params({**model.get_params(), "n_train": len(x_fit), "threshold": threshold})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "threshold"})
        mlflow.xgboost.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    result: Dict[str, Any] = {
        "model": model,
        "metrics": metrics,
        "threshold": threshold,
        "run_id": run_id,
        "version": None,
    }
    if register:
        reference = x_fit.sample(
            min(_REFERENCE_SAMPLE, len(x_fit)), random_state=settings.random_seed
        )
        result["version"] = registry.register_model(
            model, metrics, threshold, reference.reset_index(drop=True), run_id, tags
        )
    logger.info("Trained model: pr_auc=%.4f recall=%.3f", metrics["pr_auc"], metrics["recall"])
    return result
