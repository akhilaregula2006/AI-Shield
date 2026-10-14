"""Model evaluation metrics tuned for heavily imbalanced fraud data."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from configs.settings import settings


def find_best_threshold(y_true, proba) -> float:
    """Threshold maximizing F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return settings.decision_threshold
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true, proba, threshold: Optional[float] = None) -> Dict[str, float]:
    """PR-AUC (primary), ROC-AUC, and precision/recall/F1 at ``threshold``."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    threshold = settings.decision_threshold if threshold is None else threshold
    preds = (proba >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "pr_auc": 0.0 if single_class else float(average_precision_score(y_true, proba)),
        "roc_auc": 0.0 if single_class else float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }
