"""Prometheus instrumentation for the API, model, drift, and self-healing loop."""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "aishield_http_requests_total", "HTTP requests", ["method", "path", "status"]
)
PREDICTIONS = Counter("aishield_predictions_total", "Predictions served", ["outcome"])
PREDICTION_LATENCY = Histogram(
    "aishield_prediction_latency_seconds",
    "Latency of /predict handling",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
FRAUD_SCORE = Histogram(
    "aishield_fraud_score",
    "Distribution of fraud probabilities",
    buckets=[i / 10 for i in range(11)],
)
MODEL_VERSION = Gauge("aishield_model_version", "Production model version being served")
DRIFT_FEATURE_SHARE = Gauge("aishield_drift_feature_share", "Share of features with drift")
DRIFT_MAX_PSI = Gauge("aishield_drift_max_psi", "Largest per-feature PSI")
HEALING_ACTIONS = Counter(
    "aishield_self_healing_actions_total", "Self-healing decisions taken", ["action", "outcome"]
)
