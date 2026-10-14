"""
Offline demo of the self-healing loop (no API needed):

    python3 -m scripts.run_pipeline        # train + promote the baseline model
    python3 -m scripts.simulate_drift      # drift -> detect -> retrain -> champion/challenger

Generates traffic from a shifted regime (new fraud pattern + covariate shift), shows the
champion degrading, detects drift, and lets the engine retrain and (if better) promote.
"""

import json

from src.deployment import registry
from src.evaluation.metrics import compute_metrics
from src.feature_engineering.features import FEATURE_COLUMNS, TARGET, engineer_features
from src.ingestion.loader import generate_synthetic_data
from src.monitoring.drift import detect_drift, save_report
from src.self_healing.engine import decide, heal


def main() -> None:
    champion = registry.load_model()
    print(
        f"Champion: v{champion.version}  PR-AUC (original data) = {champion.metrics['pr_auc']:.3f}"
    )

    live = engineer_features(
        generate_synthetic_data(n_rows=6000, seed=99, regime="B", covariate_shift=1.0)
    )
    degraded = compute_metrics(
        live[TARGET], champion.model.predict_proba(live[FEATURE_COLUMNS])[:, 1], champion.threshold
    )
    print(
        "Champion on drifted traffic:    "
        f"PR-AUC = {degraded['pr_auc']:.3f}  recall = {degraded['recall']:.3f}"
    )

    report = detect_drift(champion.reference, live)
    save_report(report)
    action = decide(report)
    print(
        f"Drift: {len(report.drifted_features)} features ({report.drift_share:.0%}), "
        f"max PSI {report.max_psi:.2f} -> decision: {action.value.upper()}"
    )
    if action.value != "retrain":
        return

    outcome = heal(live)
    print(json.dumps(outcome.to_dict(), indent=2))
    print(f"Production is now v{registry.get_production_version()}")


if __name__ == "__main__":
    main()
