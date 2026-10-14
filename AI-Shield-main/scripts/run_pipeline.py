"""
End-to-end batch pipeline: ingest -> validate -> clean -> features -> feature store -> train ->
register (-> promote if no production model exists yet).

    python3 -m scripts.run_pipeline
"""

from configs.settings import settings
from src.deployment import registry
from src.feature_engineering.features import engineer_features
from src.feature_store import store
from src.ingestion.loader import load_raw_data
from src.preprocessing.cleaning import clean
from src.training.train import train_model
from src.utils.logger import get_logger
from src.validation.schema import validate_raw

logger = get_logger("pipeline")


def main() -> None:
    raw = clean(validate_raw(load_raw_data()))
    settings.data_validated_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(settings.data_validated_dir / "validated.parquet", index=False)

    features = engineer_features(raw)
    manifest = store.write_features(features, source="run_pipeline")

    result = train_model(
        features, tags={"trigger": "pipeline", "feature_store": str(manifest["version"])}
    )
    if registry.get_production_version() is None:
        registry.promote(result["version"], reason="initial model")
    logger.info("Pipeline done. Model v%s metrics: %s", result["version"], result["metrics"])


if __name__ == "__main__":
    main()
