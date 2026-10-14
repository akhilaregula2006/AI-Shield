"""In-process model service: holds the production model + explainer and hot-swaps on reload."""

import threading
from typing import Any, Dict, List, Optional

import pandas as pd

from src.deployment import registry
from src.explainability.shap_explainer import ShapExplainer
from src.feature_engineering.features import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: Optional[registry.LoadedModel] = None
        self._explainer: Optional[ShapExplainer] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def version(self) -> Optional[int]:
        return self._loaded.version if self._loaded else None

    @property
    def reference(self) -> Optional[pd.DataFrame]:
        return self._loaded.reference if self._loaded else None

    def load(self) -> bool:
        """Load the production model; returns False if none is registered yet."""
        try:
            loaded = registry.load_model()
        except LookupError:
            logger.warning("No production model registered; /predict will return 503")
            return False
        with self._lock:
            self._loaded, self._explainer = loaded, None  # explainer rebuilt lazily
        logger.info("Serving model v%d", loaded.version)
        return True

    def predict(
        self, features: pd.DataFrame, explain: bool = False, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        with self._lock:
            loaded = self._loaded
            if loaded is None:
                raise RuntimeError("Model not loaded")
            if explain and self._explainer is None:
                self._explainer = ShapExplainer(loaded.model)
            explainer = self._explainer
        x = features[FEATURE_COLUMNS]
        scores = loaded.model.predict_proba(x)[:, 1]
        explanations = explainer.explain(x, top_k) if explain and explainer else None
        results = []
        for i, score in enumerate(scores):
            item: Dict[str, Any] = {
                "fraud_probability": float(score),
                "is_fraud": bool(score >= loaded.threshold),
                "threshold": loaded.threshold,
                "model_version": loaded.version,
            }
            if explanations is not None:
                item["explanation"] = explanations[i]
            results.append(item)
        return results


model_service = ModelService()
