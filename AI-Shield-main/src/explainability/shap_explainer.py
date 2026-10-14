"""Per-prediction SHAP explanations for the tree model."""

from typing import Any, Dict, List

import pandas as pd


class ShapExplainer:
    """Wraps ``shap.TreeExplainer``; imported lazily because shap is slow to import."""

    def __init__(self, model: Any) -> None:
        import shap  # pylint: disable=import-outside-toplevel

        self._explainer = shap.TreeExplainer(model)

    def explain(self, features: pd.DataFrame, top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Top-k features by |SHAP| for each row; positive contribution pushes toward fraud."""
        values = self._explainer.shap_values(features)
        if isinstance(values, list):  # older shap versions return one array per class
            values = values[-1]
        explanations = []
        for row_idx in range(len(features)):
            order = abs(values[row_idx]).argsort()[::-1][:top_k]
            explanations.append(
                [
                    {
                        "feature": str(features.columns[i]),
                        "value": float(features.iloc[row_idx, i]),
                        "contribution": float(values[row_idx][i]),
                    }
                    for i in order
                ]
            )
        return explanations
