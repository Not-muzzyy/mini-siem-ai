"""SHAP explainability utilities for trained attack classification models.

This module integrates SHAP with persisted scikit-learn models (including
RandomForest-based pipelines) to produce prediction-level explanations,
top feature drivers, and visualization-ready outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


@dataclass(frozen=True)
class ShapExplainabilityConfig:
    """Configuration for SHAP-based explanation behavior."""

    top_n_features: int = 5
    check_additivity: bool = False


class AttackShapExplainer:
    """Reusable SHAP explainer for attack classification pipelines."""

    def __init__(
        self,
        model_path: str | Path,
        feature_names: list[str],
        config: ShapExplainabilityConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_names = feature_names
        self.config = config or ShapExplainabilityConfig()

        self._pipeline = self._load_pipeline(self.model_path)
        self._classifier = self._extract_classifier(self._pipeline)

    def explain(self, input_data: pd.DataFrame) -> dict[str, Any]:
        """Generate SHAP explanations for model predictions.

        Args:
            input_data: DataFrame containing features expected by the model.

        Returns:
            A dictionary with:
                - predictions: list of predicted class labels
                - top_contributors: per-row top feature contributors
                - explanation_summary: human-readable summaries per row
                - visualization_data: structured arrays for downstream plotting
        """
        validated = self._validate_input(input_data)

        transformed_X = self._pipeline.named_steps["preprocessor"].transform(validated)
        explainer = shap.TreeExplainer(self._classifier)
        shap_values_raw = explainer.shap_values(
            transformed_X,
            check_additivity=self.config.check_additivity,
        )

        predictions = self._pipeline.predict(validated)
        probabilities = self._pipeline.predict_proba(validated)
        classes = self._classifier.classes_

        shap_values_by_class = self._normalize_shap_values(shap_values_raw, classes)
        result_rows = []
        top_contributors_rows = []

        for idx, predicted_label in enumerate(predictions):
            class_index = int(np.where(classes == predicted_label)[0][0])
            class_shap = shap_values_by_class[class_index][idx]
            feature_impacts = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "shap_value": class_shap,
                    "abs_shap_value": np.abs(class_shap),
                }
            ).sort_values("abs_shap_value", ascending=False)

            top_features = feature_impacts.head(self.config.top_n_features)
            top_contributors = [
                {
                    "feature": row.feature,
                    "impact": float(row.shap_value),
                    "direction": "increases_risk" if row.shap_value > 0 else "decreases_risk",
                }
                for row in top_features.itertuples(index=False)
            ]

            class_probability = float(probabilities[idx, class_index])
            summary_text = self._build_summary(
                predicted_label=str(predicted_label),
                class_probability=class_probability,
                top_contributors=top_contributors,
            )

            result_rows.append(
                {
                    "row_index": int(validated.index[idx]) if np.issubdtype(type(validated.index[idx]), np.integer) else idx,
                    "predicted_label": str(predicted_label),
                    "predicted_probability": class_probability,
                    "explanation_summary": summary_text,
                }
            )

            for contributor in top_contributors:
                top_contributors_rows.append(
                    {
                        "row_index": result_rows[-1]["row_index"],
                        "predicted_label": str(predicted_label),
                        **contributor,
                    }
                )

        explanation_df = pd.DataFrame(result_rows)
        top_contributors_df = pd.DataFrame(top_contributors_rows)

        visualization_data = {
            "feature_names": self.feature_names,
            "classes": [str(cls) for cls in classes],
            "predictions": [str(label) for label in predictions],
            "predicted_probabilities": probabilities.tolist(),
            "shap_values": {
                str(classes[i]): shap_values_by_class[i].tolist()
                for i in range(len(classes))
            },
            "input_data": validated.to_dict(orient="records"),
        }

        return {
            "predictions": explanation_df,
            "top_contributors": top_contributors_df,
            "explanation_summary": explanation_df[["row_index", "explanation_summary"]],
            "visualization_data": visualization_data,
        }

    @staticmethod
    def _load_pipeline(model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)

    @staticmethod
    def _extract_classifier(pipeline):
        if not hasattr(pipeline, "named_steps") or "classifier" not in pipeline.named_steps:
            raise ValueError("Model must be a pipeline containing a 'classifier' step.")
        classifier = pipeline.named_steps["classifier"]
        if classifier.__class__.__name__ != "RandomForestClassifier":
            raise ValueError("Classifier must be RandomForestClassifier for TreeExplainer usage.")
        return classifier

    def _validate_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        if input_data is None or input_data.empty:
            raise ValueError("input_data must be a non-empty DataFrame.")

        missing_features = [feature for feature in self.feature_names if feature not in input_data.columns]
        if missing_features:
            raise ValueError(f"Missing expected feature columns: {', '.join(sorted(missing_features))}")

        return input_data[self.feature_names].copy()

    @staticmethod
    def _normalize_shap_values(shap_values_raw: Any, classes: np.ndarray) -> list[np.ndarray]:
        """Normalize SHAP output format across SHAP versions.

        TreeExplainer may return:
        - list[np.ndarray], one per class
        - np.ndarray with shape (n_samples, n_features) for binary class margin
        - np.ndarray with shape (n_samples, n_features, n_classes)
        """
        if isinstance(shap_values_raw, list):
            return [np.asarray(values) for values in shap_values_raw]

        shap_values_arr = np.asarray(shap_values_raw)

        if shap_values_arr.ndim == 2:
            return [shap_values_arr for _ in classes]

        if shap_values_arr.ndim == 3:
            return [shap_values_arr[:, :, i] for i in range(shap_values_arr.shape[2])]

        raise ValueError("Unsupported SHAP output shape encountered.")

    @staticmethod
    def _build_summary(
        predicted_label: str,
        class_probability: float,
        top_contributors: list[dict[str, Any]],
    ) -> str:
        if not top_contributors:
            return (
                f"Predicted class '{predicted_label}' with probability "
                f"{class_probability:.3f}. No SHAP contributors were identified."
            )

        drivers = ", ".join(
            f"{item['feature']} ({item['impact']:+.4f}, {item['direction']})"
            for item in top_contributors
        )
        return (
            f"Predicted class '{predicted_label}' with probability {class_probability:.3f}. "
            f"Top contributing features: {drivers}."
        )


def explain_attack_predictions_with_shap(
    input_data: pd.DataFrame,
    model_path: str | Path,
    feature_names: list[str],
    config: ShapExplainabilityConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for one-shot SHAP explanation generation."""
    explainer = AttackShapExplainer(
        model_path=model_path,
        feature_names=feature_names,
        config=config,
    )
    return explainer.explain(input_data)
