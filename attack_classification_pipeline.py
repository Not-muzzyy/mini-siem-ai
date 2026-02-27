"""Supervised attack classification pipeline utilities.

This module provides training and inference functions for a network attack
classifier built with scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    scoring: str = "f1_weighted"
    n_jobs: int = -1


DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
}


def train_attack_classifier(
    data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "label",
    model_output_path: str | Path = "artifacts/attack_classifier.joblib",
    scaler_output_path: str | Path = "artifacts/attack_scaler.joblib",
    config: TrainingConfig | None = None,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train and evaluate an attack classifier with hyperparameter tuning.

    Args:
        data: Input DataFrame containing feature and label columns.
        feature_columns: Columns to use as model input features.
        label_column: Target column name.
        model_output_path: Path where the trained model pipeline is persisted.
        scaler_output_path: Path where the fitted scaler is persisted.
        config: Optional training configuration overrides.
        param_grid: Optional GridSearchCV parameter grid.

    Returns:
        Dictionary containing the trained pipeline, best params, evaluation,
        and artifact paths.
    """
    cfg = config or TrainingConfig()
    _validate_training_input(data, feature_columns, label_column)

    X = data[feature_columns].copy()
    y = data[label_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_columns)],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=cfg.random_state)),
        ]
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid or DEFAULT_PARAM_GRID,
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model: Pipeline = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    model_output_path = Path(model_output_path)
    scaler_output_path = Path(scaler_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)

    fitted_scaler = best_model.named_steps["preprocessor"].named_transformers_["num"]
    joblib.dump(best_model, model_output_path)
    joblib.dump(fitted_scaler, scaler_output_path)

    metrics = {
        "precision": report_dict["weighted avg"]["precision"],
        "recall": report_dict["weighted avg"]["recall"],
        "f1_score": report_dict["weighted avg"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    return {
        "model": best_model,
        "best_params": grid.best_params_,
        "metrics": metrics,
        "model_path": str(model_output_path),
        "scaler_path": str(scaler_output_path),
        "feature_columns": feature_columns,
        "label_column": label_column,
    }


def predict_attack(
    input_data: pd.DataFrame,
    model_path: str | Path = "artifacts/attack_classifier.joblib",
    return_probabilities: bool = True,
) -> pd.DataFrame:
    """Generate attack predictions using a persisted trained model.

    Args:
        input_data: DataFrame of feature rows expected by the trained model.
        model_path: Path to persisted model pipeline.
        return_probabilities: Whether to include class probabilities.

    Returns:
        DataFrame containing predicted labels and optional probabilities.
    """
    if input_data is None or input_data.empty:
        raise ValueError("input_data must be a non-empty DataFrame.")

    model = joblib.load(model_path)
    predictions = model.predict(input_data)

    output = pd.DataFrame({"predicted_label": predictions}, index=input_data.index)

    if return_probabilities and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)
        probability_columns = [f"proba_{cls}" for cls in model.classes_]
        proba_df = pd.DataFrame(probabilities, columns=probability_columns, index=input_data.index)
        output = pd.concat([output, proba_df], axis=1)

    return output


def _validate_training_input(data: pd.DataFrame, feature_columns: list[str], label_column: str) -> None:
    if data is None or data.empty:
        raise ValueError("data must be a non-empty DataFrame.")

    if not feature_columns:
        raise ValueError("feature_columns must contain at least one feature name.")

    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing_features))}")

    if label_column not in data.columns:
        raise ValueError(f"Missing label column: {label_column}")
