"""Risk scoring engine for mini-SIEM detections.

This module computes a weighted risk score (0-100) from model output and
behavioral indicators, then maps the score to a risk category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskWeights:
    """Configurable weights for each risk signal.

    All weights should be non-negative and sum to 1.0.
    """

    ml_probability: float = 0.45
    anomaly_count: float = 0.25
    critical_port_access: float = 0.15
    baseline_deviation: float = 0.15


@dataclass(frozen=True)
class RiskScoringConfig:
    """Configuration for risk scoring and categorization."""

    weights: RiskWeights = RiskWeights()
    anomaly_count_cap: int = 20
    baseline_deviation_cap: float = 3.0


class RiskScoringError(ValueError):
    """Raised when risk scoring input/configuration is invalid."""


class RiskScorer:
    """Weighted risk scoring engine."""

    def __init__(self, config: RiskScoringConfig | None = None) -> None:
        self.config = config or RiskScoringConfig()
        self._validate_config(self.config)

    def score(
        self,
        ml_probability_score: float,
        anomalies_in_window: int,
        critical_port_access: bool,
        historical_behavior_baseline: float,
    ) -> dict[str, Any]:
        """Compute risk score and risk category.

        Args:
            ml_probability_score: Model probability in range [0, 1].
            anomalies_in_window: Count of anomalies in active time window.
            critical_port_access: Whether critical ports were accessed.
            historical_behavior_baseline: Baseline deviation indicator where
                larger positive values indicate stronger deviation.

        Returns:
            Dictionary with risk score (0-100), category, and component detail.
        """
        self._validate_inputs(
            ml_probability_score=ml_probability_score,
            anomalies_in_window=anomalies_in_window,
            historical_behavior_baseline=historical_behavior_baseline,
        )

        normalized_ml = self._clamp(ml_probability_score, 0.0, 1.0)
        normalized_anomalies = self._normalize_anomaly_count(anomalies_in_window)
        normalized_critical_port = 1.0 if critical_port_access else 0.0
        normalized_baseline = self._normalize_baseline_deviation(historical_behavior_baseline)

        weights = self.config.weights
        weighted_sum = (
            weights.ml_probability * normalized_ml
            + weights.anomaly_count * normalized_anomalies
            + weights.critical_port_access * normalized_critical_port
            + weights.baseline_deviation * normalized_baseline
        )

        risk_score = round(weighted_sum * 100.0, 2)
        risk_category = self._map_risk_category(risk_score)

        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "components": {
                "ml_probability_score_normalized": normalized_ml,
                "anomalies_in_window_normalized": normalized_anomalies,
                "critical_port_access_normalized": normalized_critical_port,
                "historical_behavior_baseline_normalized": normalized_baseline,
            },
            "weights": {
                "ml_probability": weights.ml_probability,
                "anomaly_count": weights.anomaly_count,
                "critical_port_access": weights.critical_port_access,
                "baseline_deviation": weights.baseline_deviation,
            },
        }

    @staticmethod
    def _validate_config(config: RiskScoringConfig) -> None:
        weights = config.weights
        weight_values = [
            weights.ml_probability,
            weights.anomaly_count,
            weights.critical_port_access,
            weights.baseline_deviation,
        ]

        if any(weight < 0 for weight in weight_values):
            raise RiskScoringError("All weights must be non-negative.")

        total = sum(weight_values)
        if abs(total - 1.0) > 1e-9:
            raise RiskScoringError("Weights must sum to 1.0.")

        if config.anomaly_count_cap <= 0:
            raise RiskScoringError("anomaly_count_cap must be > 0.")

        if config.baseline_deviation_cap <= 0:
            raise RiskScoringError("baseline_deviation_cap must be > 0.")

    @staticmethod
    def _validate_inputs(
        ml_probability_score: float,
        anomalies_in_window: int,
        historical_behavior_baseline: float,
    ) -> None:
        if not 0.0 <= ml_probability_score <= 1.0:
            raise RiskScoringError("ml_probability_score must be between 0 and 1.")

        if anomalies_in_window < 0:
            raise RiskScoringError("anomalies_in_window must be >= 0.")

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))

    def _normalize_anomaly_count(self, anomalies_in_window: int) -> float:
        capped = min(anomalies_in_window, self.config.anomaly_count_cap)
        return capped / self.config.anomaly_count_cap

    def _normalize_baseline_deviation(self, historical_behavior_baseline: float) -> float:
        # Negative values indicate below-baseline behavior and contribute 0 risk.
        positive_deviation = max(0.0, historical_behavior_baseline)
        capped = min(positive_deviation, self.config.baseline_deviation_cap)
        return capped / self.config.baseline_deviation_cap

    @staticmethod
    def _map_risk_category(risk_score: float) -> str:
        if risk_score < 25:
            return "Low"
        if risk_score < 50:
            return "Medium"
        if risk_score < 75:
            return "High"
        return "Critical"


def calculate_risk_score(
    ml_probability_score: float,
    anomalies_in_window: int,
    critical_port_access: bool,
    historical_behavior_baseline: float,
    config: RiskScoringConfig | None = None,
) -> dict[str, Any]:
    """Convenience API for one-shot risk scoring."""
    scorer = RiskScorer(config=config)
    return scorer.score(
        ml_probability_score=ml_probability_score,
        anomalies_in_window=anomalies_in_window,
        critical_port_access=critical_port_access,
        historical_behavior_baseline=historical_behavior_baseline,
    )
