"""Feature engineering utilities for network log aggregation.

This module transforms raw network logs into source-IP-level features in
fixed 5-minute windows for downstream ML pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp",
    "source_ip",
    "destination_ip",
    "protocol",
    "port",
    "packet_size",
    "flag",
    "label",
}

FAILED_LOGIN_KEYWORDS = ("failed", "fail", "login_failed", "auth_fail")
SYN_KEYWORDS = ("syn",)


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration for network feature engineering."""

    window_size: str = "5min"
    connection_rate_denominator_seconds: int = 300


class NetworkFeatureEngineer:
    """Builds windowed per-source-IP features from raw network logs."""

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()

    def transform(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        """Aggregate logs into ML-ready features.

        Args:
            raw_logs: DataFrame with required network log columns.

        Returns:
            DataFrame aggregated by source IP and 5-minute window with
            engineered features.
        """
        validated = self._validate_input(raw_logs)
        prepared = self._prepare_columns(validated)
        grouped = prepared.groupby(
            [
                "source_ip",
                pd.Grouper(key="timestamp", freq=self.config.window_size),
            ],
            sort=True,
            dropna=False,
        )

        features = grouped.apply(self._build_group_features)
        features = features.reset_index().rename(columns={"timestamp": "window_start"})
        features["window_end"] = (
            features["window_start"]
            + pd.to_timedelta(self.config.window_size)
        )

        ordered_columns = [
            "source_ip",
            "window_start",
            "window_end",
            "failed_login_ratio",
            "connection_rate",
            "unique_ports_accessed",
            "avg_packet_size",
            "port_entropy",
            "inter_request_time_mean",
            "inter_request_time_std",
            "syn_flag_ratio",
            "label",
        ]
        return features[ordered_columns]

    def _validate_input(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        if raw_logs is None or raw_logs.empty:
            raise ValueError("raw_logs must be a non-empty DataFrame.")

        missing = REQUIRED_COLUMNS.difference(raw_logs.columns)
        if missing:
            missing_fmt = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns: {missing_fmt}")

        logs = raw_logs.copy()
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], utc=True, errors="coerce")
        if logs["timestamp"].isna().any():
            raise ValueError("timestamp column contains invalid datetime values.")

        logs["packet_size"] = pd.to_numeric(logs["packet_size"], errors="coerce")
        logs["port"] = pd.to_numeric(logs["port"], errors="coerce")
        if logs[["packet_size", "port"]].isna().any().any():
            raise ValueError("packet_size and port must be numeric.")

        return logs

    def _prepare_columns(self, logs: pd.DataFrame) -> pd.DataFrame:
        logs = logs.sort_values(["source_ip", "timestamp"]).copy()
        logs["is_failed_login"] = self._contains_keywords(logs["flag"], FAILED_LOGIN_KEYWORDS)
        logs["is_syn"] = self._contains_keywords(logs["flag"], SYN_KEYWORDS)
        return logs

    @staticmethod
    def _contains_keywords(series: pd.Series, keywords: Iterable[str]) -> pd.Series:
        pattern = "|".join(re.escape(keyword) for keyword in keywords)
        return series.astype(str).str.lower().str.contains(pattern, na=False)

    def _build_group_features(self, group: pd.DataFrame) -> pd.Series:
        timestamps = group["timestamp"].sort_values()
        inter_request = timestamps.diff().dt.total_seconds().dropna()

        port_counts = group["port"].value_counts(normalize=True)
        port_entropy = -np.sum(port_counts * np.log2(port_counts)) if not port_counts.empty else 0.0

        label_mode = group["label"].mode()
        group_label = label_mode.iloc[0] if not label_mode.empty else "unknown"

        return pd.Series(
            {
                "failed_login_ratio": group["is_failed_login"].mean(),
                "connection_rate": len(group) / self.config.connection_rate_denominator_seconds,
                "unique_ports_accessed": int(group["port"].nunique()),
                "avg_packet_size": float(group["packet_size"].mean()),
                "port_entropy": float(port_entropy),
                "inter_request_time_mean": float(inter_request.mean()) if not inter_request.empty else 0.0,
                "inter_request_time_std": float(inter_request.std(ddof=0)) if not inter_request.empty else 0.0,
                "syn_flag_ratio": group["is_syn"].mean(),
                "label": group_label,
            }
        )


def engineer_network_features(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for one-shot feature generation."""
    return NetworkFeatureEngineer().transform(raw_logs)
