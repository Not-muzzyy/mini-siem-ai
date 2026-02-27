"""LLM threat reasoning utilities for structured security analysis.

This module sends structured security context to an LLM and returns a strict
JSON response suitable for SIEM workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib import error, request


EXPECTED_RESPONSE_FIELDS = [
    "attack_classification_refinement",
    "threat_level",
    "technical_explanation",
    "business_impact",
    "recommended_response_steps",
    "long_term_mitigation",
]


class LLMThreatReasoningError(RuntimeError):
    """Raised when LLM threat reasoning fails."""


@dataclass(frozen=True)
class LLMReasoningConfig:
    """Configuration for LLM threat reasoning requests."""

    model: str = "gpt-4o-mini"
    timeout_seconds: int = 30
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    temperature: float = 0.1
    api_base_url: str = "https://api.openai.com/v1"


class LLMThreatReasoner:
    """Sends structured security analysis prompts to an LLM."""

    def __init__(self, config: LLMReasoningConfig | None = None) -> None:
        self.config = config or LLMReasoningConfig()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMThreatReasoningError(
                "Missing OPENAI_API_KEY environment variable."
            )

    def analyze(
        self,
        predicted_attack_type: str,
        shap_top_features: list[dict[str, Any]],
        risk_score: float,
        aggregated_behavior_summary: str,
    ) -> dict[str, str]:
        """Create LLM-driven threat reasoning JSON.

        Args:
            predicted_attack_type: Model-predicted attack type.
            shap_top_features: Top SHAP features with impact/context.
            risk_score: Numeric risk score from risk engine.
            aggregated_behavior_summary: Human-readable behavior summary.

        Returns:
            Dict with expected structured analysis fields.
        """
        payload = self._build_payload(
            predicted_attack_type=predicted_attack_type,
            shap_top_features=shap_top_features,
            risk_score=risk_score,
            aggregated_behavior_summary=aggregated_behavior_summary,
        )

        response_json = self._request_with_retry(payload)
        content = self._extract_content(response_json)
        parsed = self._parse_and_validate_output(content)
        return parsed

    def _build_payload(
        self,
        predicted_attack_type: str,
        shap_top_features: list[dict[str, Any]],
        risk_score: float,
        aggregated_behavior_summary: str,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a cybersecurity threat analysis assistant. "
            "Return only valid JSON with exactly these keys: "
            f"{', '.join(EXPECTED_RESPONSE_FIELDS)}. "
            "Be concise, technical, and actionable."
        )

        user_payload = {
            "predicted_attack_type": predicted_attack_type,
            "shap_top_features": shap_top_features,
            "risk_score": risk_score,
            "aggregated_behavior_summary": aggregated_behavior_summary,
            "response_schema": {
                key: "string" for key in EXPECTED_RESPONSE_FIELDS
            },
        }

        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze the following structured security context and "
                        "return the required JSON object.\n"
                        + json.dumps(user_payload, ensure_ascii=False)
                    ),
                },
            ],
        }

    def _request_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.config.api_base_url.rstrip('/')}/chat/completions"
        backoff = self.config.initial_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                request_body = json.dumps(payload).encode("utf-8")
                req = request.Request(
                    endpoint,
                    data=request_body,
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

                with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    response_text = resp.read().decode("utf-8")
                    return json.loads(response_text)

            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2

        raise LLMThreatReasoningError(
            f"LLM request failed after {self.config.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _extract_content(response_json: dict[str, Any]) -> str:
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMThreatReasoningError(
                f"Unexpected LLM response structure: {exc}"
            ) from exc

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return cleaned.strip()

    def _parse_and_validate_output(self, content: str) -> dict[str, str]:
        cleaned = self._strip_markdown_fences(content)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMThreatReasoningError(
                f"LLM output is not valid JSON: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMThreatReasoningError("LLM output JSON must be an object.")

        missing = [key for key in EXPECTED_RESPONSE_FIELDS if key not in parsed]
        if missing:
            raise LLMThreatReasoningError(
                f"LLM output missing required keys: {', '.join(missing)}"
            )

        normalized = {key: str(parsed.get(key, "")).strip() for key in EXPECTED_RESPONSE_FIELDS}
        return normalized


def run_structured_threat_reasoning(
    predicted_attack_type: str,
    shap_top_features: list[dict[str, Any]],
    risk_score: float,
    aggregated_behavior_summary: str,
    config: LLMReasoningConfig | None = None,
) -> dict[str, str]:
    """Convenience API for one-shot LLM threat reasoning."""
    reasoner = LLMThreatReasoner(config=config)
    return reasoner.analyze(
        predicted_attack_type=predicted_attack_type,
        shap_top_features=shap_top_features,
        risk_score=risk_score,
        aggregated_behavior_summary=aggregated_behavior_summary,
    )
