"""Advanced Streamlit dashboard for the mini-SIEM workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from attack_classification_pipeline import train_attack_classifier
from llm_threat_reasoning import LLMThreatReasoningError, run_structured_threat_reasoning
from risk_scoring_engine import calculate_risk_score


DARK_THEME_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #090d14 0%, #0c1220 45%, #0a101b 100%);
        color: #d3def4;
    }
    .main-card {
        background: rgba(21, 29, 44, 0.78);
        border: 1px solid rgba(104, 146, 214, 0.25);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .section-title {
        color: #90b4ff;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .risk-gauge-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .risk-gauge {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow: inset 0 0 0 10px rgba(10, 16, 27, 0.85), 0 8px 22px rgba(0,0,0,0.45);
    }
    .risk-gauge:after {
        content: '';
        width: 120px;
        height: 120px;
        background: #0b1220;
        border-radius: 50%;
        position: absolute;
    }
    .risk-gauge-value {
        position: relative;
        z-index: 2;
        font-size: 1.2rem;
        font-weight: 700;
        color: #dbe8ff;
        text-align: center;
    }
</style>
"""


@dataclass
class DashboardState:
    trained: bool = False
    metrics: dict[str, Any] | None = None
    confusion_matrix: list[list[int]] | None = None
    class_report: dict[str, Any] | None = None
    shap_importance: pd.DataFrame | None = None
    risk_output: dict[str, Any] | None = None


def init_page() -> None:
    st.set_page_config(
        page_title="Mini-SIEM Security Dashboard",
        page_icon="🛡️",
        layout="wide",
    )
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    st.title("🛡️ Mini-SIEM Security Operations Dashboard")
    st.caption("Upload data, evaluate model quality, inspect explainability, score risk, and generate AI threat narratives.")


def render_sidebar_upload() -> pd.DataFrame | None:
    st.sidebar.header("Dataset Upload")
    uploaded = st.sidebar.file_uploader(
        "Upload intrusion dataset (CSV)",
        type=["csv"],
        help="CSV should include engineered features plus the `label` column.",
    )

    if not uploaded:
        st.sidebar.info("Upload a dataset to start analysis.")
        return None

    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(df):,} records")
        return df
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to read CSV: {exc}")
        return None


def _derive_shap_like_importance(model_pipeline, feature_columns: list[str]) -> pd.DataFrame:
    classifier = model_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        return pd.DataFrame(
            {"feature": feature_columns, "importance": values}
        ).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_columns, "importance": [0.0] * len(feature_columns)})


def _risk_inputs_from_test_frame(test_frame: pd.DataFrame, predictions: pd.Series, probability_max: float) -> dict[str, Any]:
    anomaly_count = int((predictions != "benign").sum())
    critical_ports = {22, 3389, 445, 1433, 3306}
    critical_port_access = False
    if "port" in test_frame.columns:
        critical_port_access = bool(test_frame["port"].isin(critical_ports).any())

    baseline = float(test_frame.select_dtypes(include=["number"]).std().mean()) if not test_frame.empty else 0.0
    return {
        "ml_probability_score": float(probability_max),
        "anomalies_in_window": anomaly_count,
        "critical_port_access": critical_port_access,
        "historical_behavior_baseline": baseline,
    }


def run_training_workflow(df: pd.DataFrame) -> DashboardState:
    state = DashboardState()
    if "label" not in df.columns:
        st.error("Dataset must include a `label` column.")
        return state

    feature_columns = [col for col in df.columns if col != "label"]
    if not feature_columns:
        st.error("Dataset must include feature columns in addition to `label`.")
        return state

    with st.spinner("Training model and evaluating performance..."):
        result = train_attack_classifier(
            data=df,
            feature_columns=feature_columns,
            label_column="label",
            model_output_path="artifacts/dashboard_attack_classifier.joblib",
            scaler_output_path="artifacts/dashboard_scaler.joblib",
        )

        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_columns],
            df["label"],
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )
        _ = X_train, y_train

        model = result["model"]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

        shap_importance = _derive_shap_like_importance(model, feature_columns)

        risk_inputs = _risk_inputs_from_test_frame(
            test_frame=X_test,
            predictions=pd.Series(predictions),
            probability_max=float(probabilities.max()) if len(probabilities) else 0.0,
        )
        risk_output = calculate_risk_score(**risk_inputs)

    state.trained = True
    state.metrics = result["metrics"]
    state.confusion_matrix = cm.tolist()
    state.class_report = report
    state.shap_importance = shap_importance
    state.risk_output = risk_output
    return state


def render_metrics_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run analysis to view classification metrics.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{state.metrics['precision']:.3f}")
    c2.metric("Recall", f"{state.metrics['recall']:.3f}")
    c3.metric("F1-score", f"{state.metrics['f1_score']:.3f}")

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(state.class_report).transpose(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_confusion_matrix_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run analysis to visualize the confusion matrix.")
        return

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=pd.DataFrame(state.confusion_matrix).values)
    disp.plot(ax=ax, colorbar=False)
    st.pyplot(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)


def render_shap_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run analysis to inspect SHAP feature importance.")
        return

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>SHAP Feature Importance (Model Proxy)</div>", unsafe_allow_html=True)
    top_df = state.shap_importance.head(15)
    st.bar_chart(top_df.set_index("feature")["importance"])
    st.dataframe(top_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_risk_and_ai_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run analysis to view risk score and generate AI threat report.")
        return

    left, right = st.columns([1, 1])
    risk = state.risk_output
    score = float(risk["risk_score"])
    category = risk["risk_category"]

    left.markdown("<div class='main-card'>", unsafe_allow_html=True)
    left.markdown("<div class='section-title'>Risk Score Gauge</div>", unsafe_allow_html=True)

    if score < 25:
        color = "#25C281"
    elif score < 50:
        color = "#F6C445"
    elif score < 75:
        color = "#F08C2E"
    else:
        color = "#E35252"

    gauge_html = f"""
    <div class='risk-gauge-wrap'>
      <div class='risk-gauge' style='background: conic-gradient({color} {score}%, #2a3348 {score}% 100%);'>
        <div class='risk-gauge-value'>{score:.1f}<br/><span style='font-size:0.85rem;font-weight:500'>{category}</span></div>
      </div>
    </div>
    """
    left.markdown(gauge_html, unsafe_allow_html=True)
    left.json(risk)
    left.markdown("</div>", unsafe_allow_html=True)

    right.markdown("<div class='main-card'>", unsafe_allow_html=True)
    right.markdown("<div class='section-title'>AI Threat Report</div>", unsafe_allow_html=True)

    if right.button("Generate AI Threat Report", use_container_width=True):
        try:
            shap_payload = state.shap_importance.head(5).to_dict(orient="records")
            summary = (
                "Model indicates elevated malicious activity in uploaded sample. "
                "Prioritize accounts/endpoints tied to high-importance features."
            )
            response = run_structured_threat_reasoning(
                predicted_attack_type=category,
                shap_top_features=shap_payload,
                risk_score=score,
                aggregated_behavior_summary=summary,
            )
            right.success("Threat report generated.")
            right.code(json.dumps(response, indent=2), language="json")
        except LLMThreatReasoningError as exc:
            right.warning(f"Could not generate LLM report: {exc}")
        except Exception as exc:  # noqa: BLE001
            right.error(f"Unexpected error: {exc}")

    right.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_page()

    data = render_sidebar_upload()
    col_a, col_b = st.columns([1, 3])
    run_clicked = col_a.button("Run SIEM Analysis", use_container_width=True, disabled=data is None)
    col_b.caption("Tip: Upload a labeled intrusion dataset and click analysis to populate all tabs.")

    if "dashboard_state" not in st.session_state:
        st.session_state.dashboard_state = DashboardState()

    if run_clicked and data is not None:
        st.session_state.dashboard_state = run_training_workflow(data)

    tabs = st.tabs([
        "📈 Classification Metrics",
        "🧩 Confusion Matrix",
        "🔍 SHAP Importance",
        "⚠️ Risk & AI Report",
    ])

    with tabs[0]:
        render_metrics_tab(st.session_state.dashboard_state)
    with tabs[1]:
        render_confusion_matrix_tab(st.session_state.dashboard_state)
    with tabs[2]:
        render_shap_tab(st.session_state.dashboard_state)
    with tabs[3]:
        render_risk_and_ai_tab(st.session_state.dashboard_state)


if __name__ == "__main__":
    main()
