"""Advanced Streamlit dashboard for the mini-SIEM workflow.

Fixes applied vs original:
- Real SHAP TreeExplainer connected (not feature_importances_ proxy)
- Feature engineering pipeline integrated before training
- Sample dataset auto-loaded for demo
- Improved error handling throughout
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from attack_classification_pipeline import train_attack_classifier
from network_feature_engineering import engineer_network_features
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
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .section-title {
        color: #90b4ff;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        margin-bottom: 0.5rem;
    }
    .risk-gauge-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
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
    .stat-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.1rem;
    }
</style>
"""

FEATURE_COLUMNS = [
    "failed_login_ratio",
    "connection_rate",
    "unique_ports_accessed",
    "avg_packet_size",
    "port_entropy",
    "inter_request_time_mean",
    "inter_request_time_std",
    "syn_flag_ratio",
]

SAMPLE_DATASET_PATH = Path("data/sample_intrusion_dataset.csv")


@dataclass
class DashboardState:
    trained: bool = False
    metrics: dict[str, Any] | None = None
    confusion_matrix: list[list[int]] | None = None
    class_report: dict[str, Any] | None = None
    shap_importance: pd.DataFrame | None = None
    shap_summary_fig: Any | None = None
    risk_output: dict[str, Any] | None = None
    attack_counts: dict[str, int] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    model_path: str = "artifacts/dashboard_attack_classifier.joblib"


def init_page() -> None:
    st.set_page_config(
        page_title="Mini-SIEM Security Dashboard",
        page_icon="🛡️",
        layout="wide",
    )
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    st.title("🛡️ Mini-SIEM Security Operations Dashboard")
    st.caption(
        "AI-powered intrusion detection — upload network logs, train the classifier, "
        "inspect SHAP explainability, score risk, and generate LLM threat reports."
    )


def render_sidebar_upload() -> pd.DataFrame | None:
    st.sidebar.header("📂 Dataset")
    use_sample = st.sidebar.checkbox("Use sample dataset", value=True)
    if use_sample and SAMPLE_DATASET_PATH.exists():
        df = pd.read_csv(SAMPLE_DATASET_PATH)
        st.sidebar.success(f"✅ Sample dataset: {len(df):,} records")
        st.sidebar.caption("Contains: benign, brute_force, scan, ddos, c2, data_exfiltration")
        return df
    uploaded = st.sidebar.file_uploader(
        "Upload raw network logs (CSV)",
        type=["csv"],
        help="Must contain: timestamp, source_ip, destination_ip, protocol, port, packet_size, flag, label",
    )
    if not uploaded:
        st.sidebar.info("Upload a CSV or enable sample dataset above.")
        return None
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"✅ Loaded {len(df):,} records")
        return df
    except Exception as exc:
        st.sidebar.error(f"Failed to read CSV: {exc}")
        return None


def _compute_real_shap(
    model_pipeline,
    X_test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, Any]:
    try:
        classifier = model_pipeline.named_steps["classifier"]
        preprocessor = model_pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X_test)
        explainer = shap.TreeExplainer(classifier)
        shap_values_raw = explainer.shap_values(X_transformed, check_additivity=False)
        if isinstance(shap_values_raw, list):
            shap_array = np.array(shap_values_raw)
            mean_abs = np.abs(shap_array).mean(axis=(0, 1))
        elif np.asarray(shap_values_raw).ndim == 3:
            mean_abs = np.abs(shap_values_raw).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values_raw).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": mean_abs,
        }).sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0c1220")
        ax.set_facecolor("#0c1220")
        colors = ["#4e8ef7" if v > 0 else "#e35252" for v in importance_df["importance"]]
        ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP Value|", color="#90b4ff")
        ax.set_title("Feature Importance (Real SHAP)", color="#90b4ff", fontweight="bold")
        ax.tick_params(colors="#d3def4")
        ax.spines[:].set_color("#2a3348")
        plt.tight_layout()
        return importance_df, fig
    except Exception as exc:
        st.warning(f"SHAP computation fell back to feature_importances_: {exc}")
        classifier = model_pipeline.named_steps["classifier"]
        importance_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": classifier.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance_df, None


def _risk_inputs_from_predictions(X_test, predictions, probability_max):
    anomaly_count = int((predictions != "benign").sum())
    critical_ports = {22, 3389, 445, 1433, 3306, 4444, 1337, 8443}
    critical_port_access = False
    if "port" in X_test.columns:
        critical_port_access = bool(X_test["port"].isin(critical_ports).any())
    baseline = float(X_test.select_dtypes(include=["number"]).std().mean()) if not X_test.empty else 0.0
    return {
        "ml_probability_score": float(np.clip(probability_max, 0.0, 1.0)),
        "anomalies_in_window": anomaly_count,
        "critical_port_access": critical_port_access,
        "historical_behavior_baseline": float(np.clip(baseline, 0.0, 3.0)),
    }


def _needs_feature_engineering(df: pd.DataFrame) -> bool:
    raw_log_cols = {"timestamp", "source_ip", "destination_ip", "protocol", "packet_size", "flag"}
    return raw_log_cols.issubset(df.columns)


def run_training_workflow(df: pd.DataFrame) -> DashboardState:
    state = DashboardState()
    with st.spinner("🔧 Preparing features..."):
        if _needs_feature_engineering(df):
            try:
                engineered = engineer_network_features(df)
            except Exception as exc:
                st.error(f"Feature engineering failed: {exc}")
                return state
        else:
            engineered = df.copy()
        if "label" not in engineered.columns:
            st.error("Dataset must include a `label` column.")
            return state
        feature_columns = [col for col in engineered.columns if col not in ("label", "source_ip", "window_start", "window_end")]
        if not feature_columns:
            st.error("No feature columns found.")
            return state
        engineered = engineered.dropna(subset=feature_columns + ["label"])
    attack_counts = engineered["label"].value_counts().to_dict()
    with st.spinner("🤖 Training RandomForest classifier with hyperparameter tuning..."):
        try:
            result = train_attack_classifier(
                data=engineered,
                feature_columns=feature_columns,
                label_column="label",
                model_output_path="artifacts/dashboard_attack_classifier.joblib",
                scaler_output_path="artifacts/dashboard_scaler.joblib",
            )
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return state
    with st.spinner("📊 Evaluating on held-out test set..."):
        X_train, X_test, y_train, y_test = train_test_split(
            engineered[feature_columns], engineered["label"],
            test_size=0.2, random_state=42, stratify=engineered["label"],
        )
        model = result["model"]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    with st.spinner("🔍 Computing real SHAP explanations..."):
        shap_importance, shap_fig = _compute_real_shap(model, X_test, feature_columns)
    with st.spinner("⚠️ Calculating risk score..."):
        risk_inputs = _risk_inputs_from_predictions(X_test, predictions, float(probabilities.max()))
        risk_output = calculate_risk_score(**risk_inputs)
    state.trained = True
    state.metrics = result["metrics"]
    state.confusion_matrix = cm.tolist()
    state.class_report = report
    state.shap_importance = shap_importance
    state.shap_summary_fig = shap_fig
    state.risk_output = risk_output
    state.attack_counts = attack_counts
    state.feature_columns = feature_columns
    state.model_path = "artifacts/dashboard_attack_classifier.joblib"
    return state


def render_overview_banner(state: DashboardState) -> None:
    if not state.trained:
        return
    cols = st.columns(6)
    attack_labels = {
        "benign": ("✅", "#25C281"),
        "brute_force": ("🔑", "#F6C445"),
        "scan": ("🔭", "#4e8ef7"),
        "ddos": ("💥", "#F08C2E"),
        "c2": ("🕹️", "#c97bff"),
        "data_exfiltration": ("📤", "#E35252"),
    }
    for i, (label, count) in enumerate(state.attack_counts.items()):
        icon, color = attack_labels.get(label, ("⚠️", "#aaa"))
        cols[i % 6].markdown(
            f"<div class='main-card' style='text-align:center'>"
            f"<div style='font-size:1.6rem'>{icon}</div>"
            f"<div style='color:{color};font-weight:700;font-size:1.1rem'>{count}</div>"
            f"<div style='font-size:0.75rem;color:#8899bb'>{label.replace('_',' ').title()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_metrics_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run SIEM Analysis to view classification metrics.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Precision", f"{state.metrics['precision']:.3f}")
    c2.metric("📡 Recall", f"{state.metrics['recall']:.3f}")
    c3.metric("⚡ F1-Score", f"{state.metrics['f1_score']:.3f}")
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Per-Class Classification Report</div>", unsafe_allow_html=True)
    report_df = pd.DataFrame(state.class_report).transpose()
    st.dataframe(
        report_df.style.background_gradient(subset=["precision", "recall", "f1-score"], cmap="Blues"),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_confusion_matrix_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run SIEM Analysis to visualize the confusion matrix.")
        return
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Confusion Matrix — Predicted vs Actual</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0c1220")
    ax.set_facecolor("#0c1220")
    cm_array = np.array(state.confusion_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_array)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.tick_params(colors="#d3def4", labelsize=8)
    ax.set_xlabel("Predicted Label", color="#90b4ff")
    ax.set_ylabel("True Label", color="#90b4ff")
    ax.title.set_color("#90b4ff")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_shap_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run SIEM Analysis to inspect SHAP feature importance.")
        return
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Real SHAP Feature Importance</div>", unsafe_allow_html=True)
    st.caption("Shows which features drive the model's attack/benign decisions most strongly.")
    if state.shap_summary_fig is not None:
        st.pyplot(state.shap_summary_fig, use_container_width=True)
    else:
        st.bar_chart(state.shap_importance.set_index("feature")["importance"])
    st.markdown("**Top Feature Drivers (ranked)**")
    top_df = state.shap_importance.copy()
    top_df["rank"] = range(1, len(top_df) + 1)
    st.dataframe(top_df[["rank", "feature", "importance"]], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📖 What Each Feature Means</div>", unsafe_allow_html=True)
    feature_explanations = {
        "failed_login_ratio": "Ratio of failed login attempts → high = brute force 🔑",
        "connection_rate": "Connections per second → very high = DDoS 💥",
        "unique_ports_accessed": "Number of distinct ports touched → high = port scan 🔭",
        "avg_packet_size": "Average bytes per packet → tiny = flood, huge = exfiltration",
        "port_entropy": "Randomness of port usage → high entropy = scanning behavior",
        "inter_request_time_mean": "Average time between requests → very regular = C2 beacon 🕹️",
        "inter_request_time_std": "Consistency of timing → near-zero std = malware beacon",
        "syn_flag_ratio": "Ratio of SYN packets → near 1.0 = SYN flood attack",
    }
    for feat, explanation in feature_explanations.items():
        if feat in state.feature_columns:
            st.markdown(f"**`{feat}`** — {explanation}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_risk_and_ai_tab(state: DashboardState) -> None:
    if not state.trained:
        st.info("Run SIEM Analysis to view risk score and generate AI threat report.")
        return
    left, right = st.columns([1, 1])
    risk = state.risk_output
    score = float(risk["risk_score"])
    category = risk["risk_category"]
    left.markdown("<div class='main-card'>", unsafe_allow_html=True)
    left.markdown("<div class='section-title'>⚠️ Composite Risk Score</div>", unsafe_allow_html=True)
    color_map = {"Low": "#25C281", "Medium": "#F6C445", "High": "#F08C2E", "Critical": "#E35252"}
    color = color_map.get(category, "#4e8ef7")
    gauge_html = f"""
    <div class='risk-gauge-wrap'>
      <div class='risk-gauge' style='background: conic-gradient({color} {score}%, #2a3348 {score}% 100%);'>
        <div class='risk-gauge-value'>
          {score:.1f}<br/>
          <span style='font-size:0.85rem;font-weight:500;color:{color}'>{category}</span>
        </div>
      </div>
    </div>
    """
    left.markdown(gauge_html, unsafe_allow_html=True)
    left.markdown("**Score Components**")
    components = risk.get("components", {})
    comp_df = pd.DataFrame([
        {"Signal": k.replace("_normalized", "").replace("_", " ").title(), "Value": f"{v:.3f}"}
        for k, v in components.items()
    ])
    left.dataframe(comp_df, use_container_width=True, hide_index=True)
    left.markdown("</div>", unsafe_allow_html=True)
    right.markdown("<div class='main-card'>", unsafe_allow_html=True)
    right.markdown("<div class='section-title'>🤖 AI Threat Report (LLM-Powered)</div>", unsafe_allow_html=True)
    right.caption("Requires OPENAI_API_KEY environment variable to be set.")
    if right.button("🚀 Generate AI Threat Report", use_container_width=True):
        top_shap = state.shap_importance.head(5).to_dict(orient="records")
        behavior_summary = (
            f"Model detected elevated malicious activity. "
            f"Risk category: {category} (score: {score:.1f}/100). "
            f"Top indicator: {state.shap_importance.iloc[0]['feature']} "
            f"with importance {state.shap_importance.iloc[0]['importance']:.4f}. "
            f"Attack distribution: {json.dumps({k: v for k, v in list(state.attack_counts.items())[:4]})}."
        )
        try:
            with st.spinner("Consulting AI threat analyst..."):
                response = run_structured_threat_reasoning(
                    predicted_attack_type=category,
                    shap_top_features=top_shap,
                    risk_score=score,
                    aggregated_behavior_summary=behavior_summary,
                )
            right.success("✅ Threat report generated!")
            for field_name, value in response.items():
                right.markdown(f"**{field_name.replace('_', ' ').title()}**")
                right.info(value)
        except LLMThreatReasoningError as exc:
            right.warning(f"LLM report unavailable: {exc}")
            right.caption("Set OPENAI_API_KEY environment variable to enable AI reports.")
        except Exception as exc:
            right.error(f"Unexpected error: {exc}")
    right.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_page()
    data = render_sidebar_upload()
    col_a, col_b = st.columns([1, 3])
    run_clicked = col_a.button(
        "🚀 Run SIEM Analysis",
        use_container_width=True,
        disabled=data is None,
    )
    col_b.caption(
        "Upload raw network logs or use the sample dataset. "
        "The pipeline will auto-run feature engineering → ML training → SHAP → risk scoring."
    )
    if "dashboard_state" not in st.session_state:
        st.session_state.dashboard_state = DashboardState()
    if run_clicked and data is not None:
        st.session_state.dashboard_state = run_training_workflow(data)
    state: DashboardState = st.session_state.dashboard_state
    render_overview_banner(state)
    tabs = st.tabs([
        "📈 Classification Metrics",
        "🧩 Confusion Matrix",
        "🔍 SHAP Explainability",
        "⚠️ Risk & AI Report",
    ])
    with tabs[0]:
        render_metrics_tab(state)
    with tabs[1]:
        render_confusion_matrix_tab(state)
    with tabs[2]:
        render_shap_tab(state)
    with tabs[3]:
        render_risk_and_ai_tab(state)


if __name__ == "__main__":
    main()
