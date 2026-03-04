# 🛡️ Mini-SIEM — AI-Powered Security Operations Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/ML-RandomForest-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XAI-SHAP-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LLM-GPT--4o--mini-green?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A production-grade Security Information & Event Management system combining ML attack classification,
  SHAP explainability, LLM threat reasoning, automated PDF incident reports, and a real-time Streamlit dashboard.
</p>

-----

## 🤯 What is a SIEM?

A **SIEM (Security Information & Event Management)** system is the central nervous system of enterprise cybersecurity. It ingests network logs, detects attacks in real time, explains the reasoning, and generates reports for security analysts.

Real SIEM products (IBM QRadar, Splunk, Microsoft Sentinel) cost **₹50L–₹5Cr/year**. This project builds a fully working mini version using open-source Python! 🔥

-----

## 🏗️ Architecture

```
Raw Network Logs (CSV)
        │
        ▼
① Feature Engineering         network_feature_engineering.py
  9 features per IP per 5-min window
        │
        ▼
② ML Attack Classifier         attack_classification_pipeline.py
  RandomForest + GridSearchCV
  → Attack type + confidence %
        │
        ├─────────────────────────┐
        ▼                         ▼
③ Risk Scoring Engine         ④ SHAP Explainability
   risk_scoring_engine.py        shap_explainability.py
   → 0-100 risk score            → Why did AI decide this?
   → Low/Medium/High/Critical    → Top feature drivers
        │                         │
        └──────────┬──────────────┘
                   ▼
        ⑤ LLM Threat Reasoning     llm_threat_reasoning.py
           GPT-4o-mini
           → Narrative threat assessment
           → Business impact + response steps
                   │
                   ▼
        ⑥ PDF Incident Report      incident_response_report.py
           Auto-generated professional PDF
           (built from scratch — no external PDF libs!)
                   │
                   ▼
        ⑦ Streamlit Dashboard      streamlit_dashboard.py
           Live metrics, SHAP charts, risk gauge, AI reports
```

-----

## ✨ Features

|Module                   |Capability                                                                       |
|-------------------------|---------------------------------------------------------------------------------|
|🔧 **Feature Engineering**|9 ML features from raw logs — port entropy, SYN ratio, C2 beacon timing, and more|
|🤖 **Attack Classifier**  |Detects 6 attack types with hyperparameter tuning via GridSearchCV               |
|🔍 **SHAP Explainability**|Real TreeExplainer — explains WHY the model flagged each event                   |
|⚠️ **Risk Scoring**       |Weighted 4-signal composite score (0-100) with category mapping                  |
|🧠 **LLM Reasoning**      |GPT-4o-mini generates narrative threat assessments with retry + backoff          |
|📄 **PDF Reports**        |Professional incident response PDFs built from raw bytes — zero dependencies     |
|📊 **Dashboard**          |Streamlit UI with confusion matrix, SHAP charts, risk gauge, and AI reports      |

-----

## 🎯 Attack Types Detected

|Attack               |Description        |Key Signal                                  |
|---------------------|-------------------|--------------------------------------------|
|✅ `benign`           |Normal traffic     |Balanced, regular behavior                  |
|🔑 `brute_force`      |Password guessing  |High failed_login_ratio                     |
|🔭 `scan`             |Port reconnaissance|High port_entropy, unique_ports             |
|💥 `ddos`             |Flood attack       |Extreme connection_rate, high syn_flag_ratio|
|🕹️ `c2`               |Malware beacon     |Near-zero inter_request_time_std            |
|📤 `data_exfiltration`|Data theft         |Very high avg_packet_size                   |

-----

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Not-muzzyy/try.git
cd try

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run streamlit_dashboard.py
```

> **Optional:** Set `OPENAI_API_KEY` environment variable to enable LLM threat reports.
> 
> ```bash
> export OPENAI_API_KEY=your_key_here
> ```

-----

## 🚀 Quick Start

1. Run `streamlit run streamlit_dashboard.py`
1. Enable **“Use sample dataset”** in the sidebar *(included in repo)*
1. Click **“Run SIEM Analysis”**
1. Explore all 4 tabs — metrics, confusion matrix, SHAP, risk + AI report

-----

## 🗂️ Project Structure

```
mini-siem/
│
├── streamlit_dashboard.py          → Main UI — run this
├── network_feature_engineering.py  → Raw logs → 9 ML features
├── attack_classification_pipeline.py → ML training + inference
├── shap_explainability.py          → SHAP TreeExplainer integration
├── risk_scoring_engine.py          → 4-signal weighted risk score
├── llm_threat_reasoning.py         → GPT-4o-mini threat analysis
├── incident_response_report.py     → Raw PDF byte generation
├── mini_siem_design.md             → Full architecture document
│
├── data/
│   └── sample_intrusion_dataset.csv → 2000-row labeled dataset
│
├── artifacts/                      → Saved ML models (auto-created)
├── logs/                           → Log storage
└── requirements.txt
```

-----

## 🛠️ Tech Stack

|Technology                |Purpose                     |
|--------------------------|----------------------------|
|Python 3.10+              |Core language               |
|Streamlit                 |Interactive dashboard       |
|Scikit-learn              |RandomForest + GridSearchCV |
|SHAP                      |TreeExplainer explainability|
|Pandas / NumPy            |Data processing             |
|Matplotlib                |Charts and visualizations   |
|OpenAI API (stdlib urllib)|LLM threat reasoning        |
|Raw PDF bytes             |Incident report generation  |

-----

## 🧠 Key Engineering Highlights

**Zero external PDF dependencies** — `incident_response_report.py` generates PDFs by writing raw PDF specification bytes directly in Python. No ReportLab, no FPDF.

**Multi-version SHAP compatibility** — `shap_explainability.py` handles all 3 SHAP output formats (list, 2D array, 3D array) to prevent version-upgrade breakage.

**Exponential backoff retry** — `llm_threat_reasoning.py` uses stdlib `urllib` with exponential backoff — no `requests` library needed.

**Time-windowed features** — `network_feature_engineering.py` aggregates logs into 5-minute windows per source IP, computing Shannon entropy for port scanning detection.

**Configurable risk fusion** — `risk_scoring_engine.py` uses validated weighted scoring where weights must sum to exactly 1.0 (with float precision tolerance of 1e-9).

-----

## 📊 Sample Results

```
Classification Report (sample dataset):
              precision    recall  f1-score
benign            0.97      0.98      0.97
brute_force       0.99      0.98      0.98
c2                0.96      0.94      0.95
data_exfil        0.95      0.96      0.95
ddos              0.99      0.99      0.99
scan              0.98      0.99      0.98

Weighted F1: 0.978
```

-----

## 🔮 Future Roadmap

- [ ] Deploy on Streamlit Cloud (public demo)
- [ ] Real-time log streaming via Kafka/WebSocket
- [ ] MITRE ATT&CK framework mapping
- [ ] Multi-tenant support with RBAC
- [ ] Email/Slack alert notifications
- [ ] Extended to 15+ attack types

-----

## 👨‍💻 About the Author

**Mohammed Muzamil C**
Final Year BCA Student | Cybersecurity & Machine Learning
Nandi Institute of Management & Science College, Ballari
Vijayanagara Sri Krishnadevaraya University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/muzzammilc7)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Not-muzzyy)

-----

## 📄 License

MIT License — open source and free to use.

-----

<p align="center">
  ⭐ If this project helped you, please star the repository!
</p>
