# Mini-SIEM Architecture Design

## 1) Module responsibilities

### A. Log Ingestion & Normalization Layer
- Collects structured network telemetry from sources such as firewall logs, NetFlow/IPFIX, DNS logs, proxy logs, and endpoint network events.
- Enforces a common schema (`event_time`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `bytes_in`, `bytes_out`, `action`, `device_id`, `tenant_id`, etc.).
- Performs deduplication, time normalization (UTC), enrichment with asset inventory tags, and integrity checks.

### B. Feature Store & Label Management
- Stores engineered, model-ready features at event/session/window granularity.
- Manages supervised labels (e.g., `benign`, `scan`, `brute_force`, `c2`, `data_exfiltration`) and lineage metadata.
- Supports offline training and online inference consistency by using versioned feature definitions.

### C. Supervised ML Attack Classifier
- Performs multi-class attack classification over network events/sessions.
- Uses calibrated probability outputs for downstream risk scoring.
- Supports model versioning, drift monitoring, and rollback.

### D. Explainability Service (SHAP)
- Computes local SHAP values per high-risk prediction for analyst transparency.
- Produces global importance summaries to validate model behavior against threat hypotheses.
- Flags unstable explanations (large variance over similar samples) as potential model quality issues.

### E. LLM Threat Reasoning Engine
- Converts structured detections + SHAP explanations + historical context into narrative threat assessments.
- Executes controlled reasoning tasks: hypothesis generation, kill-chain mapping, confidence rationale, and recommended next actions.
- Operates with strict prompt boundaries and retrieval from curated internal knowledge (playbooks, ATT&CK mappings, prior incidents).

### F. Risk Scoring Engine
- Fuses ML confidence, SHAP-derived driver severity, rule hits, asset criticality, user/entity risk, and temporal persistence.
- Produces normalized risk scores (0–100) at event, entity, and incident level.
- Applies policy thresholds to trigger alerts, escalations, or automated containment workflows.

### G. Case Management & Analyst UI
- Presents alert timeline, evidence bundle, SHAP top factors, LLM reasoning summary, and recommended triage actions.
- Tracks analyst decisions for feedback loops and label correction.

### H. Governance, Security, and Audit Layer
- Centralizes RBAC/ABAC, data retention, encryption, immutable audit logs, and model governance controls.
- Ensures traceability: input logs → features → model output → explanation → analyst action.

## 2) Data flow diagram explanation

1. **Collection:** Network devices and sensors emit structured logs to an ingestion bus.
2. **Normalization/Enrichment:** Logs are validated, normalized to a common schema, enriched with CMDB/asset/user context, and persisted in hot+cold storage.
3. **Feature Generation:** Streaming and batch feature pipelines compute statistical, temporal, and contextual features and publish them to the feature store.
4. **Classification:** Online inference service scores incoming events/sessions with the supervised model and returns class probabilities.
5. **Explainability:** For scores above configurable thresholds (or sampled baseline traffic), SHAP service computes feature attributions.
6. **Reasoning:** LLM engine consumes: (a) model output, (b) SHAP evidence, (c) correlated events, and (d) threat intel context to create a structured assessment.
7. **Risk Fusion:** Risk engine calculates composite score and severity tier; correlation rules can aggregate low-signal events into a higher-confidence incident.
8. **Alerting/Response:** High-severity findings enter case management/SOAR for analyst triage or automated controls.
9. **Feedback Loop:** Analyst verdicts and post-incident outcomes update labels, retraining sets, thresholds, and prompt templates.

## 3) Feature engineering strategy

### A. Core network behavior features
- Flow volume and directionality: `bytes_in/out`, packet counts, ratio metrics.
- Connection behavior: unique destination count, failed/successful connection ratios, port entropy.
- Protocol/service features: protocol-specific flags, uncommon port-service pairs.

### B. Temporal and sequence features
- Sliding windows (1m, 5m, 1h): burstiness, periodicity, beacon-like intervals.
- Session progression markers: connection retries, time-to-first-success, long-lived sessions.
- Baseline deviation: z-score or robust deviation from entity historical norms.

### C. Entity and graph-context features
- Asset criticality, exposure status, business role.
- Peer-group deviation (host compared to similar role-based cohort).
- Graph indicators: fan-out/fan-in anomalies, rare communication edges, lateral movement patterns.

### D. Threat-intel and policy features
- Indicator matches (IP/domain reputation, known C2 infra, geo-risk indicators).
- Policy violations (blocked outbound, denied privileged segment access).

### E. Label and dataset quality strategy
- Balanced sampling or class-weighting for imbalanced attack classes.
- Time-aware train/validation/test splitting to avoid leakage.
- Hard-negative mining to reduce false positives in noisy environments.

### F. Explainability-aligned feature governance
- Prefer semantically interpretable features where possible.
- Track feature drift and SHAP drift; unstable key features trigger review before promotion.

## 4) Evaluation metrics

### A. ML classifier quality
- **Macro-F1 / Weighted-F1:** handles class imbalance across attack types.
- **Precision/Recall by class:** especially critical for high-impact attacks (e.g., exfiltration).
- **PR-AUC:** preferred over ROC-AUC in imbalanced security datasets.
- **Calibration metrics (Brier score / ECE):** ensures predicted probabilities are actionable for risk fusion.

### B. SIEM detection effectiveness
- **Alert precision (true positive rate among alerts).**
- **Detection latency:** event time to alert generation.
- **MTTD / MTTR impact:** operational SOC outcomes.
- **Incident coverage:** % of ATT&CK tactics/techniques meaningfully detected.

### C. Explainability and analyst utility
- **Explanation consistency:** similar events should have similar top SHAP drivers.
- **Analyst acceptance rate:** % alerts where explanations are deemed useful.
- **Triage acceleration:** reduction in average investigation time.

### D. LLM reasoning quality and safety
- **Factual grounding score:** claims attributable to provided evidence.
- **Hallucination rate:** unsupported assertions per assessment.
- **Actionability score:** analyst-rated usefulness of recommendations.

### E. Risk engine performance
- **Risk-to-incident correlation:** how strongly high scores map to confirmed incidents.
- **Threshold stability:** alert volume volatility under normal traffic changes.
- **Business impact weighting validity:** alignment with critical asset protection outcomes.

## 5) Security considerations

### A. Data protection and privacy
- Encrypt data in transit (mTLS) and at rest (KMS-backed keys).
- Minimize PII; tokenize or pseudonymize sensitive fields where feasible.
- Enforce retention schedules and legal/regulatory handling constraints.

### B. Access control and isolation
- Fine-grained RBAC/ABAC for telemetry, models, and investigations.
- Tenant/workspace isolation for multi-organization deployments.
- Just-in-time privileged access with full audit trail.

### C. Pipeline and model integrity
- Signed model artifacts and provenance verification before deployment.
- Feature pipeline integrity checks to detect poisoning or schema tampering.
- Canary deployments and rollback safeguards for model updates.

### D. Adversarial and abuse resistance
- Detect evasion patterns (mimicry traffic, low-and-slow behavior).
- Monitor for data poisoning indicators in training feedback loops.
- Rate-limit and validate external intel feeds to reduce supply-chain injection risk.

### E. LLM-specific guardrails
- Restrict prompts to structured evidence; disallow open-ended internet lookups in critical paths unless brokered.
- Apply policy filters to prevent sensitive data exfiltration in generated text.
- Require citation-style evidence binding in LLM outputs for analyst trust.

### F. Operational resilience
- Queue buffering and backpressure handling for ingestion spikes.
- Graceful degradation modes (rule-based fallback if ML/LLM components fail).
- Disaster recovery: replicated storage, tested backups, and incident runbooks.
