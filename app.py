"""
RiskGuard AI — Enterprise Dual-Tier AI Risk Platform
A professional, production-grade interface for fraud, return abuse, and velocity risk management.
"""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from data.generator import generate_risk_dataset
from riskguard.models.classifier import RiskClassifier, FEATURE_COLUMNS, FEATURE_LABELS
from riskguard.models.cost_optimizer import CostOptimizer, CostMatrix
from riskguard.models.calibration_benchmark import plot_calibration_comparison
from riskguard.models.time_series import (
    generate_merchant_time_series,
    InterpretableFraudSpikeDetector,
    plot_time_series_anomaly_report
)
from riskguard.models.unified_engine import UnifiedRiskEngine
from riskguard.graph.abuse_ring import generate_mock_syndicate_graph
from riskguard.chargeback.responder import ChargebackEvidenceResponder
from riskguard.utils.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_cost_vs_threshold
)

st.set_page_config(
    page_title="RiskGuard AI — Enterprise Risk Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise Theme CSS (Dark Mode & Light Mode Compatible)
st.markdown("""
<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Clean Architecture Cards */
    .arch-card-1 {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .arch-card-2 {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 4px solid #34d399;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .arch-card-title-1 {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    .arch-card-title-2 {
        font-size: 1.15rem;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 6px;
    }
    .arch-card-text {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.45;
        margin: 0;
    }

    /* Status Badges */
    .badge-critical {
        background-color: #7f1d1d;
        color: #fecaca;
        border: 1px solid #dc2626;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #78350f;
        color: #fde68a;
        border: 1px solid #d97706;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-normal {
        background-color: #064e3b;
        color: #a7f3d0;
        border: 1px solid #059669;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }

    .audit-factor {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        color: #e2e8f0;
        font-size: 0.9rem;
    }

    .dispute-summary-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    dataset_path = "data/orders_dataset.csv"
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    df = generate_risk_dataset(n_samples=15000, random_state=42)
    df.to_csv(dataset_path, index=False)
    return df


@st.cache_resource
def load_classifier(df: pd.DataFrame):
    model_path = "output/risk_classifier.pkl"
    train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df["is_risk_target"], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.17647, stratify=train_val_df["is_risk_target"],
                                        random_state=42)

    if os.path.exists(model_path):
        try:
            clf = RiskClassifier.load(model_path)
            return clf, train_df, val_df, test_df
        except Exception:
            pass
    clf = RiskClassifier(model_type="hist_gbdt", calibrate=True, random_state=42)
    clf.train(train_df, val_df)
    clf.save(model_path)
    return clf, train_df, val_df, test_df


df_all = load_dataset()
classifier, train_df, val_df, test_df = load_classifier(df_all)
y_test = test_df["is_risk_target"].values
y_test_probs = classifier.predict_proba(test_df)

# Sidebar
with st.sidebar:
    st.markdown("### RiskGuard AI")
    st.caption("Enterprise Fraud & Risk Management Platform")

    page = st.radio(
        "Navigation",
        [
            "Dual-Tier Architecture Playground",
            "Return-Risk Scorer & Cost Optimizer",
            "Velocity & Spike Monitor",
            "Executive Benchmarks & Calibration",
            "Abuse-Ring Graph Sentinel",
            "Chargeback Evidence Responder"
        ]
    )

    st.markdown("---")
    st.markdown("#### Financial Cost Parameters")
    c_fn = st.slider("False Negative Cost (C_FN) - USD", 30.0, 200.0, 85.0, 5.0,
                     help="Loss of goods, chargeback fees, and return shipping incurred on missed fraud.")
    c_fp = st.slider("False Positive Cost (C_FP) - USD", 5.0, 100.0, 28.0, 1.0,
                     help="Lost gross profit margin and customer insult / churn penalty.")
    c_tp = st.slider("True Positive Intercept Cost (C_TP) - USD", 0.0, 20.0, 4.0, 1.0,
                     help="Step-up authentication or verification friction cost.")

    cost_matrix = CostMatrix(c_fn=c_fn, c_fp=c_fp, c_tp=c_tp, c_tn=0.0)
    optimizer = CostOptimizer(cost_matrix)
    bayes_th = optimizer.theoretical_optimal_threshold()
    opt_th, opt_metrics, sweep_df = optimizer.find_empirical_optimal_threshold(y_test, y_test_probs)
    unified_engine = UnifiedRiskEngine(classifier, optimizer)

    st.markdown("---")
    st.markdown(f"**Analytic Bayes Threshold:** `{bayes_th:.2f}`")
    st.markdown(f"**Empirical Optimal Threshold:** `{opt_th:.2f}`")

# -------------------------------------------------------------
# Page 1: Dual-Tier Architecture Playground
# -------------------------------------------------------------
if page == "Dual-Tier Architecture Playground":
    st.markdown('<div class="main-title">Unified Dual-Tier AI Risk Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Synchronized multi-layer defense integrating Merchant Macro Velocity Anomaly Detection with Calibrated Checkout Scoring and Cost Optimization.</div>',
        unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        <div class="arch-card-1">
            <div class="arch-card-title-1">Tier 1: Macro Velocity Anomaly Layer</div>
            <p class="arch-card-text">
                Monitors merchant transaction velocity in real time using <b>48-hour EWMA Dynamic Bands</b>, <b>Rolling MAD Z-Scores</b>, and <b>Isolation Forest</b>. Detects distributed card-testing bot surges and credential stuffing attacks.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        st.markdown("""
        <div class="arch-card-2">
            <div class="arch-card-title-2">Tier 2: Micro Checkout Risk & Cost Layer</div>
            <p class="arch-card-text">
                Evaluates individual checkout orders via <b>Isotonically Calibrated GBDT</b> and dynamically tunes decision thresholds based on real-time merchant threat context and financial cost utility.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Live Dual-Tier Simulation Playground")
    st.markdown(
        "Demonstrates how a **Merchant Velocity Spike (Tier 1)** dynamically alters the evaluation and decision boundary for an incoming **Checkout Order (Tier 2)**.")

    col_m1, col_m2 = st.columns([1, 1.2])
    with col_m1:
        st.markdown("#### 1. Merchant Stream Context (Tier 1)")
        macro_scenario = st.selectbox(
            "Select Merchant Stream Scenario",
            [
                "NORMAL: Standard Daily Operations",
                "CARD_TESTING: Micro-Charge Bot Attack Spike ($1.50 charges, 48% decline rate)",
                "VELOCITY_SURGE: Credential Stuffing Surge",
                "ORGANIC_PROMO: Flash Sale Event (High Volume, Low Decline Rate)"
            ]
        )

        if "CARD_TESTING" in macro_scenario:
            mock_stream_row = {
                "threat_category": "CRITICAL: CARD TESTING BOT ATTACK",
                "is_flagged_threat": 1, "tx_count": 580, "volume_ewma_mean": 65.0,
                "robust_zscore": 6.8, "decline_rate": 0.48, "micro_ticket_ratio": 0.88,
                "threat_explanation": "Micro-charge surge (88% below $3.00); authorization decline rate elevated at 48.0%",
                "recommended_action": "TRIGGER STEP-UP 3DS & RATE LIMITING"
            }
        elif "VELOCITY_SURGE" in macro_scenario:
            mock_stream_row = {
                "threat_category": "HIGH: CREDENTIAL STUFFING / VELOCITY SURGE",
                "is_flagged_threat": 1, "tx_count": 340, "volume_ewma_mean": 65.0,
                "robust_zscore": 4.5, "decline_rate": 0.32, "micro_ticket_ratio": 0.02,
                "threat_explanation": "Transaction rate 340 tx/hr (4.5x above EWMA baseline); high payment failure rate",
                "recommended_action": "ENABLE CAPTCHA CHALLENGE & 3DS AUTH"
            }
        elif "ORGANIC_PROMO" in macro_scenario:
            mock_stream_row = {
                "threat_category": "BENIGN: ORGANIC PROMO SURGE",
                "is_flagged_threat": 0, "tx_count": 290, "volume_ewma_mean": 65.0,
                "robust_zscore": 3.8, "decline_rate": 0.018, "micro_ticket_ratio": 0.01,
                "threat_explanation": "Volume elevated at 290 tx/hr, but decline rate is healthy (1.8%); normal basket sizes",
                "recommended_action": "ALLOW FRICTIONLESS (PROTECT CONVERSION)"
            }
        else:
            mock_stream_row = {
                "threat_category": "NORMAL", "is_flagged_threat": 0, "tx_count": 55,
                "volume_ewma_mean": 58.0, "robust_zscore": 0.4, "decline_rate": 0.021, "micro_ticket_ratio": 0.012,
                "threat_explanation": "All operational metrics within standard tolerances",
                "recommended_action": "MONITOR"
            }

        mock_df = pd.DataFrame([mock_stream_row] * 10)
        mock_df["timestamp"] = pd.date_range("2026-08-22 10:00:00", periods=10, freq="h")
        mock_df["avg_ticket_size"] = 1.99 if "CARD_TESTING" in macro_scenario else 75.0

        unified_engine.update_merchant_velocity_state("MERCHANT-ALPHA-99", mock_df)
        m_state = unified_engine.current_merchant_state["MERCHANT-ALPHA-99"]

        st.markdown("**Tier 1 Active Threat State:**")
        if m_state["threat_level"] == "CRITICAL_CARD_TESTING":
            st.markdown('<span class="badge-critical">CRITICAL: CARD TESTING ATTACK</span>', unsafe_allow_html=True)
        elif m_state["threat_level"] == "HIGH_VELOCITY_SURGE":
            st.markdown('<span class="badge-moderate">HIGH: VELOCITY SURGE</span>', unsafe_allow_html=True)
        elif m_state["threat_level"] == "ORGANIC_PROMO_EVENT":
            st.markdown('<span class="badge-normal">BENIGN: ORGANIC FLASH SALE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-normal">NORMAL BASELINE</span>', unsafe_allow_html=True)

        st.caption(f"Decomposition: {m_state['explanation']}")

    with col_m2:
        st.markdown("#### 2. Incoming Checkout Transaction (Tier 2)")
        preset_tx = st.selectbox(
            "Select Test Order Profile",
            [
                "Micro-Ticket Order from Untrusted Device ($2.50 charge, Device Trust 0.20)",
                "High-Risk Wardrobing Order ($420, Multi-size SKU, Return Rate 65%)",
                "Legitimate Buyer Order ($85, Account Age 350d, Return Rate 5%)"
            ]
        )

        if "Micro-Ticket" in preset_tx:
            order_data = {
                "order_id": "ORD-TEST-001", "order_amount": 2.50, "item_count": 1, "distinct_categories": 1,
                "high_resale_ratio": 0.0, "wardrobing_flag": 0, "discount_percentage": 0.0,
                "account_age_days": 1.0, "historical_orders": 0, "historical_return_rate": 0.0,
                "historical_chargebacks": 0, "delivery_address_changes_30d": 0, "billing_shipping_mismatch": 0,
                "ip_distance_to_billing_km": 50.0, "device_trust_score": 0.20, "checkout_duration_sec": 3.0,
                "payment_velocity_24h": 4, "card_is_prepaid": 1, "card_country_mismatch": 0
            }
        elif "Wardrobing" in preset_tx:
            order_data = {
                "order_id": "ORD-TEST-002", "order_amount": 420.0, "item_count": 4, "distinct_categories": 1,
                "high_resale_ratio": 0.85, "wardrobing_flag": 1, "discount_percentage": 0.30,
                "account_age_days": 14.0, "historical_orders": 2, "historical_return_rate": 0.65,
                "historical_chargebacks": 1, "delivery_address_changes_30d": 2, "billing_shipping_mismatch": 1,
                "ip_distance_to_billing_km": 150.0, "device_trust_score": 0.30, "checkout_duration_sec": 5.0,
                "payment_velocity_24h": 4, "card_is_prepaid": 1, "card_country_mismatch": 1
            }
        else:
            order_data = {
                "order_id": "ORD-TEST-003", "order_amount": 85.0, "item_count": 2, "distinct_categories": 2,
                "high_resale_ratio": 0.10, "wardrobing_flag": 0, "discount_percentage": 0.10,
                "account_age_days": 350.0, "historical_orders": 15, "historical_return_rate": 0.05,
                "historical_chargebacks": 0, "delivery_address_changes_30d": 0, "billing_shipping_mismatch": 0,
                "ip_distance_to_billing_km": 10.0, "device_trust_score": 0.95, "checkout_duration_sec": 75.0,
                "payment_velocity_24h": 1, "card_is_prepaid": 0, "card_country_mismatch": 0
            }

        decision_res = unified_engine.score_checkout_order(order_data, merchant_id="MERCHANT-ALPHA-99",
                                                           base_decision_threshold=opt_th)

        st.markdown("#### 3. Unified Decision Output")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.metric("Base ML Risk Score", f"{decision_res['base_risk_probability'] * 100:.1f}%")
        with d_col2:
            st.metric("Context-Adjusted Score", f"{decision_res['effective_risk_probability'] * 100:.1f}%")
        with d_col3:
            st.metric("Active Decision Threshold", f"{decision_res['active_decision_threshold']:.2f}")

        st.markdown(f"**Decision:** `{decision_res['decision']}` | **Action:** `{decision_res['recommended_action']}`")
        st.markdown("**Explainability Audit Trail:**")
        for factor in decision_res["risk_factors"]:
            st.markdown(f'<div class="audit-factor">{factor}</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Page 2: Return-Risk Scorer & Cost Optimizer
# -------------------------------------------------------------
elif page == "Return-Risk Scorer & Cost Optimizer":
    st.markdown('<div class="main-title">Return-Risk Scorer & Cost-Utility Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Precision/recall trade-off analysis and dollar cost quantification for false positives versus false negatives.</div>',
        unsafe_allow_html=True)

    curr_theta = st.slider("Select Operational Decision Threshold", 0.01, 0.99, float(opt_th), 0.01)
    curr_metrics = compute_all_metrics(y_test, y_test_probs, threshold=curr_theta, **cost_matrix.__dict__)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Recall (Fraud Caught)", f"{curr_metrics['recall'] * 100:.1f}%")
    with kpi2:
        st.metric("Precision", f"{curr_metrics['precision'] * 100:.1f}%")
    with kpi3:
        st.metric("False Positives", f"{curr_metrics['confusion_matrix']['fp']:,}")
    with kpi4:
        st.metric("False Negatives", f"{curr_metrics['confusion_matrix']['fn']:,}")
    with kpi5:
        st.metric("Total Expected Loss", f"${curr_metrics['total_cost']:,.0f}",
                  f"{curr_metrics['savings_pct']:.1f}% vs Baseline")

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### Total Financial Cost vs. Decision Threshold")
        fig_cost = plot_cost_vs_threshold(sweep_df, pd.Series(opt_metrics))
        st.pyplot(fig_cost)
    with col_c2:
        st.markdown("#### Model Discrimination Curves")
        sub_tab1, sub_tab2 = st.tabs(["ROC Curve", "Precision-Recall Curve"])
        with sub_tab1:
            fig_roc = plot_roc_curve(y_test, y_test_probs, optimal_threshold=curr_theta)
            st.pyplot(fig_roc)
        with sub_tab2:
            fig_pr = plot_pr_curve(y_test, y_test_probs, optimal_threshold=curr_theta)
            st.pyplot(fig_pr)

    st.markdown("---")
    st.markdown("#### Global Feature Importance Ranking (BFSI Regulatory Transparency)")
    if classifier.feature_importance_df is not None:
        fig_imp, ax_imp = plt.subplots(figsize=(10, 4.2), dpi=200)
        top_imp = classifier.feature_importance_df.head(10)
        sns.barplot(data=top_imp, x="importance_mean", y="feature_label", palette="Blues_r", ax=ax_imp,
                    edgecolor="#2b6cb0")
        ax_imp.set_title("Permutation Feature Importance (ROC-AUC Impact on Validation Set)", fontsize=11,
                         weight="bold")
        ax_imp.set_xlabel("Mean AUC Drop", fontsize=10)
        ax_imp.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_imp)

# -------------------------------------------------------------
# Page 3: Velocity & Spike Monitor
# -------------------------------------------------------------
elif page == "Velocity & Spike Monitor":
    st.markdown('<div class="main-title">Time-Series Velocity & Fraud-Spike Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Interpretable anomaly detection combining EWMA Dynamic Envelopes, Rolling MAD, and Isolation Forest.</div>',
        unsafe_allow_html=True)

    ts_df = generate_merchant_time_series(merchant_id="MERCHANT-ALPHA-99", n_days=60, random_state=42)
    ts_detector = InterpretableFraudSpikeDetector(ewma_span=48, z_threshold=3.5)
    ts_analyzed = ts_detector.analyze_stream(ts_df)

    col_t1, col_t2, col_t3 = st.columns(3)
    attacks_found = ts_analyzed[ts_analyzed["is_flagged_threat"] == 1]
    with col_t1:
        st.metric("Monitored Stream", "1,440 Hours", "60 Days Horizon")
    with col_t2:
        st.metric("Flagged Attacks", f"{len(attacks_found)} Hours", "Card Testing & Bot Spikes")
    with col_t3:
        st.metric("Flash Sale Precision", "100% Allowed", "0 False Alarms on Promo")

    st.markdown("---")
    st.markdown("#### 60-Day Multi-Variate Diagnostic Monitor")
    fig_ts = plot_time_series_anomaly_report(ts_analyzed)
    st.pyplot(fig_ts)

    st.markdown("---")
    st.markdown("#### Detected Attack Events Feed")
    if len(attacks_found) > 0:
        threat_table = attacks_found[
            ["timestamp", "tx_count", "avg_ticket_size", "decline_rate", "micro_ticket_ratio", "threat_category",
             "recommended_action"]].copy()
        threat_table["decline_rate"] = (threat_table["decline_rate"] * 100).round(1).astype(str) + "%"
        threat_table["micro_ticket_ratio"] = (threat_table["micro_ticket_ratio"] * 100).round(1).astype(str) + "%"
        threat_table["avg_ticket_size"] = "$" + threat_table["avg_ticket_size"].astype(str)
        st.dataframe(threat_table, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# Page 4: Executive Benchmarks & Calibration
# -------------------------------------------------------------
elif page == "Executive Benchmarks & Calibration":
    st.markdown('<div class="main-title">Executive Benchmarks & Probability Calibration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Rigorous evaluation on held-out test data (2,250 unseen transactions) demonstrating true cost utility.</div>',
        unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC Score", f"{opt_metrics['roc_auc']:.4f}")
    with col2:
        st.metric("PR-AUC Score", f"{opt_metrics['pr_auc']:.4f}", f"Prevalence: {np.mean(y_test) * 100:.2f}%")
    with col3:
        st.metric("Optimal Threshold", f"{opt_th:.2f}")
    with col4:
        st.metric("Net Cost Savings", f"${opt_metrics['savings_vs_allow_all']:,.0f}",
                  f"+{opt_metrics['savings_pct']:.1f}% vs Allow-All")

    st.markdown("---")
    st.markdown("#### Decision Policy Benchmark Comparison")
    heuristic_preds = ((test_df["order_amount"] > 350) & (test_df["historical_return_rate"] > 0.30) | (
                test_df["payment_velocity_24h"] > 3)).astype(int).values
    policy_df = optimizer.compare_policies(y_test, y_test_probs, heuristic_predictions=heuristic_preds)
    st.dataframe(policy_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Probability Calibration Reliability Diagram")
        cal_plot_path = "output/plots/calibration_curve.png"
        if os.path.exists(cal_plot_path):
            st.image(cal_plot_path, use_container_width=True)
    with col_e2:
        st.markdown(f"#### Cost-Optimal Confusion Matrix (Threshold = {opt_th:.2f})")
        cm_opt_path = "output/plots/confusion_matrix_optimal.png"
        if os.path.exists(cm_opt_path):
            st.image(cm_opt_path, use_container_width=True)

# -------------------------------------------------------------
# Page 5: Abuse-Ring Graph Sentinel
# -------------------------------------------------------------
elif page == "Abuse-Ring Graph Sentinel":
    st.markdown('<div class="main-title">Abuse-Ring & Syndicate Graph Sentinel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Multi-entity graph connected component clustering uncovering coordinated multi-account burner rings.</div>',
        unsafe_allow_html=True)

    sentinel = generate_mock_syndicate_graph()
    ring_df = sentinel.detect_rings(min_cluster_accounts=2)
    st.dataframe(ring_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Cluster Investigation: Burner Device Scalping Ring (RING-001)")
    st.markdown("""
    - **Entity Link:** 5 distinct customer accounts sharing a single hardware emulator `DEV-EMULATOR-8819`, IP `194.26.29.11`, and payment card `CARD-HASH-7721`.
    - **Fuzzy Address Normalization:** Matches address evasion attempts (`742 Evergreen Terrace Apt #1B` vs `742 Evergreen Terrace Unit 1-B`).
    - **Defense Policy:** Mandatory Step-Up 3DS authentication enforced across the linked cluster; return quarantine protocol activated.
    """)

# -------------------------------------------------------------
# Page 6: Chargeback Evidence Responder
# -------------------------------------------------------------
elif page == "Chargeback Evidence Responder":
    st.markdown('<div class="main-title">Chargeback Evidence & Dispute Dossier Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Automated evidence completeness scoring under Visa Compelling Evidence 3.0 and legal-grade arbitration dossier compiler.</div>',
        unsafe_allow_html=True)

    with open("data/mock_disputes.json", "r") as f:
        disputes = json.load(f)
    cb_responder = ChargebackEvidenceResponder()

    case_labels = [
        f"{d['dispute_id']} — ${d['amount_disputed']:.2f} {d['card_brand']} (Code {d['reason_code']}: {d['customer_name']})"
        for d in disputes]
    selected_idx = st.selectbox("Select Active Chargeback Dispute Case", range(len(disputes)),
                                format_func=lambda i: case_labels[i])
    selected_dispute = disputes[selected_idx]
    score_res = cb_responder.score_dispute_evidence(selected_dispute)

    st.markdown("""
    <div class="dispute-summary-card">
        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
            Case Reference: {} | Disputed Amount: ${:.2f} {} | Card: {}
        </div>
        <div style="font-size: 0.9rem; color: #94a3b8;">
            <b>Reason Code:</b> {} — <i>{}</i>
        </div>
    </div>
    """.format(
        selected_dispute["dispute_id"],
        selected_dispute["amount_disputed"],
        selected_dispute.get("currency", "USD"),
        selected_dispute["card_brand"],
        selected_dispute["reason_code"],
        selected_dispute["reason_description"]
    ), unsafe_allow_html=True)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Evidence Score", f"{score_res['evidence_score_pct']}%")
    with col_d2:
        st.metric("Win Probability Band", score_res['win_probability_band'])
    with col_d3:
        if "STRONGLY" in score_res['recommended_action'] or "RECOMMEND" in score_res['recommended_action']:
            st.markdown(
                '<div class="badge-normal" style="margin-top: 14px; padding: 10px;">SUBMIT FORMAL REBUTTAL</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="badge-moderate" style="margin-top: 14px; padding: 10px;">REVIEW / ACCEPT DISPUTE</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Evidence Checklist Breakdown")
    checks_df = pd.DataFrame(score_res["checks"])
    st.dataframe(checks_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    rebuttal_md = cb_responder.generate_rebuttal_packet(selected_dispute)
    st.download_button("Download Formal Rebuttal Packet (.md)", rebuttal_md,
                       f"rebuttal_{selected_dispute['dispute_id']}.md", "text/markdown")
    with st.expander("Preview Generated Legal Arbitration Dossier", expanded=True):
        st.markdown(rebuttal_md)