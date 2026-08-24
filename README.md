# RiskSentinel AI

> **Defensive AI-powered risk management for fraud, abuse rings, transaction anomalies, and chargebacks.**

RiskSentinel AI combines **calibrated ML, behavioral analytics, graph intelligence, and cost-aware decisioning** into a unified risk platform that routes transactions through **ALLOW → STEP-UP → BLOCK** decisions.

## System Architecture

![Workflow](images/Workflow.png)

## Core Intelligence

| Module | Purpose |
|---|---|
| **Calibrated GBDT Scorer** | Transaction-level return/fraud risk scoring |
| **Velocity Detector** | Detects abnormal transaction spikes and behavior |
| **Abuse-Ring Sentinel** | Finds coordinated multi-account abuse through shared entities |
| **Chargeback Responder** | Generates structured dispute evidence |

![Unified Platform](images/Unified_Dual_Tier_AI_Risk_Platform.png)

## Risk Intelligence

### Abuse-Ring Detection
Identifies suspicious account clusters through shared **devices, IPs, cards, and addresses**.

![Abuse Ring](images/Abuse_Ring_Syndicate_Graph_Sentinel.png)

### Merchant & Velocity Monitoring
Detects unusual transaction patterns and behavioral spikes.

![Velocity Monitor](images/Merchant_Anomaly_Velocity_Monitor.png)

### Return-Risk & Cost Optimization
Combines calibrated probability with financial cost to select better decision thresholds.

![Return Risk](images/Return_Risk_Scorer_Cost_Utility_Optimizer.png)

## Explainable ML

![Feature Importance](images/Global_Feature_Importance_Ranking.png)

![Precision Recall](images/Precision_Recall_Curve.png)

![Calibration & Confusion Matrix](images/Probability_Calibration_Reliability_Diagram_Cost_Optimal_Confusion_Matrix.png)

![Financial Cost & ROC](images/Total_Financial_Cost_vs_Decision_Threshold_ROC_Curve.png)

## Chargeback Intelligence

Automatically structures transaction and dispute evidence into a review-ready dossier.

![Chargeback Evidence](images/Chargeback_Evidence_Dispute_Dossier_Generator.png)

## Key Idea

Traditional fraud detection asks:

> **"Is this transaction fraudulent?"**

RiskSentinel asks:

> **"What action minimizes expected financial loss while controlling customer friction?"**

This enables **risk-aware, explainable, and financially optimized decisions** instead of relying on arbitrary thresholds.

## Tech Stack

**Python · Scikit-Learn · Pandas · NumPy · Streamlit · Matplotlib · Seaborn**

## Project Structure

```text
RiskSentinel_AI/
├── app.py
├── evaluate_all.py
├── test_suite.py
├── data/
├── riskguard/
│   ├── chargeback/
│   ├── graph/
│   ├── models/
│   └── utils/
├── output/
└── images/
