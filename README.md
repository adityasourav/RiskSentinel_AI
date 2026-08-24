# 🛡️ RiskGuard AI

### Unified AI-Powered Fraud, Return Abuse & Chargeback Risk Management Platform

RiskGuard AI is an end-to-end AI risk management platform designed to detect, evaluate, and respond to financial risk across **merchant transaction streams, checkout orders, return abuse, coordinated abuse rings, and chargeback disputes**.

The platform combines **machine learning, anomaly detection, graph-based risk analysis, probability calibration, cost-sensitive decision optimization, and automated dispute evidence generation** into a unified decision framework.

> Built as a prototype for the Razorpay Hackathon to demonstrate how AI can move beyond simple fraud classification toward **cost-aware, explainable, and multi-layer financial risk management**.

---

## 🚀 Why RiskGuard AI?

Traditional fraud detection systems often focus on one question:

> **"Is this transaction fraudulent?"**

RiskGuard AI focuses on a broader question:

> **"What is the financial risk, how confident are we, what context surrounds the transaction, and what action minimizes the expected loss?"**

A transaction that looks normal in isolation may become risky when combined with:

- Abnormally high merchant transaction velocity
- Suspicious device behavior
- Multiple accounts sharing devices
- Shared IP addresses or billing addresses
- Previous chargebacks
- High return-abuse behavior
- Cross-border mismatches
- Coordinated account activity
- Untrusted devices
- Suspicious checkout patterns

RiskGuard AI therefore uses a **multi-layer defense architecture** rather than relying on a single classifier.

---

## 🧠 Core Idea

RiskGuard AI follows a multi-stage risk pipeline:

```text
Data Sources
     ↓
Entity Linking & Feature Engineering
     ↓
┌───────────────────────────────────────┐
│       Defense & ML Intelligence        │
│                                         │
│  1. Return/Fraud Risk Classifier       │
│  2. Fraud Spike & Velocity Detector    │
│  3. Abuse-Ring Graph Sentinel          │
│  4. Chargeback Evidence Responder      │
└───────────────────────────────────────┘
     ↓
Risk Context & Probability Calibration
     ↓
Cost-Sensitive Decision Engine
     ↓
┌────────────┬────────────┬────────────┐
│   ALLOW    │  STEP-UP   │   BLOCK    │
└────────────┴────────────┴────────────┘
     ↓
Monitoring + Evaluation + Evidence
```

---

## 🏗️ System Architecture

The system is organized into four major layers.

### 1️⃣ Data & Pipeline Layer

The platform combines multiple sources of risk information.

**Multi-Entity Graph Store**
Maintains relationships between:
- Customer accounts
- Devices
- Cards
- IP addresses
- Billing/shipping addresses
- Historical transactions

These relationships are used to identify coordinated behavior.

**Merchant Transaction Logs**
Contains transaction-level information such as:
- Authorization events
- Declines
- Refunds
- Chargebacks
- Returns
- Transaction timestamps
- Transaction amounts

**Order & Checkout Stream**
Provides real-time checkout information including:
- Customer information
- Device information
- Payment information
- Shipping information
- Order characteristics

**Entity Linking**
Links related entities using:
- IP
- Device
- Card
- Address
- Account relationships

This helps identify connections that would not be visible from a single transaction.

**Velocity & Time-Series Aggregation**
Transaction streams are transformed into time-series features using techniques such as:
- EWMA
- Rolling MAD
- Rolling windows
- CUSUM-style change detection

**Feature Engineering**
The final risk vector combines:
- Behavioral features
- Transaction velocity
- Graph features
- Device trust
- Card behavior
- Geographic signals
- Historical risk
- Checkout context

### 2️⃣ Defense & ML Core

RiskGuard AI contains four major intelligence modules.

#### 🔹 Module 1 — Return-Risk & Fraud Classifier

A calibrated gradient-boosted decision tree model evaluates individual transaction/order risk. The model generates a probability:

```
Risk Probability ∈ [0, 1]
```

The probability is then calibrated so that predicted probabilities better represent real-world likelihood.

```text
Raw Model Score
      ↓
Probability Calibration
      ↓
Calibrated Risk Probability
      ↓
Cost-Sensitive Decision
```

The model uses features such as:
- Past chargebacks
- Card velocity
- Past return rate
- Device trust
- Cross-border mismatch
- IP-to-billing distance
- Promotional/discount behavior
- Cart characteristics

#### 🔹 Module 2 — Fraud Spike Detector

This module monitors merchant-level transaction activity in near real time. It detects abnormal transaction velocity using:

**EWMA** — Exponentially Weighted Moving Average tracks the changing baseline of transaction volume:

```
EWMA(t) = α × X(t) + (1 - α) × EWMA(t-1)
```

**Rolling MAD** — Median Absolute Deviation provides a robust measure of variation and is less sensitive to extreme outliers.

**Isolation Forest** — Used to identify unusual behavioral patterns that differ from normal merchant activity.

**CUSUM** — Detects persistent changes in the statistical behavior of a transaction stream.

**Dynamic Anomaly Bands** — A dynamic upper threshold is generated around the merchant baseline. When transaction volume crosses the threshold, the system raises a threat signal.

#### 🔹 Module 3 — Abuse-Ring & Syndicate Graph Sentinel

Fraud is often coordinated across multiple accounts. RiskGuard AI models relationships between:

```text
Customer
   ↕
Device
   ↕
IP Address
   ↕
Card
   ↕
Address
```

Graph connectivity is used to identify suspicious clusters.

**Graph Analysis**
The system uses connected-component / Union-Find style clustering to identify groups of entities that share suspicious infrastructure. The resulting cluster is assigned a Syndicate Risk Score.

Example:

```text
RING-001
├── 5 Accounts
├── 1 Device
├── 1 Card
├── 1 IP
├── 5 Addresses
└── Risk Score = 100
```

This allows the system to identify coordinated abuse rather than evaluating accounts independently.

#### 🔹 Module 4 — Chargeback Evidence Responder

RiskGuard AI also addresses the post-transaction side of financial risk. For disputed transactions, the system evaluates available evidence such as:

- Address Verification (AVS)
- CVV verification
- Delivery confirmation
- Customer history
- Support interaction history
- Transaction information
- Dispute reason codes

The evidence is converted into an evidence score and probability band. The system can generate a structured dispute dossier containing:

```text
Case Information
      ↓
Evidence Checklist
      ↓
Evidence Score
      ↓
Win Probability Band
      ↓
Arbitration / Rebuttal Packet
```

This can reduce manual investigation effort and improve consistency in dispute handling.

### 3️⃣ Decision & Financial Optimization Layer

This is one of the key differentiators of RiskGuard AI.

A traditional classifier may optimize for **accuracy**. RiskGuard AI instead considers **expected financial cost**.

#### 💰 Cost-Sensitive Decision Engine

Different classification mistakes have different financial consequences:

| Outcome | Meaning | Consequence |
|---|---|---|
| False Negative | Fraud is allowed | Potential financial loss |
| False Positive | Legitimate customer is blocked | Customer friction + lost revenue |
| True Positive | Fraud is intercepted | Fraud loss avoided |
| True Negative | Legitimate transaction allowed | Frictionless customer experience |

The system therefore calculates an expected cost for different decision thresholds:

```
E[Cost(θ)] = FP(θ) × C_FP + FN(θ) × C_FN + TP(θ) × C_TP + TN(θ) × C_TN
```

The system searches for a threshold that minimizes expected financial loss.

#### 🎯 Bayesian Threshold Optimization

The platform also derives a cost-sensitive threshold based on the relative costs of classification outcomes. Instead of blindly using `Threshold = 0.50`, the system determines a threshold based on the business cost structure — making the decision boundary adaptable to operational requirements.

#### ⚡ Dual-Tier Real-Time Decision Architecture

RiskGuard AI uses two complementary risk layers.

**Tier 1 — Macro Velocity & Threat Layer**
Evaluates the broader merchant environment. It asks: *"Is something unusual happening around this merchant right now?"*
Signals include:
- Transaction spikes
- Velocity anomalies
- Merchant-level threat state
- Coordinated activity
- Abnormal decline behavior

**Tier 2 — Micro Checkout Risk Layer**
Evaluates the individual checkout transaction. It asks: *"How risky is this specific transaction given the current merchant context?"*

Tier 1 modifies the context used by Tier 2, creating a synchronized pipeline:

```text
Merchant Context
       ↓
Individual Transaction Risk
       ↓
Unified Decision
```

### 🚦 Final Decision

The system produces three operational outcomes instead of a simple binary fraud / not-fraud call:

- 🟢 **ALLOW** — Low-risk transaction. Customer proceeds normally.
- 🟡 **STEP-UP** — Transaction requires additional verification (OTP, 3DS, additional authentication).
- 🔴 **BLOCK** — Transaction is considered high risk and is stopped.

---

## 📊 Model Evaluation

RiskGuard AI includes an automated evaluation suite covering:

- ROC-AUC
- Precision / Recall
- Precision-Recall AUC
- Confusion Matrix
- Probability Calibration
- Expected Financial Cost
- Threshold Optimization
- Feature Importance

**ROC Curve**
The ROC curve evaluates the classifier's ability to distinguish between risky and legitimate transactions across different thresholds. The current evaluation demonstrates **ROC-AUC ≈ 0.7416**. AUC above 0.5 indicates discrimination better than random classification.

**Precision-Recall Curve**
Precision-Recall analysis is especially useful when fraud/risk cases are relatively rare. The current evaluation shows **PR-AUC ≈ 0.2888**, against a prevalence baseline of approximately **6.53%**. This demonstrates why accuracy alone would not be an appropriate metric for this problem.

**Threshold & Cost Optimization**
The platform allows operators to change the operational decision threshold and observe recall, precision, false positives, false negatives, and expected financial loss — making the model operationally interpretable rather than treating the threshold as a fixed ML parameter.

**Financial Cost Curve**
The cost curve demonstrates how financial loss changes as the decision threshold changes. The optimal threshold is selected around the region where expected financial loss is minimized.

**Feature Importance**
Permutation feature importance is used to estimate how much each feature contributes to model discrimination. Important signals include:

- Past chargeback disputes
- Card velocity
- Past return rate
- Wardrobing / multi-size indicator
- Device fingerprint trust
- Cross-border card mismatch
- IP-to-billing distance
- Promo / discount rate
- Cart item count
- Distinct SKU categories

**Merchant Anomaly Detection**
The velocity monitor visualizes hourly transaction count, EWMA baseline, dynamic anomaly limits, flagged risk events, average transaction value, micro-ticket percentage, and merchant decline rate — enabling detection of sudden changes in merchant behavior.

**Probability Calibration**
Probability calibration evaluates whether predicted probabilities correspond to observed outcomes:

```text
Uncalibrated Model
        ↓
Calibration
        ↓
RiskGuard Calibrated Model
```

The current evaluation demonstrates a low calibration error: **ECE ≈ 0.0065**. Better calibration is important because the probabilities are used directly by the downstream decision engine.

**Cost-Optimal Confusion Matrix**

|                 | Predicted: Legit | Predicted: Risk |
|---|---|---|
| **Actual: Legit** | TN | FP |
| **Actual: Risk** | FN | TP |

The platform uses the confusion matrix together with the financial cost parameters to determine whether a threshold is actually useful from a business perspective.

---

## 🕸️ Abuse-Ring Detection

The graph sentinel identifies connected clusters of potentially coordinated accounts. Example output:

```text
RING-001
Syndicate Tier: CONFIRMED ABUSE RING
Risk Score: 100
Accounts: 5
Devices: 1
Cards: 1
IPs: 1
Addresses: 5
Orders: 5
```

This allows risk teams to investigate the network behind suspicious activity, rather than only individual transactions.

---

## 💳 Chargeback Evidence Automation

The chargeback module creates an evidence-based assessment. Example evidence categories:

| Evidence | Status |
|---|---|
| Address Verification | Partial |
| CVV Security Match | Pass |
| Carrier Delivery Proof | Pass |
| Prior Customer History | Neutral |
| Support Interaction History | Partial |

The system then generates:

```text
Evidence Score
      ↓
Win Probability Band
      ↓
Dispute Review
      ↓
Structured Rebuttal / Arbitration Dossier
```

---

## 📊 Current Demonstration Results

| Metric | Result |
|---|---|
| ROC-AUC | ~0.7416 |
| PR-AUC | ~0.2888 |
| Calibration ECE | ~0.0065 |
| Empirical Optimal Threshold | ~0.27 |
| PR Baseline | ~6.53% |
| Decision Outcomes | Allow / Step-Up / Block |

> Metrics depend on the evaluation dataset and configured cost parameters. They are intended to demonstrate the system's decision-making pipeline rather than represent production performance.

---

## 🛠️ Technology Stack

**Programming**
- Python 3.x

**Machine Learning**
- Scikit-learn
- Gradient Boosting / GBDT
- Isolation Forest
- Probability Calibration
- ROC / Precision-Recall analysis

**Data Processing**
- Pandas
- NumPy

**Statistical / Time-Series Detection**
- EWMA
- Rolling statistics
- Median Absolute Deviation (MAD)
- CUSUM-style change detection

**Graph Analytics**
- Union-Find / Connected Components
- Multi-entity relationship analysis
- Graph-based risk scoring

**Visualization**
- Matplotlib
- Streamlit

**Application**
- Streamlit
- Python-based modular risk engine

---

## 📁 Project Structure

```text
Ai_Risk_Manager/
│
├── app.py
│
├── data/
│   └── *.csv
│
├── images/
│   ├── riskguard_ai_workflow_architecture.png
│   ├── abuse_ring_syndicate_graph.png
│   ├── chargeback_evidence_dossier.png
│   ├── global_feature_importance.png
│   ├── merchant_anomaly_velocity.png
│   ├── precision_recall_curve.png
│   ├── calibration_confusion_matrix.png
│   ├── return_risk_cost_optimizer.png
│   ├── financial_cost_threshold_roc.png
│   └── unified_dual_tier_platform.png
│
├── output/
│
├── riskguard/
│   ├── chargeback/
│   ├── graph/
│   ├── models/
│   └── utils/
│
├── evaluate_all.py
├── run_demo.py
├── test_suite.py
├── requirements.txt
└── README.md
```

> **Note:** Image filenames have been simplified to lowercase, hyphen/underscore-safe names (no spaces, `&`, or em-dashes) to keep the README links clean and avoid rendering issues on GitHub. Rename your files in `images/` to match, or update the links above to match your actual filenames.

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd Ai_Risk_Manager
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser. If Streamlit does not automatically open the browser, use the local URL displayed in the terminal.

---

## 🧪 Running Evaluation

Run the complete evaluation suite:

```bash
python evaluate_all.py
```

For demonstration/testing:

```bash
python run_demo.py
```

Run tests:

```bash
python test_suite.py
```

---

## 🎮 Streamlit Dashboard

The interactive dashboard provides multiple modules:

```text
├── Dual-Tier Architecture Playground
├── Return-Risk Scorer & Cost Optimizer
├── Velocity & Spike Monitor
├── Executive Benchmarks & Calibration
├── Abuse-Ring Graph Sentinel
└── Chargeback Evidence Responder
```

This allows users to interact with the risk system rather than simply viewing static model predictions.

---

## 🔄 End-to-End Workflow

```text
                 ┌──────────────────┐
                 │   Data Sources    │
                 └────────┬─────────┘
                          ↓
              ┌───────────────────────┐
              │ Entity Linking         │
              │ IP / Device / Card     │
              │ Address / Account      │
              └───────────┬───────────┘
                          ↓
              ┌───────────────────────┐
              │ Feature Engineering    │
              │ Behavioral + Graph +   │
              │ Velocity Features      │
              └───────────┬───────────┘
                          ↓
        ┌─────────────────┼──────────────────┐
        ↓                 ↓                  ↓
   Fraud Model      Spike Detector     Graph Sentinel
        │                 │                  │
        └─────────────────┼──────────────────┘
                          ↓
                Risk Context Layer
                          ↓
               Probability Calibration
                          ↓
              Cost-Sensitive Optimizer
                          ↓
             ┌────────────┼────────────┐
             ↓            ↓            ↓
           ALLOW       STEP-UP       BLOCK
                          │
                          ↓
               Monitoring & Analytics
                          │
                          ↓
              Chargeback Evidence
                  & Reporting
```

---

## 🌟 Key Advantages

1. **Multi-Layer Defense** — combines transaction-level ML, merchant-level anomaly detection, graph-based abuse detection, and chargeback intelligence instead of relying on a single model.
2. **Cost-Aware Decisions** — the optimal threshold is determined according to financial consequences rather than blindly selecting 0.5.
3. **Real-Time Context** — merchant-level threat conditions can influence the interpretation of individual transactions.
4. **Explainability** — risk teams can inspect feature importance, threshold behavior, confusion matrix, calibration, graph relationships, and the evidence checklist.
5. **Network-Level Fraud Detection** — graph analysis helps identify coordinated abuse across multiple accounts, devices, IPs, cards, and addresses.
6. **Reduced Manual Investigation** — automated evidence scoring and dossier generation can reduce repetitive chargeback investigation work.
7. **Customer-Friendly Risk Controls** — ALLOW (low risk) → STEP-UP (additional verification) → BLOCK (high risk), reducing unnecessary customer friction.

---

## 💡 Potential Business Applications

**Payment Platforms**
- Transaction fraud detection
- Card testing detection
- Account takeover detection
- Suspicious checkout detection

**E-Commerce**
- Return abuse
- Refund abuse
- Wardrobing detection
- Multi-account abuse

**FinTech**
- Payment risk scoring
- Merchant monitoring
- Device intelligence
- Transaction anomaly detection

**Marketplaces**
- Seller fraud
- Buyer abuse
- Coordinated account networks

**Chargeback Operations**
- Evidence collection
- Dispute prioritization
- Automated rebuttal preparation
- Evidence completeness scoring

---

## 🔐 Security & Privacy Considerations

This prototype is designed around structured risk signals and does not require exposing sensitive payment credentials.

A production implementation should additionally include:

- PCI-DSS compliant payment handling
- Encryption at rest and in transit
- Tokenization of payment identifiers
- Role-based access control
- Audit logging
- Data retention policies
- PII minimization
- Model monitoring
- Bias/fairness evaluation
- Secure API authentication

Sensitive card information should never be stored directly in the application.

---

## 📈 Production Roadmap

**Phase 1 — Data Infrastructure**
- Kafka / event streaming
- Redis
- PostgreSQL
- Feature Store

**Phase 2 — ML Infrastructure**
- MLflow
- Model registry
- Automated retraining
- Drift detection

**Phase 3 — Real-Time Risk Engine**
- FastAPI inference service
- Low-latency scoring
- Webhook integration
- Real-time alerts

**Phase 4 — Graph Infrastructure**
- Neo4j / graph database
- Large-scale entity resolution
- Graph embeddings
- Community detection

**Phase 5 — Production Monitoring**
- Model performance monitoring
- Data drift monitoring
- Threshold monitoring
- False-positive monitoring
- Financial-loss monitoring

---

## 🏆 What Makes RiskGuard AI Different?

Most basic fraud projects stop at:

```text
Input → ML Model → Fraud / Not Fraud
```

RiskGuard AI extends this into:

```text
Transaction
     +
Merchant Context
     +
Entity Relationships
     +
Behavioral Anomalies
     +
Calibrated Probability
     +
Financial Cost
     +
Operational Context
     ↓
Optimal Risk Decision
     ↓
Allow / Step-Up / Block
     ↓
Continuous Monitoring
     ↓
Chargeback Evidence Automation
```

The objective is not simply to build a model with a high accuracy score. The objective is to build a decision intelligence layer that can help payment businesses minimize financial loss while maintaining a better customer experience.

---

## 📸 Project Screenshots

- Unified Risk Platform
- Return Risk & Cost Optimization
- Merchant Anomaly Detection
- Abuse Ring Detection
- Feature Importance
- Chargeback Evidence
- System Architecture

> Add your actual screenshot images to the `images/` folder and embed them here using standard Markdown image syntax, e.g. `![Unified Risk Platform](images/unified_dual_tier_platform.png)`.

---

## 👥 Intended Users

RiskGuard AI is designed conceptually for:

- Payment Risk Teams
- Fraud Analysts
- Trust & Safety Teams
- Chargeback Operations
- Merchant Risk Teams
- FinTech Product Teams
- E-Commerce Risk Teams

---

## ⚠️ Disclaimer

RiskGuard AI is a hackathon/prototype implementation intended for research, demonstration, and educational purposes. The displayed performance metrics are based on the project's evaluation/demo data and should not be interpreted as production-level financial or fraud-loss guarantees.

A production deployment would require extensive validation, security controls, regulatory review, monitoring, and testing against real-world transaction distributions.

This project does not implement production Razorpay API/payment integration — it is a conceptual prototype built for the Razorpay Hackathon.

---

## 👨‍💻 Built For

**Razorpay Hackathon**
Project: RiskGuard AI
Category: AI / ML / FinTech / Fraud & Risk Management

---

## ⭐ Future Vision

RiskGuard AI aims to evolve from a fraud classifier into a complete Financial Risk Intelligence Platform capable of:

```text
Detect → Understand → Score → Calibrate → Optimize → Respond → Learn
```

The long-term goal is to make financial risk decisions faster, more explainable, more cost-aware, and more adaptive to emerging attack patterns.
