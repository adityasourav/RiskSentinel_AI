# AI Risk Manager — Comprehensive Evaluation & Benchmark Report

## 1. Executive Summary & Core Results
This evaluation document details the empirical performance of **RiskGuard AI** across held-out test data, demonstrating mathematically rigorous decision boundaries, probability calibration, explainable anomaly detection, multi-entity abuse clustering, and automated chargeback representation.

### Held-Out Test Set Performance (2,250 Unseen Orders, 6.53% Risk Prevalence)
- **ROC-AUC:** `0.7416`
- **PR-AUC (Average Precision):** `0.2888` (vs. random prevalence baseline of `6.53%`)
- **Expected Calibration Error (ECE):** `0.0065` (Isotonic Probability Calibration)
- **Bayes Theoretical Optimal Threshold (θ*):** `0.26`
- **Empirical Minimum-Cost Threshold (θ*):** `0.27`
- **Net Financial Loss at θ*:** `$10,907` (representing **$1,588 (12.7%)** net financial savings vs. Allow-All baseline)

---

## 2. Decision Policy Financial Comparison Table
| Policy                              | Threshold θ | Recall (Fraud Caught) | Precision | False Positives | False Negatives | Total Financial Loss | Net Savings ($) | Savings (%) |
| ----------------------------------- | ----------- | --------------------- | --------- | --------------- | --------------- | -------------------- | --------------- | ----------- |
| 1. Allow All (Baseline)             | 1.00        | 0.0%                  | 0.0%      | 0               | 147             | $12,495              | $0              | 0.0%        |
| 2. Block All (Over-conservative)    | 0.00        | 100.0%                | 6.5%      | 2103            | 0               | $59,472              | $-46,977        | -376.0%     |
| 3. Default ML (θ = 0.50)            | 0.50        | 5.4%                  | 80.0%     | 2               | 139             | $11,903              | $592            | 4.7%        |
| 4. Theoretical Bayes (θ = 0.26)     | 0.26        | 28.6%                 | 38.5%     | 67              | 105             | $10,969              | $1,526          | 12.2%       |
| 5. RiskGuard Optimal ML (θ* = 0.27) | 0.27        | 27.2%                 | 40.4%     | 59              | 107             | $10,907              | $1,588          | 12.7%       |
| 6. Naive Rule Engine                | N/A         | 11.6%                 | 23.3%     | 56              | 130             | $12,686              | $-191           | -1.5%       |

---

## 3. False Positive Cost & Utility Reasoning
Standard ML benchmarks optimize accuracy or F1 score assuming symmetric costs ($C_{FP} = C_{FN}$). In real-world e-commerce and BFSI risk management:
- **False Negative Cost ($C_{FN} = \$85.00$):** Missed fraud incurs chargeback fees (\$15), complete merchandise loss, administrative penalties, and return shipping costs.
- **False Positive Cost ($C_{FP} = \$28.00$):** Falsely intercepting a legitimate customer results in lost gross profit margin, customer insult friction, and reduced Customer Lifetime Value (CLTV).
- **True Positive Intercept Cost ($C_{TP} = \$4.00$):** Frictionless 3DS step-up authentication or streamlined automated return verification cost.

By formulating the expected cost function $E[\text{Cost}(\theta)]$ and calibrating class probabilities with isotonic regression, the optimal threshold shifts from the arbitrary $\theta=0.50$ down to **$\theta^* = 0.27$**, capturing **27.2%** of fraudulent transactions while minimizing false-positive customer insult friction.

---

## 4. Evaluation Plots Generated
1. **Probability Reliability & Calibration Diagram:** `output/plots/calibration_curve.png`
2. **Receiver Operating Characteristic (ROC) Curve:** `output/plots/roc_curve.png`
3. **Precision-Recall (PR) Curve:** `output/plots/pr_curve.png`
4. **Expected Cost vs. Decision Threshold θ:** `output/plots/cost_vs_threshold.png`
5. **Standard vs. Cost-Optimal Confusion Matrices:** `output/plots/confusion_matrix_default.png`, `output/plots/confusion_matrix_optimal.png`
6. **Time-Series Velocity Anomaly Monitor:** `output/plots/time_series_anomaly.png`
