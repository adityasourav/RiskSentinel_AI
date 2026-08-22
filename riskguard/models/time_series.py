"""
Interpretable Time-Series Anomaly & Fraud-Spike Velocity Detector
Implements EWMA, Rolling MAD, and Multivariate Isolation Forest for BFSI explainable fraud detection.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def generate_merchant_time_series(
        merchant_id: str = "MERCHANT-ALPHA-99",
        n_days: int = 60,
        random_state: int = 42
) -> pd.DataFrame:
    np.random.seed(random_state)
    n_hours = max(24, n_days * 24)
    date_range = pd.date_range(start="2026-06-01", periods=n_hours, freq="h")

    hour_of_day = date_range.hour.to_numpy()
    day_of_week = date_range.dayofweek.to_numpy()

    diurnal_curve = 1.0 + 0.6 * np.sin((hour_of_day - 6) * np.pi / 12)
    weekend_factor = np.where(day_of_week >= 5, 1.25, 1.0)

    base_volume = np.array(45 * diurnal_curve * weekend_factor + np.random.poisson(lam=10, size=n_hours), dtype=float)
    avg_ticket_size = np.array(np.random.normal(loc=75.0, scale=8.0, size=n_hours), dtype=float)
    decline_rate = np.array(np.random.beta(a=2, b=90, size=n_hours), dtype=float)
    new_device_ratio = np.array(np.random.beta(a=3, b=15, size=n_hours), dtype=float)
    micro_ticket_ratio = np.array(np.random.beta(a=1, b=80, size=n_hours), dtype=float)

    attack_label = ["NORMAL"] * n_hours

    start_1 = min(n_hours - 8, max(0, int(n_hours * 0.30)))
    for h in range(start_1, min(n_hours, start_1 + 8)):
        base_volume[h] += np.random.randint(350, 600)
        avg_ticket_size[h] = np.random.uniform(1.50, 3.20)
        decline_rate[h] = np.random.uniform(0.38, 0.55)
        new_device_ratio[h] = np.random.uniform(0.85, 0.96)
        micro_ticket_ratio[h] = np.random.uniform(0.75, 0.92)
        attack_label[h] = "CARD_TESTING_SPIKE"

    start_2 = min(n_hours - 6, max(0, int(n_hours * 0.70)))
    for h in range(start_2, min(n_hours, start_2 + 6)):
        base_volume[h] += np.random.randint(200, 350)
        avg_ticket_size[h] = np.random.uniform(280.0, 420.0)
        decline_rate[h] = np.random.uniform(0.25, 0.40)
        new_device_ratio[h] = np.random.uniform(0.80, 0.92)
        micro_ticket_ratio[h] = 0.01
        attack_label[h] = "VELOCITY_FRAUD_SURGE"

    start_3 = min(n_hours - 10, max(0, int(n_hours * 0.50)))
    for h in range(start_3, min(n_hours, start_3 + 10)):
        base_volume[h] += np.random.randint(180, 260)
        avg_ticket_size[h] = np.random.normal(loc=72.0, scale=5.0)
        decline_rate[h] = np.random.uniform(0.015, 0.025)
        new_device_ratio[h] = np.random.uniform(0.20, 0.30)
        micro_ticket_ratio[h] = np.random.uniform(0.01, 0.02)
        attack_label[h] = "ORGANIC_FLASH_SALE"

    return pd.DataFrame({
        "timestamp": date_range,
        "merchant_id": merchant_id,
        "tx_count": base_volume.round(0).astype(int),
        "avg_ticket_size": avg_ticket_size.round(2),
        "decline_rate": decline_rate.round(4),
        "new_device_ratio": new_device_ratio.round(4),
        "micro_ticket_ratio": micro_ticket_ratio.round(4),
        "ground_truth_event": attack_label
    })


class InterpretableFraudSpikeDetector:
    def __init__(
            self,
            ewma_span: int = 48,
            z_threshold: float = 3.5,
            contamination: float = 0.02,
            random_state: int = 42
    ):
        self.ewma_span = ewma_span
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.random_state = random_state
        self.iforest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=random_state
        )

    def analyze_stream(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "new_device_ratio" not in out.columns:
            out["new_device_ratio"] = 0.16
        if "micro_ticket_ratio" not in out.columns:
            out["micro_ticket_ratio"] = 0.01
        if "decline_rate" not in out.columns:
            out["decline_rate"] = 0.02
        if "avg_ticket_size" not in out.columns:
            out["avg_ticket_size"] = 75.0

        # 1. EWMA Baseline
        out["volume_ewma_mean"] = out["tx_count"].ewm(span=self.ewma_span).mean()
        out["volume_ewma_std"] = out["tx_count"].ewm(span=self.ewma_span).std().fillna(1.0)
        out["volume_upper_envelope"] = out["volume_ewma_mean"] + (self.z_threshold * out["volume_ewma_std"])
        out["ewma_anomaly"] = (out["tx_count"] > out["volume_upper_envelope"]).astype(int)

        # 2. Rolling MAD
        roll_median = out["tx_count"].rolling(window=min(24, len(out)), min_periods=1).median().bfill()
        roll_mad = (out["tx_count"] - roll_median).abs().rolling(window=min(24, len(out)),
                                                                 min_periods=1).median().bfill()
        roll_mad = np.maximum(roll_mad, 1.0)
        out["robust_zscore"] = (0.6745 * (out["tx_count"] - roll_median) / roll_mad).round(2)
        out["mad_anomaly"] = (out["robust_zscore"] > self.z_threshold).astype(int)

        # 3. Micro-ticket card testing heuristic
        out["card_testing_flag"] = (
                (out["micro_ticket_ratio"] > 0.40) &
                (out["decline_rate"] > 0.20) &
                (out["new_device_ratio"] > 0.60)
        ).astype(int)

        # 4. Multivariate Isolation Forest
        feature_cols = ["tx_count", "avg_ticket_size", "decline_rate", "new_device_ratio", "micro_ticket_ratio"]
        X_mat = out[feature_cols].values
        self.iforest.fit(X_mat)
        if_preds = self.iforest.predict(X_mat)
        out["iforest_anomaly"] = (if_preds == -1).astype(int)
        out["iforest_score"] = (-self.iforest.decision_function(X_mat)).round(3)

        is_attack = []
        threat_categories = []
        explanations = []
        recommendations = []

        for idx, row in out.iterrows():
            reasons = []
            flag = 0
            cat = "NORMAL"
            rec = "MONITOR"

            if row["card_testing_flag"] == 1 or (row["micro_ticket_ratio"] > 0.50 and row["decline_rate"] > 0.15):
                flag = 1
                cat = "CRITICAL: CARD TESTING BOT ATTACK"
                rec = "TRIGGER STEP-UP 3DS & RATE LIMITING"
                reasons.append(
                    f"Micro-charge spike: {row['micro_ticket_ratio'] * 100:.1f}% charges < $3.00 (Normal: ~1%).")
                reasons.append(f"Abnormal decline rate: {row['decline_rate'] * 100:.1f}% (Normal: ~2%).")
                reasons.append(f"New device ratio: {row['new_device_ratio'] * 100:.1f}%.")
            elif row["ewma_anomaly"] == 1 and row["decline_rate"] > 0.12 and row["avg_ticket_size"] > 180:
                flag = 1
                cat = "HIGH: CREDENTIAL STUFFING / VELOCITY SURGE"
                rec = "ENABLE CAPTCHA CHALLENGE & 3DS AUTH"
                reasons.append(f"Transaction surge: {row['tx_count']} tx/hr ({row['robust_zscore']}σ robust z-score).")
                reasons.append(f"Elevated cart ticket: ${row['avg_ticket_size']:.2f}.")
                reasons.append(f"Decline rate elevated: {row['decline_rate'] * 100:.1f}%.")
            elif row["ewma_anomaly"] == 1 and row["decline_rate"] <= 0.04:
                flag = 0
                cat = "BENIGN: ORGANIC PROMO SURGE"
                rec = "ALLOW FRICTIONLESS (HEALTHY CONVERSION)"
                reasons.append(
                    f"Volume surge {row['tx_count']} tx/hr, but decline rate is healthy ({row['decline_rate'] * 100:.1f}%).")
            elif row["iforest_anomaly"] == 1 and (row["decline_rate"] > 0.10 or row["new_device_ratio"] > 0.70):
                flag = 1
                cat = "MODERATE: MULTIVARIATE RISK ANOMALY"
                rec = "INSPECT RECENT MERCHANT LOGS"
                reasons.append(f"Isolation forest anomaly score: {row['iforest_score']:.3f}.")
            else:
                cat = "NORMAL"
                reasons.append("Metrics within standard operational tolerances.")

            is_attack.append(flag)
            threat_categories.append(cat)
            explanations.append("; ".join(reasons))
            recommendations.append(rec)

        out["is_flagged_threat"] = is_attack
        out["threat_category"] = threat_categories
        out["threat_explanation"] = explanations
        out["recommended_action"] = recommendations
        return out


def plot_time_series_anomaly_report(
        df_analyzed: pd.DataFrame,
        save_path: Optional[str] = None
) -> plt.Figure:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=200)

    ax1.plot(df_analyzed["timestamp"], df_analyzed["tx_count"], color="#2b6cb0", lw=1.2, label="Hourly Transactions")
    ax1.plot(df_analyzed["timestamp"], df_analyzed["volume_ewma_mean"], color="#718096", lw=1.5, linestyle="--",
             label="EWMA Baseline")
    ax1.plot(df_analyzed["timestamp"], df_analyzed["volume_upper_envelope"], color="#e53e3e", lw=1.0, linestyle=":",
             label="Upper Dynamic Limit (k=3.5σ)")

    attacks = df_analyzed[df_analyzed["is_flagged_threat"] == 1]
    if len(attacks) > 0:
        ax1.scatter(attacks["timestamp"], attacks["tx_count"], color="#e53e3e", s=45, zorder=5,
                    label="Flagged Risk Threat")

    ax1.set_ylabel("Tx Count / Hour", fontsize=10, weight="bold")
    ax1.set_title(f"Merchant Anomaly & Velocity Monitor ({df_analyzed['merchant_id'].iloc[0]})", fontsize=12,
                  weight="bold", color="#1a202c")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    ax2.plot(df_analyzed["timestamp"], df_analyzed["avg_ticket_size"], color="#319795", lw=1.2,
             label="Avg Ticket Size ($)")
    ax2.set_ylabel("Avg Ticket ($)", fontsize=10, weight="bold", color="#319795")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(df_analyzed["timestamp"], df_analyzed["micro_ticket_ratio"] * 100, color="#dd6b20", lw=1.2,
                  linestyle="--", label="Micro-Charges < $3 (%)")
    ax2_twin.set_ylabel("Micro-Ticket %", fontsize=10, weight="bold", color="#dd6b20")
    ax2_twin.grid(False)

    ax3.plot(df_analyzed["timestamp"], df_analyzed["decline_rate"] * 100, color="#805ad5", lw=1.2,
             label="Decline Rate (%)")
    ax3.axhline(5.0, color="#e53e3e", linestyle=":", lw=1.2, label="Decline Warning Threshold (5%)")
    ax3.set_ylabel("Decline Rate (%)", fontsize=10, weight="bold", color="#805ad5")
    ax3.set_xlabel("Time Horizon", fontsize=10, weight="bold")
    ax3.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig