"""
Unified Dual-Tier AI Risk Engine
Combines:
  Tier 1: Macro Merchant-Level Time-Series Velocity & Fraud-Spike Anomaly Detector
  Tier 2: Micro Order-Level Calibrated GBDT Return & Fraud Risk Scorer with Cost Optimization
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from riskguard.models.classifier import RiskClassifier
from riskguard.models.cost_optimizer import CostOptimizer, CostMatrix
from riskguard.models.time_series import InterpretableFraudSpikeDetector


class UnifiedRiskEngine:
    def __init__(
            self,
            classifier: RiskClassifier,
            cost_optimizer: CostOptimizer,
            spike_detector: Optional[InterpretableFraudSpikeDetector] = None
    ):
        self.classifier = classifier
        self.cost_optimizer = cost_optimizer
        self.spike_detector = spike_detector or InterpretableFraudSpikeDetector(ewma_span=48, z_threshold=3.5)
        self.current_merchant_state: Dict[str, Dict[str, Any]] = {}

    def update_merchant_velocity_state(
            self,
            merchant_id: str,
            recent_hourly_stream: pd.DataFrame
    ) -> Dict[str, Any]:
        analyzed = self.spike_detector.analyze_stream(recent_hourly_stream)
        latest_record = analyzed.iloc[-1]

        threat_cat = latest_record["threat_category"]
        is_threat = bool(latest_record["is_flagged_threat"] == 1)

        if "CARD TESTING" in threat_cat:
            threat_level = "CRITICAL_CARD_TESTING"
            risk_multiplier = 1.35
            threshold_adjustment = -0.08
        elif "CREDENTIAL STUFFING" in threat_cat or "VELOCITY" in threat_cat:
            threat_level = "HIGH_VELOCITY_SURGE"
            risk_multiplier = 1.20
            threshold_adjustment = -0.05
        elif "ORGANIC" in threat_cat:
            threat_level = "ORGANIC_PROMO_EVENT"
            risk_multiplier = 0.95
            threshold_adjustment = +0.02
        else:
            threat_level = "NORMAL_BASELINE"
            risk_multiplier = 1.00
            threshold_adjustment = 0.00

        state = {
            "merchant_id": merchant_id,
            "timestamp": str(latest_record["timestamp"]),
            "current_volume_per_hr": int(latest_record["tx_count"]),
            "volume_ewma_baseline": round(float(latest_record["volume_ewma_mean"]), 1),
            "robust_zscore": float(latest_record["robust_zscore"]),
            "decline_rate_pct": round(float(latest_record["decline_rate"]) * 100, 1),
            "micro_ticket_ratio_pct": round(float(latest_record["micro_ticket_ratio"]) * 100, 1),
            "threat_level": threat_level,
            "is_threat_active": is_threat,
            "risk_multiplier": risk_multiplier,
            "threshold_adjustment": threshold_adjustment,
            "explanation": latest_record["threat_explanation"],
            "recommended_merchant_policy": latest_record["recommended_action"]
        }
        self.current_merchant_state[merchant_id] = state
        return state

    def score_checkout_order(
            self,
            order_features: Dict[str, Any],
            merchant_id: Optional[str] = None,
            base_decision_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        row = pd.Series(order_features)
        base_prob = float(self.classifier.predict_proba(pd.DataFrame([row]))[0])

        merchant_state = self.current_merchant_state.get(merchant_id, {
            "threat_level": "NORMAL_BASELINE",
            "risk_multiplier": 1.00,
            "threshold_adjustment": 0.00,
            "is_threat_active": False,
            "explanation": "Normal merchant velocity baseline."
        })

        effective_prob = base_prob
        context_factors = []

        if merchant_state["threat_level"] == "CRITICAL_CARD_TESTING":
            if float(row.get("order_amount", 50)) < 10.0 or float(row.get("device_trust_score", 1.0)) < 0.40:
                effective_prob = min(0.99, base_prob * 1.5 + 0.25)
                context_factors.append(
                    "⚠️ MACRO ALERT: Merchant is under active Card-Testing Bot Attack. Micro-charge / unverified device flagged.")
        elif merchant_state["threat_level"] == "HIGH_VELOCITY_SURGE":
            if int(row.get("payment_velocity_24h", 1)) >= 3:
                effective_prob = min(0.99, base_prob * 1.3)
                context_factors.append(
                    "⚠️ MACRO ALERT: Merchant velocity surge detected; high customer card velocity amplified.")
        elif merchant_state["threat_level"] == "ORGANIC_PROMO_EVENT":
            context_factors.append(
                "ℹ️ MACRO CONTEXT: Merchant active flash sale. Standard frictionless checkout prioritized.")

        optimal_th = base_decision_threshold or self.cost_optimizer.find_empirical_optimal_threshold(
            np.array([0, 1]), np.array([0.1, 0.9])
        )[0]
        active_th = max(0.05, min(0.85, optimal_th + merchant_state.get("threshold_adjustment", 0.0)))

        is_flagged = bool(effective_prob >= active_th)
        if effective_prob > 0.65:
            tier = "CRITICAL RISK"
            action = "BLOCK & MANUAL REVIEW"
        elif effective_prob >= active_th:
            tier = "MODERATE RISK"
            action = "TRIGGER STEP-UP 3DS AUTHENTICATION"
        else:
            tier = "LOW RISK"
            action = "APPROVE FRICTIONLESS"

        explanation = self.classifier.explain_order(order_features)
        all_factors = context_factors + explanation["risk_factors"]

        return {
            "order_id": order_features.get("order_id", "ORD-NEW"),
            "merchant_id": merchant_id or "MERCHANT-DEFAULT",
            "merchant_threat_state": merchant_state["threat_level"],
            "base_risk_probability": round(base_prob, 4),
            "effective_risk_probability": round(effective_prob, 4),
            "active_decision_threshold": round(active_th, 4),
            "decision": "INTERCEPT" if is_flagged else "ALLOW",
            "risk_tier": tier,
            "recommended_action": action,
            "risk_factors": all_factors
        }