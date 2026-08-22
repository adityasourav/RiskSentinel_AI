"""
Calibrated Gradient Boosted Classifier for Checkout Risk Scoring
Trains and evaluates calibrated ensemble models with feature explainability for BFSI compliance.
"""

import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

FEATURE_COLUMNS = [
    "order_amount",
    "item_count",
    "distinct_categories",
    "high_resale_ratio",
    "wardrobing_flag",
    "discount_percentage",
    "account_age_days",
    "historical_orders",
    "historical_return_rate",
    "historical_chargebacks",
    "delivery_address_changes_30d",
    "billing_shipping_mismatch",
    "ip_distance_to_billing_km",
    "device_trust_score",
    "checkout_duration_sec",
    "payment_velocity_24h",
    "card_is_prepaid",
    "card_country_mismatch"
]

FEATURE_LABELS = {
    "order_amount": "Order Value ($)",
    "item_count": "Cart Item Count",
    "distinct_categories": "Distinct SKU Categories",
    "high_resale_ratio": "High-Resale Item Ratio",
    "wardrobing_flag": "Wardrobing Multi-Size Indicator",
    "discount_percentage": "Promo / Discount Rate",
    "account_age_days": "Account Age (Days)",
    "historical_orders": "Historical Completed Orders",
    "historical_return_rate": "Past Return Rate",
    "historical_chargebacks": "Past Chargeback Disputes",
    "delivery_address_changes_30d": "Address Changes (30d)",
    "billing_shipping_mismatch": "Billing/Shipping Mismatch",
    "ip_distance_to_billing_km": "IP-to-Billing Distance (km)",
    "device_trust_score": "Device Fingerprint Trust",
    "checkout_duration_sec": "Checkout Dwell Time (s)",
    "payment_velocity_24h": "Card Velocity (24h)",
    "card_is_prepaid": "Prepaid Card Indicator",
    "card_country_mismatch": "Cross-Border Card Mismatch"
}

FEATURE_DEFAULTS = {
    "order_amount": 75.0,
    "item_count": 2,
    "distinct_categories": 1,
    "high_resale_ratio": 0.20,
    "wardrobing_flag": 0,
    "discount_percentage": 0.10,
    "account_age_days": 180.0,
    "historical_orders": 5,
    "historical_return_rate": 0.10,
    "historical_chargebacks": 0,
    "delivery_address_changes_30d": 0,
    "billing_shipping_mismatch": 0,
    "ip_distance_to_billing_km": 20.0,
    "device_trust_score": 0.85,
    "checkout_duration_sec": 60.0,
    "payment_velocity_24h": 1,
    "card_is_prepaid": 0,
    "card_country_mismatch": 0
}


class RiskClassifier:
    def __init__(
            self,
            model_type: str = "hist_gbdt",
            calibrate: bool = True,
            target_col: str = "is_risk_target",
            random_state: int = 42
    ):
        self.model_type = model_type
        self.calibrate = calibrate
        self.target_col = target_col
        self.random_state = random_state
        self.feature_names = FEATURE_COLUMNS
        self.model = None
        self.is_fitted = False
        self.feature_importance_df = None

        if model_type == "xgboost" and HAS_XGBOOST:
            base_model = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=4.0,
                random_state=random_state,
                eval_metric="logloss"
            )
        elif model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                class_weight="balanced",
                random_state=random_state
            )
        else:
            base_model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=5,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=random_state
            )

        if self.calibrate:
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                method="isotonic",
                cv=3
            )
        else:
            self.model = base_model

    def train(
            self,
            train_df: pd.DataFrame,
            val_df: Optional[pd.DataFrame] = None
    ) -> "RiskClassifier":
        X_train = train_df[self.feature_names].values
        y_train = train_df[self.target_col].values

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        eval_df = val_df if val_df is not None else train_df
        X_eval = eval_df[self.feature_names].values
        y_eval = eval_df[self.target_col].values

        perm = permutation_importance(
            self.model,
            X_eval,
            y_eval,
            n_repeats=5,
            random_state=self.random_state,
            scoring="roc_auc"
        )

        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "feature_label": [FEATURE_LABELS.get(f, f) for f in self.feature_names],
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        self.feature_importance_df = imp_df
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray | Dict[str, Any]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if isinstance(X, pd.DataFrame):
            df_full = X.copy()
            for col, val in FEATURE_DEFAULTS.items():
                if col not in df_full.columns:
                    df_full[col] = val
            X_mat = df_full[self.feature_names].values
        else:
            X_mat = X
        return self.model.predict_proba(X_mat)[:, 1]

    def predict(self, X: pd.DataFrame | np.ndarray | Dict[str, Any], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def explain_order(self, order_series: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(order_series, dict):
            row = pd.Series(order_series)
        else:
            row = order_series

        prob = float(self.predict_proba(pd.DataFrame([row]))[0])

        risk_factors = []
        if float(row.get("historical_chargebacks", 0)) > 0:
            risk_factors.append(
                f"Customer has {int(row.get('historical_chargebacks', 0))} prior chargeback disputes on record.")
        if float(row.get("historical_return_rate", 0)) > 0.40:
            risk_factors.append(
                f"High historical return rate ({float(row.get('historical_return_rate', 0)) * 100:.1f}%).")
        if int(row.get("wardrobing_flag", 0)) == 1:
            risk_factors.append("Wardrobing signature: Multi-size purchasing of identical SKU items.")
        if float(row.get("device_trust_score", 1.0)) < 0.40:
            risk_factors.append(
                f"Low device trust score ({float(row.get('device_trust_score', 0)):.2f}) indicating proxy or emulator.")
        if int(row.get("card_is_prepaid", 0)) == 1:
            risk_factors.append("Payment executed using an anonymous prepaid debit card.")
        if int(row.get("card_country_mismatch", 0)) == 1:
            risk_factors.append("Cross-border card issuing country mismatch.")
        if int(row.get("payment_velocity_24h", 1)) >= 3:
            risk_factors.append(
                f"High card velocity: {int(row.get('payment_velocity_24h', 1))} checkout attempts in 24 hours.")
        if float(row.get("high_resale_ratio", 0)) > 0.60 and float(row.get("order_amount", 0)) > 200:
            risk_factors.append(
                f"High-liquidity resale goods cart (${float(row.get('order_amount', 0)):.2f}, {float(row.get('high_resale_ratio', 0)) * 100:.0f}% resale items).")
        if float(row.get("checkout_duration_sec", 60)) < 7.0:
            risk_factors.append(
                f"Abnormally rapid checkout duration ({float(row.get('checkout_duration_sec', 0)):.1f}s) suggesting scripted automation.")

        if not risk_factors:
            risk_factors.append(
                "Transaction exhibits standard legitimate buyer patterns across all velocity and identity dimensions.")

        return {
            "risk_probability": prob,
            "risk_tier": "CRITICAL RISK" if prob > 0.65 else ("MODERATE RISK" if prob > 0.25 else "LOW RISK"),
            "recommended_action": "BLOCK & REVIEW" if prob > 0.65 else (
                "STEP-UP 3DS AUTH" if prob > 0.25 else "APPROVE FRICTIONLESS"),
            "risk_factors": risk_factors
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_type": self.model_type,
                "calibrate": self.calibrate,
                "target_col": self.target_col,
                "feature_names": self.feature_names,
                "feature_importance_df": self.feature_importance_df,
                "is_fitted": self.is_fitted
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "RiskClassifier":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(model_type=data["model_type"], calibrate=data["calibrate"], target_col=data["target_col"])
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        obj.feature_importance_df = data["feature_importance_df"]
        obj.is_fitted = data["is_fitted"]
        return obj