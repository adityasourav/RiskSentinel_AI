"""
Cost-Utility Matrix & Decision Threshold Optimization Engine
Formalizes financial risk tradeoffs (False Positive Customer Friction vs False Negative Fraud Loss)
and derives cost-optimal decision boundaries.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from riskguard.utils.metrics import compute_all_metrics


@dataclass
class CostMatrix:
    c_fn: float = 85.0
    c_fp: float = 28.0
    c_tp: float = 4.0
    c_tn: float = 0.0


class CostOptimizer:
    def __init__(self, cost_matrix: Optional[CostMatrix] = None):
        self.cost_matrix = cost_matrix or CostMatrix()

    def theoretical_optimal_threshold(self) -> float:
        num = self.cost_matrix.c_fp - self.cost_matrix.c_tn
        den = (self.cost_matrix.c_fp - self.cost_matrix.c_tn) + (self.cost_matrix.c_fn - self.cost_matrix.c_tp)
        if den <= 0:
            return 0.5
        return float(num / den)

    def sweep_thresholds(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            steps: int = 100
    ) -> pd.DataFrame:
        thresholds = np.linspace(0.01, 0.99, steps)
        records: List[Dict[str, Any]] = []

        for th in thresholds:
            res = compute_all_metrics(
                y_true=y_true,
                y_prob=y_prob,
                threshold=th,
                c_fn=self.cost_matrix.c_fn,
                c_fp=self.cost_matrix.c_fp,
                c_tp=self.cost_matrix.c_tp,
                c_tn=self.cost_matrix.c_tn
            )
            records.append(res)

        return pd.DataFrame(records)

    def find_empirical_optimal_threshold(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray
    ) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
        sweep_df = self.sweep_thresholds(y_true, y_prob, steps=200)
        best_idx = sweep_df["total_cost"].idxmin()
        best_row = sweep_df.loc[best_idx].to_dict()
        best_threshold = float(best_row["threshold"])
        return best_threshold, best_row, sweep_df

    def compare_policies(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            heuristic_predictions: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        opt_th, opt_metrics, _ = self.find_empirical_optimal_threshold(y_true, y_prob)
        theo_th = self.theoretical_optimal_threshold()

        metrics_allow_all = compute_all_metrics(y_true, y_prob, threshold=1.0, **self.cost_matrix.__dict__)
        metrics_block_all = compute_all_metrics(y_true, y_prob, threshold=0.0, **self.cost_matrix.__dict__)
        metrics_default_ml = compute_all_metrics(y_true, y_prob, threshold=0.50, **self.cost_matrix.__dict__)
        metrics_theo_ml = compute_all_metrics(y_true, y_prob, threshold=theo_th, **self.cost_matrix.__dict__)
        metrics_opt_ml = compute_all_metrics(y_true, y_prob, threshold=opt_th, **self.cost_matrix.__dict__)

        policies = [
            {
                "Policy": "1. Allow All (Baseline)",
                "Threshold θ": "1.00",
                "Recall (Fraud Caught)": f"{metrics_allow_all['recall'] * 100:.1f}%",
                "Precision": f"{metrics_allow_all['precision'] * 100:.1f}%",
                "False Positives": metrics_allow_all["confusion_matrix"]["fp"],
                "False Negatives": metrics_allow_all["confusion_matrix"]["fn"],
                "Total Financial Loss": f"${metrics_allow_all['total_cost']:,.0f}",
                "Net Savings ($)": "$0",
                "Savings (%)": "0.0%"
            },
            {
                "Policy": "2. Block All (Over-conservative)",
                "Threshold θ": "0.00",
                "Recall (Fraud Caught)": f"{metrics_block_all['recall'] * 100:.1f}%",
                "Precision": f"{metrics_block_all['precision'] * 100:.1f}%",
                "False Positives": metrics_block_all["confusion_matrix"]["fp"],
                "False Negatives": metrics_block_all["confusion_matrix"]["fn"],
                "Total Financial Loss": f"${metrics_block_all['total_cost']:,.0f}",
                "Net Savings ($)": f"${metrics_block_all['savings_vs_allow_all']:,.0f}",
                "Savings (%)": f"{metrics_block_all['savings_pct']:.1f}%"
            },
            {
                "Policy": "3. Default ML (θ = 0.50)",
                "Threshold θ": "0.50",
                "Recall (Fraud Caught)": f"{metrics_default_ml['recall'] * 100:.1f}%",
                "Precision": f"{metrics_default_ml['precision'] * 100:.1f}%",
                "False Positives": metrics_default_ml["confusion_matrix"]["fp"],
                "False Negatives": metrics_default_ml["confusion_matrix"]["fn"],
                "Total Financial Loss": f"${metrics_default_ml['total_cost']:,.0f}",
                "Net Savings ($)": f"${metrics_default_ml['savings_vs_allow_all']:,.0f}",
                "Savings (%)": f"{metrics_default_ml['savings_pct']:.1f}%"
            },
            {
                "Policy": f"4. Theoretical Bayes (θ = {theo_th:.2f})",
                "Threshold θ": f"{theo_th:.2f}",
                "Recall (Fraud Caught)": f"{metrics_theo_ml['recall'] * 100:.1f}%",
                "Precision": f"{metrics_theo_ml['precision'] * 100:.1f}%",
                "False Positives": metrics_theo_ml["confusion_matrix"]["fp"],
                "False Negatives": metrics_theo_ml["confusion_matrix"]["fn"],
                "Total Financial Loss": f"${metrics_theo_ml['total_cost']:,.0f}",
                "Net Savings ($)": f"${metrics_theo_ml['savings_vs_allow_all']:,.0f}",
                "Savings (%)": f"{metrics_theo_ml['savings_pct']:.1f}%"
            },
            {
                "Policy": f"5. RiskGuard Optimal ML (θ* = {opt_th:.2f})",
                "Threshold θ": f"{opt_th:.2f}",
                "Recall (Fraud Caught)": f"{metrics_opt_ml['recall'] * 100:.1f}%",
                "Precision": f"{metrics_opt_ml['precision'] * 100:.1f}%",
                "False Positives": metrics_opt_ml["confusion_matrix"]["fp"],
                "False Negatives": metrics_opt_ml["confusion_matrix"]["fn"],
                "Total Financial Loss": f"${metrics_opt_ml['total_cost']:,.0f}",
                "Net Savings ($)": f"${metrics_opt_ml['savings_vs_allow_all']:,.0f}",
                "Savings (%)": f"{metrics_opt_ml['savings_pct']:.1f}%"
            }
        ]

        if heuristic_predictions is not None:
            h_metrics = compute_all_metrics(y_true, heuristic_predictions.astype(float), threshold=0.5,
                                            **self.cost_matrix.__dict__)
            policies.append({
                "Policy": "6. Naive Rule Engine",
                "Threshold θ": "N/A",
                "Recall (Fraud Caught)": f"{h_metrics['recall'] * 100:.1f}%",
                "Precision": f"{h_metrics['precision'] * 100:.1f}%",
                "False Positives": h_metrics["confusion_matrix"]["fp"],
                "False Negatives": h_metrics["confusion_matrix"]["fn"],
                "Total Financial Loss": f"${h_metrics['total_cost']:,.0f}",
                "Net Savings ($)": f"${h_metrics['savings_vs_allow_all']:,.0f}",
                "Savings (%)": f"{h_metrics['savings_pct']:.1f}%"
            })

        return pd.DataFrame(policies)