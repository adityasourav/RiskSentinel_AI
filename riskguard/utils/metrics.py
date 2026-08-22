"""
Evaluation Metrics & Visualization Utilities for Risk Analysis
Computes rigorous ML benchmarks (ROC-AUC, PR-AUC, Confusion Matrix, Expected Cost)
and generates publication-quality plots for financial risk modeling.
"""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def compute_all_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        c_fn: float = 85.0,
        c_fp: float = 28.0,
        c_tp: float = 4.0,
        c_tn: float = 0.0
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr_arr, tpr_arr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)

    total_cost = (fn * c_fn) + (fp * c_fp) + (tp * c_tp) + (tn * c_tn)
    cost_per_tx = total_cost / len(y_true)

    n_positives = int(np.sum(y_true == 1))
    n_negatives = int(np.sum(y_true == 0))
    cost_allow_all = n_positives * c_fn
    cost_block_all = (n_negatives * c_fp) + (n_positives * c_tp)

    savings_vs_allow_all = cost_allow_all - total_cost
    savings_pct = (savings_vs_allow_all / cost_allow_all * 100.0) if cost_allow_all > 0 else 0.0

    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_val),
        "pr_auc": float(pr_auc_val),
        "total_cost": float(total_cost),
        "cost_per_tx": float(cost_per_tx),
        "cost_allow_all": float(cost_allow_all),
        "cost_block_all": float(cost_block_all),
        "savings_vs_allow_all": float(savings_vs_allow_all),
        "savings_pct": float(savings_pct)
    }


def plot_confusion_matrix(
        cm: np.ndarray,
        threshold: float,
        save_path: Optional[str] = None,
        title_suffix: str = ""
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    labels = [["TN (Legit Allowed)", "FP (Legit Intercepted)"],
              ["FN (Fraud Missed)", "TP (Fraud Intercepted)"]]

    annot = np.array([
        [f"{labels[0][0]}\n{cm[0, 0]:,}", f"{labels[0][1]}\n{cm[0, 1]:,}"],
        [f"{labels[1][0]}\n{cm[1, 0]:,}", f"{labels[1][1]}\n{cm[1, 1]:,}"]
    ])

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues", cbar=False,
        xticklabels=["Predicted Legit", "Predicted Risk"],
        yticklabels=["Actual Legit", "Actual Risk"],
        ax=ax, linewidths=1.5, linecolor="white", annot_kws={"fontsize": 11, "weight": "bold"}
    )

    ax.set_title(f"Confusion Matrix (Threshold θ = {threshold:.3f}) {title_suffix}", fontsize=13, pad=12, weight="bold",
                 color="#1a202c")
    ax.set_ylabel("Ground Truth", fontsize=11, weight="bold")
    ax.set_xlabel("Model Decision", fontsize=11, weight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        optimal_threshold: Optional[float] = None,
        save_path: Optional[str] = None
) -> plt.Figure:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    ax.plot(fpr, tpr, color="#2b6cb0", lw=2.5, label=f"Risk Classifier (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#a0aec0", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

    if optimal_threshold is not None:
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        ax.scatter(fpr[idx], tpr[idx], color="#e53e3e", s=120, zorder=5,
                   label=f"Optimal Threshold θ* = {optimal_threshold:.2f} (FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f})")
        ax.annotate(f"  θ* = {optimal_threshold:.2f}", (fpr[idx], tpr[idx]), fontsize=10, weight="bold",
                    color="#c53030")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, weight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, pad=12, weight="bold", color="#1a202c")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e0", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pr_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        optimal_threshold: Optional[float] = None,
        save_path: Optional[str] = None
) -> plt.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    baseline_prec = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    ax.plot(recall, precision, color="#319795", lw=2.5, label=f"Risk Classifier (PR-AUC = {pr_auc_val:.4f})")
    ax.axhline(baseline_prec, color="#a0aec0", linestyle="--", lw=1.5,
               label=f"Prevalence Baseline ({baseline_prec * 100:.2f}%)")

    if optimal_threshold is not None and len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        ax.scatter(recall[idx], precision[idx], color="#dd6b20", s=120, zorder=5,
                   label=f"Operating Point θ* = {optimal_threshold:.2f} (Rec={recall[idx]:.3f}, Prec={precision[idx]:.3f})")
        ax.annotate(f"  θ* = {optimal_threshold:.2f}", (recall[idx], precision[idx]), fontsize=10, weight="bold",
                    color="#c05621")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Recall (Fraud / Return Abuse Captured)", fontsize=11, weight="bold")
    ax.set_ylabel("Precision (Purity of Flagged Transactions)", fontsize=11, weight="bold")
    ax.set_title("Precision-Recall Curve (Severe Class Imbalance Evaluation)", fontsize=13, pad=12, weight="bold",
                 color="#1a202c")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e0", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_cost_vs_threshold(
        sweep_df: pd.DataFrame,
        optimal_row: pd.Series,
        save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=200)

    ax1.plot(sweep_df["threshold"], sweep_df["total_cost"], color="#805ad5", lw=2.8, label="Total Financial Cost ($)")
    allow_all_cost = float(sweep_df["cost_allow_all"].iloc[0])
    ax1.axhline(allow_all_cost, color="#e53e3e", linestyle="--", lw=1.8,
                label=f"Allow-All Baseline (${allow_all_cost:,.0f})")

    opt_theta = float(optimal_row["threshold"])
    min_cost = float(optimal_row["total_cost"])
    ax1.scatter([opt_theta], [min_cost], color="#38a169", s=140, zorder=6,
                label=f"Optimal Threshold θ* = {opt_theta:.2f} (${min_cost:,.0f})")

    ax1.annotate(
        f"  Minimum Loss\n  θ* = {opt_theta:.2f}\n  Cost = ${min_cost:,.0f}",
        xy=(opt_theta, min_cost),
        xytext=(opt_theta + 0.08, min_cost + (allow_all_cost - min_cost) * 0.15),
        arrowprops=dict(facecolor="#276749", shrink=0.08, width=1.5, headwidth=8),
        fontsize=10, weight="bold", color="#22543d"
    )

    ax1.set_xlabel("Decision Threshold θ (Interception Cutoff)", fontsize=11, weight="bold")
    ax1.set_ylabel("Total Financial Loss on Test Set ($)", fontsize=11, weight="bold", color="#805ad5")
    ax1.tick_params(axis='y', labelcolor="#805ad5")

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["savings_pct"], color="#319795", lw=2.0, linestyle=":",
             label="Cost Savings (% vs Allow All)")
    ax2.set_ylabel("Savings vs Allow-All (%)", fontsize=11, weight="bold", color="#319795")
    ax2.tick_params(axis='y', labelcolor="#319795")
    ax2.grid(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", frameon=True, facecolor="white",
               edgecolor="#cbd5e0", fontsize=9.5)

    plt.title("Cost-Utility Threshold Optimization Curve", fontsize=13, pad=12, weight="bold", color="#1a202c")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig