"""
Probability Calibration Benchmark & Reliability Diagram Engine
Computes Expected Calibration Error (ECE), Brier Score, and plots Reliability Curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Dict, Any, Tuple, Optional


def compute_expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    n_samples = len(y_true)
    ece = 0.0
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (bin_assignments == b)
        count = np.sum(mask)
        bin_counts[b] = count
        if count > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            bin_accuracies[b] = acc
            bin_confidences[b] = conf
            ece += (count / n_samples) * np.abs(acc - conf)
        else:
            bin_accuracies[b] = (bin_edges[b] + bin_edges[b + 1]) / 2.0
            bin_confidences[b] = (bin_edges[b] + bin_edges[b + 1]) / 2.0

    return float(ece), bin_accuracies, bin_confidences, bin_counts


def plot_calibration_comparison(
        y_true: np.ndarray,
        uncalibrated_probs: np.ndarray,
        calibrated_probs: np.ndarray,
        save_path: Optional[str] = None
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)

    ece_uncal, _, _, _ = compute_expected_calibration_error(y_true, uncalibrated_probs, n_bins=10)
    brier_uncal = brier_score_loss(y_true, uncalibrated_probs)
    prob_true_u, prob_pred_u = calibration_curve(y_true, uncalibrated_probs, n_bins=10)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfect Calibration (Ideal)")
    ax1.plot(prob_pred_u, prob_true_u, "s-", color="#e53e3e", lw=2, label=f"Uncalibrated GBDT (ECE = {ece_uncal:.4f})")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=10, weight="bold")
    ax1.set_ylabel("Empirical True Fraction of Positives", fontsize=10, weight="bold")
    ax1.set_title(f"Uncalibrated Model\n(Brier Score: {brier_uncal:.4f}, ECE: {ece_uncal:.4f})", fontsize=11,
                  weight="bold")
    ax1.legend(loc="lower right", frameon=True, facecolor="white")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    ece_cal, _, _, _ = compute_expected_calibration_error(y_true, calibrated_probs, n_bins=10)
    brier_cal = brier_score_loss(y_true, calibrated_probs)
    prob_true_c, prob_pred_c = calibration_curve(y_true, calibrated_probs, n_bins=10)

    ax2.plot([0, 1], [0, 1], "k:", label="Perfect Calibration (Ideal)")
    ax2.plot(prob_pred_c, prob_true_c, "o-", color="#319795", lw=2.2,
             label=f"Isotonic Calibrated (ECE = {ece_cal:.4f})")
    ax2.set_xlabel("Mean Predicted Probability", fontsize=10, weight="bold")
    ax2.set_ylabel("Empirical True Fraction of Positives", fontsize=10, weight="bold")
    ax2.set_title(f"RiskGuard Calibrated Model\n(Brier Score: {brier_cal:.4f}, ECE: {ece_cal:.4f})", fontsize=11,
                  weight="bold")
    ax2.legend(loc="lower right", frameon=True, facecolor="white")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig