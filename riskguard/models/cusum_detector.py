"""
Cumulative Sum (CUSUM) & Changepoint Detection Engine
Provides early detection of subtle, persistent shifts in transaction velocity.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


class CUSUMDetector:
    def __init__(self, drift_allowance: float = 0.5, decision_interval: float = 5.0):
        self.drift_allowance = drift_allowance
        self.decision_interval = decision_interval

    def detect_shifts(self, series: np.ndarray | pd.Series, baseline_mean: Optional[float] = None,
                      baseline_std: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vals = np.asarray(series, dtype=float)
        n = len(vals)

        mu = baseline_mean if baseline_mean is not None else np.mean(vals[:min(48, n)])
        sigma = baseline_std if baseline_std is not None else np.std(vals[:min(48, n)])
        sigma = max(sigma, 1e-4)

        standardized = (vals - mu) / sigma

        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        alarms = np.zeros(n, dtype=int)

        for t in range(1, n):
            s_pos[t] = max(0.0, s_pos[t - 1] + standardized[t] - self.drift_allowance)
            s_neg[t] = max(0.0, s_neg[t - 1] - standardized[t] - self.drift_allowance)

            if s_pos[t] > self.decision_interval or s_neg[t] > self.decision_interval:
                alarms[t] = 1

        return s_pos, s_neg, alarms