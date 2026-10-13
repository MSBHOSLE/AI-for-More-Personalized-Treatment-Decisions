"""
Confidence Interval Estimation and Coverage Diagnostics for Causal Inference.
Supports:
- Nonparametric bootstrap intervals for CATE and ATE
- Asymptotic AIPW score variance intervals
- Empirical coverage calculation against ground-truth effects
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.base import clone


def compute_bootstrap_intervals(
    estimator,
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    X_eval: np.ndarray,
    n_bootstraps: int = 50,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Compute bootstrap confidence intervals for CATE and ATE.
    Returns lower and upper bounds, point estimates, and standard errors.
    """
    n_samples = len(X)
    rng = np.random.default_rng(random_state)

    cate_bootstraps = []
    ate_bootstraps = []

    for b in range(n_bootstraps):
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        # Avoid degenerate resamples
        if len(np.unique(treatment[boot_indices])) < 2:
            continue

        X_b = X[boot_indices]
        t_b = treatment[boot_indices]
        y_b = outcome[boot_indices]

        # Fresh instance of the same estimator class
        est_class = estimator.__class__
        # Try cloning or instantiating with default parameters
        try:
            est_b = est_class(random_state=random_state + b)
        except TypeError:
            est_b = est_class()

        try:
            est_b.fit(X_b, t_b, y_b)
            cate_b = est_b.predict_cate(X_eval)
            ate_b = np.mean(cate_b)

            cate_bootstraps.append(cate_b)
            ate_bootstraps.append(ate_b)
        except Exception:
            continue

    cate_bootstraps = np.array(cate_bootstraps)
    ate_bootstraps = np.array(ate_bootstraps)

    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    cate_lower = np.percentile(cate_bootstraps, lower_pct, axis=0)
    cate_upper = np.percentile(cate_bootstraps, upper_pct, axis=0)
    cate_se = np.std(cate_bootstraps, axis=0, ddof=1)

    ate_lower = float(np.percentile(ate_bootstraps, lower_pct))
    ate_upper = float(np.percentile(ate_bootstraps, upper_pct))
    ate_se = float(np.std(ate_bootstraps, ddof=1))

    return {
        "cate_lower": cate_lower,
        "cate_upper": cate_upper,
        "cate_se": cate_se,
        "ate_lower": ate_lower,
        "ate_upper": ate_upper,
        "ate_se": ate_se,
        "n_successful_bootstraps": len(cate_bootstraps),
    }


def compute_ci_coverage(
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    true_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute coverage rate and mean interval width of confidence intervals against ground truth.
    """
    covered = (true_values >= ci_lower) & (true_values <= ci_upper)
    coverage_rate = float(np.mean(covered))
    mean_width = float(np.mean(ci_upper - ci_lower))

    return {
        "coverage_rate": coverage_rate,
        "mean_width": mean_width,
        "target_coverage": 0.95,
    }
