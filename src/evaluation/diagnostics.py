"""
Causal Diagnostics & Covariate Balance Analysis.
Includes:
- Standardized Mean Differences (SMD) before and after propensity weighting (Love Plot analysis)
- Positivity violation checks and overlap metrics
- Effective sample size (ESS) computation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_smd_unweighted(
    X: np.ndarray,
    treatment: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute unweighted Standardized Mean Differences between treated and control groups."""
    t1 = treatment == 1
    t0 = treatment == 0

    records = []
    for idx, name in enumerate(feature_names):
        col1 = X[t1, idx]
        col0 = X[t0, idx]

        m1, m0 = np.mean(col1), np.mean(col0)
        v1, v0 = np.var(col1, ddof=1), np.var(col0, ddof=1)
        pooled_sd = np.sqrt((v1 + v0) / 2.0)

        smd = (m1 - m0) / (pooled_sd + 1e-8)
        records.append(
            {
                "feature": name,
                "mean_treated": float(m1),
                "mean_control": float(m0),
                "pooled_sd": float(pooled_sd),
                "unweighted_smd": float(smd),
                "abs_unweighted_smd": float(np.abs(smd)),
            }
        )
    return pd.DataFrame(records)


def compute_smd_weighted(
    X: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
    feature_names: List[str],
    stabilize: bool = True,
) -> pd.DataFrame:
    """
    Compute IPW-weighted Standardized Mean Differences (Love plot diagnostics).
    w_i = T_i / e_i + (1 - T_i) / (1 - e_i)
    """
    ps = np.clip(propensity_scores, 0.01, 0.99)
    weights = np.where(treatment == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    if stabilize:
        p_t = np.mean(treatment)
        weights = np.where(treatment == 1, weights * p_t, weights * (1.0 - p_t))

    t1 = treatment == 1
    t0 = treatment == 0

    w1 = weights[t1]
    w0 = weights[t0]

    unw_df = compute_smd_unweighted(X, treatment, feature_names)

    weighted_smds = []
    for idx, row in unw_df.iterrows():
        feat_idx = idx
        col1 = X[t1, feat_idx]
        col0 = X[t0, feat_idx]

        w_mean_1 = np.sum(w1 * col1) / np.sum(w1)
        w_mean_0 = np.sum(w0 * col0) / np.sum(w0)

        pooled_sd = row["pooled_sd"]
        w_smd = (w_mean_1 - w_mean_0) / (pooled_sd + 1e-8)
        weighted_smds.append(
            {
                "feature": row["feature"],
                "unweighted_smd": row["unweighted_smd"],
                "abs_unweighted_smd": row["abs_unweighted_smd"],
                "weighted_mean_treated": float(w_mean_1),
                "weighted_mean_control": float(w_mean_0),
                "weighted_smd": float(w_smd),
                "abs_weighted_smd": float(np.abs(w_smd)),
                "balanced": bool(np.abs(w_smd) < 0.10),
            }
        )
    return pd.DataFrame(weighted_smds)


def check_overlap_diagnostics(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    threshold_low: float = 0.05,
    threshold_high: float = 0.95,
) -> Dict[str, float]:
    """
    Diagnose positivity violations and propensity overlap.
    """
    t1_ps = propensity_scores[treatment == 1]
    t0_ps = propensity_scores[treatment == 0]

    viol_low = np.mean(propensity_scores < threshold_low)
    viol_high = np.mean(propensity_scores > threshold_high)
    total_violations = np.mean((propensity_scores < threshold_low) | (propensity_scores > threshold_high))

    # Overlap range
    min_treated = np.min(t1_ps)
    max_control = np.max(t0_ps)
    overlap_min = max(np.min(t1_ps), np.min(t0_ps))
    overlap_max = min(np.max(t1_ps), np.max(t0_ps))

    # Effective Sample Size (ESS)
    weights = np.where(treatment == 1, 1.0 / np.clip(propensity_scores, 0.01, 0.99), 1.0 / np.clip(1.0 - propensity_scores, 0.01, 0.99))
    ess_t1 = float((np.sum(weights[treatment == 1]) ** 2) / np.sum(weights[treatment == 1] ** 2))
    ess_t0 = float((np.sum(weights[treatment == 0]) ** 2) / np.sum(weights[treatment == 0] ** 2))

    return {
        "positivity_violation_rate": float(total_violations),
        "low_support_rate": float(viol_low),
        "high_support_rate": float(viol_high),
        "common_support_range": [float(overlap_min), float(overlap_max)],
        "ess_treated": ess_t1,
        "ess_control": ess_t0,
        "nominal_n_treated": int(np.sum(treatment == 1)),
        "nominal_n_control": int(np.sum(treatment == 0)),
    }
