"""
Causal Sensitivity Analysis for Unobserved Confounding.
Implements:
1. Rosenbaum's Sensitivity Bounds (Gamma) for hidden bias in observational studies.
2. Cinelli & Hazlett (2020) Robustness Value (RV) and Partial R^2 Sensitivity Bounds.
3. Oster's (2019) coefficient stability bounds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import norm


def rosenbaum_bounds(
    y_diffs: np.ndarray,
    gamma_range: List[float] = None,
) -> pd.DataFrame:
    """
    Rosenbaum sensitivity bounds for signed-rank test on matched pair differences or treatment contrasts.
    Args:
        y_diffs: Array of outcome differences between treated and control units (e.g. matched or residual pairs).
        gamma_range: List of Gamma odds ratios (hidden bias factors).
    Returns:
        DataFrame containing p-value upper and lower bounds for each Gamma.
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

    # Filter out exact zeros
    diffs = y_diffs[y_diffs != 0]
    n = len(diffs)
    if n == 0:
        raise ValueError("Cannot calculate Rosenbaum bounds with empty differences.")

    abs_diffs = np.abs(diffs)
    ranks = np.argsort(np.argsort(abs_diffs)) + 1
    # Signed rank test statistic: sum of ranks of positive differences
    pos_mask = diffs > 0
    t_stat = np.sum(ranks[pos_mask])

    results = []
    for gamma in gamma_range:
        # Under null hypothesis of no treatment effect:
        # Each rank has probability p+ = gamma / (1 + gamma) and p- = 1 / (1 + gamma)
        p_plus = gamma / (1.0 + gamma)
        p_minus = 1.0 / (1.0 + gamma)

        # Expected value and variance under worst-case upper bound
        e_stat_upper = p_plus * np.sum(ranks)
        var_stat = (gamma / ((1.0 + gamma) ** 2)) * np.sum(ranks**2)
        std_stat = np.sqrt(max(var_stat, 1e-8))

        # Upper bound on z-score (testing whether positive effect is spurious)
        z_upper = (t_stat - e_stat_upper - 0.5) / std_stat
        p_val_upper = 1.0 - norm.cdf(z_upper)

        # Lower bound
        e_stat_lower = p_minus * np.sum(ranks)
        z_lower = (t_stat - e_stat_lower - 0.5) / std_stat
        p_val_lower = 1.0 - norm.cdf(z_lower)

        results.append(
            {
                "gamma": float(gamma),
                "z_upper": float(z_upper),
                "p_value_upper": float(np.clip(p_val_upper, 0.0, 1.0)),
                "p_value_lower": float(np.clip(p_val_lower, 0.0, 1.0)),
                "remains_significant_at_05": bool(p_val_upper < 0.05),
            }
        )

    return pd.DataFrame(results)


def cinelli_hazlett_sensitivity(
    tau_estimate: float,
    standard_error: float,
    dof: int,
    q: float = 1.0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Cinelli & Hazlett (2020) Robustness Value (RV) computation.
    Quantifies the minimum association an unobserved confounder must have with
    both treatment and outcome to bring the treatment effect to zero (q=1.0) or below statistical significance.
    Args:
        tau_estimate: Estimated treatment effect (ATE).
        standard_error: Estimated standard error of the effect.
        dof: Degrees of freedom (n - k - 1).
        q: Percent of effect to explain away (1.0 = 100%, bringing effect to 0).
        alpha: Significance level (0.05 for 95% confidence).
    """
    t_stat = np.abs(tau_estimate) / (standard_error + 1e-8)
    f_stat = t_stat**2

    # Partial R^2 of treatment with outcome
    r2_y_d = f_stat / (f_stat + dof)

    # Robustness Value (RV_q): point estimate reduction
    # RV_q = 0.5 * (sqrt(f^2 * q^2 + 4 * f * q) - f * q) / ...
    f_q = (q * t_stat) ** 2 / dof
    rv_q = (np.sqrt(f_q**2 + 4 * f_q) - f_q) / 2.0

    # Critical t for significance
    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(1.0 - alpha / 2.0, df=dof)

    # Robustness Value for statistical significance (RV_alpha)
    if t_stat > t_crit:
        f_alpha = ((t_stat - t_crit) ** 2) / dof
        rv_alpha = (np.sqrt(f_alpha**2 + 4 * f_alpha) - f_alpha) / 2.0
    else:
        rv_alpha = 0.0

    return {
        "tau_estimate": float(tau_estimate),
        "standard_error": float(standard_error),
        "t_statistic": float(t_stat),
        "degrees_of_freedom": int(dof),
        "partial_r2_treatment": float(r2_y_d),
        "robustness_value_rv_q": float(np.clip(rv_q, 0.0, 1.0)),
        "robustness_value_rv_alpha": float(np.clip(rv_alpha, 0.0, 1.0)),
        "rv_q_percent": float(np.clip(rv_q * 100.0, 0.0, 100.0)),
        "rv_alpha_percent": float(np.clip(rv_alpha * 100.0, 0.0, 100.0)),
    }


def compute_oster_bounds(
    beta_uncontrolled: float,
    beta_controlled: float,
    r2_uncontrolled: float,
    r2_controlled: float,
    r_max: float = 1.3,
) -> Dict[str, float]:
    """
    Oster (2019) coefficient stability and selection on unobservables bound (delta).
    Calculates the degree of proportional selection on unobservables (delta) required
    to drive the treatment effect to zero.
    """
    r_max_val = min(1.0, r2_controlled * r_max)
    delta_numerator = (beta_controlled * (r_max_val - r2_controlled))
    delta_denominator = (beta_uncontrolled - beta_controlled) * (r2_controlled - r2_uncontrolled)

    if np.abs(delta_denominator) < 1e-8:
        delta = 999.0
    else:
        delta = delta_numerator / delta_denominator

    return {
        "beta_uncontrolled": float(beta_uncontrolled),
        "beta_controlled": float(beta_controlled),
        "r2_uncontrolled": float(r2_uncontrolled),
        "r2_controlled": float(r2_controlled),
        "r_max": float(r_max_val),
        "oster_delta": float(delta),
        "is_robust": bool(np.abs(delta) > 1.0),
    }
