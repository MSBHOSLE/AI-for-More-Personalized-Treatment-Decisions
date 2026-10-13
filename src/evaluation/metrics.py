"""
Causal Inference Evaluation Metrics.
Includes:
- PEHE (Precision in Estimation of Heterogeneous Effect)
- ATE Bias & RMSE
- Policy Value & Regret
- Qini Curve & AUUC (Area Under Uplift Curve)
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect (PEHE).
    PEHE = sqrt(mean((tau_hat - tau_true)^2))
    """
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def compute_ate_metrics(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
) -> Dict[str, float]:
    """Compute ATE Bias, Absolute Error, and RMSE."""
    ate_hat = np.mean(tau_hat)
    ate_true = np.mean(tau_true)
    bias = float(ate_hat - ate_true)
    abs_err = float(np.abs(bias))
    return {
        "ate_hat": float(ate_hat),
        "ate_true": float(ate_true),
        "ate_bias": bias,
        "ate_abs_error": abs_err,
    }


def compute_policy_value(
    tau_hat: np.ndarray,
    y0_true: np.ndarray,
    y1_true: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate personalized treatment policy pi(X) = 1 if tau_hat(X) > threshold else 0.
    Returns:
      policy_value: Expected outcome under assigned policy.
      treat_all_value: Expected outcome if treating all patients.
      treat_none_value: Expected outcome if treating no patients.
      optimal_value: Expected outcome under oracle assignment (tau_true > threshold).
      policy_regret: optimal_value - policy_value (0 is perfect).
    """
    # Policy decisions
    pi_hat = (tau_hat > threshold).astype(int)
    tau_true = y1_true - y0_true
    pi_oracle = (tau_true > threshold).astype(int)

    val_policy = np.mean(pi_hat * y1_true + (1 - pi_hat) * y0_true)
    val_all = np.mean(y1_true)
    val_none = np.mean(y0_true)
    val_oracle = np.mean(pi_oracle * y1_true + (1 - pi_oracle) * y0_true)

    regret = max(0.0, float(val_oracle - val_policy))

    return {
        "policy_value": float(val_policy),
        "treat_all_value": float(val_all),
        "treat_none_value": float(val_none),
        "optimal_value": float(val_oracle),
        "policy_regret": regret,
        "treated_fraction": float(np.mean(pi_hat)),
    }


def compute_qini_curve(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Cumulative Uplift / Qini Curve and AUUC.
    Patients are sorted by predicted CATE in descending order.
    Returns:
      fractions: array of population fractions (0 to 1)
      cumulative_gains: cumulative difference in outcome between treated and control
      qini_auc: Area under Qini curve
    """
    n = len(tau_hat)
    order = np.argsort(-tau_hat)  # Descending order of predicted benefit

    sorted_t = treatment[order]
    sorted_y = outcome[order]

    cum_n_t = np.cumsum(sorted_t)
    cum_n_c = np.cumsum(1 - sorted_t)

    cum_y_t = np.cumsum(sorted_t * sorted_y)
    cum_y_c = np.cumsum((1 - sorted_t) * sorted_y)

    total_n_t = max(1, np.sum(treatment == 1))
    total_n_c = max(1, np.sum(treatment == 0))

    # Qini cumulative gain: cum_y_t - cum_y_c * (cum_n_t / max(1, cum_n_c))
    gain = np.zeros(n)
    for i in range(n):
        if cum_n_c[i] > 0:
            gain[i] = cum_y_t[i] - cum_y_c[i] * (cum_n_t[i] / cum_n_c[i])
        else:
            gain[i] = cum_y_t[i]

    # Sample bins for curve
    step = max(1, n // n_bins)
    sampled_indices = list(range(0, n, step))
    if (n - 1) not in sampled_indices:
        sampled_indices.append(n - 1)

    fractions = (np.array(sampled_indices) + 1) / n
    cumulative_gains = gain[sampled_indices]

    # Normalize fractions from 0
    fractions = np.insert(fractions, 0, 0.0)
    cumulative_gains = np.insert(cumulative_gains, 0, 0.0)

    # AUUC (numerical trapezoidal integration, compatible with NumPy 2.0+)
    if hasattr(np, "trapezoid"):
        auuc = float(np.trapezoid(cumulative_gains, fractions))
    else:
        auuc = float(np.trapz(cumulative_gains, fractions))

    return fractions, cumulative_gains, auuc


def evaluate_estimator_full(
    estimator_name: str,
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    y0_true: np.ndarray,
    y1_true: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> Dict[str, float]:
    """Run comprehensive causal evaluation suite for a single estimator."""
    pehe = compute_pehe(tau_hat, tau_true)
    ate_res = compute_ate_metrics(tau_hat, tau_true)
    pol_res = compute_policy_value(tau_hat, y0_true, y1_true)
    _, _, auuc = compute_qini_curve(tau_hat, treatment, outcome)

    # Correlation between predicted and true treatment effect
    corr = float(np.corrcoef(tau_hat, tau_true)[0, 1]) if np.std(tau_hat) > 1e-8 else 0.0

    return {
        "estimator": estimator_name,
        "pehe": pehe,
        "ate_hat": ate_res["ate_hat"],
        "ate_true": ate_res["ate_true"],
        "ate_bias": ate_res["ate_bias"],
        "ate_abs_error": ate_res["ate_abs_error"],
        "policy_value": pol_res["policy_value"],
        "policy_regret": pol_res["policy_regret"],
        "effect_correlation": corr,
        "auuc": auuc,
        "treated_fraction": pol_res["treated_fraction"],
    }
