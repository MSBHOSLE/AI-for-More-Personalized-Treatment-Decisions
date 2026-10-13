"""
Unit tests for evaluation metrics and confidence interval computation.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_pehe,
    compute_ate_metrics,
    compute_policy_value,
    compute_qini_curve,
    evaluate_estimator_full,
)
from src.evaluation.confidence_intervals import compute_ci_coverage


def test_metrics():
    tau_true = np.array([1.0, 2.0, 3.0, 4.0])
    tau_hat = np.array([1.1, 1.9, 3.2, 3.8])

    pehe = compute_pehe(tau_hat, tau_true)
    assert pehe > 0.0
    assert np.isclose(pehe, np.sqrt(np.mean((tau_hat - tau_true) ** 2)))

    ate_res = compute_ate_metrics(tau_hat, tau_true)
    assert np.isclose(ate_res["ate_hat"], 2.5)
    assert np.isclose(ate_res["ate_true"], 2.5)
    assert np.isclose(ate_res["ate_bias"], 0.0)

    y0 = np.array([2.0, 3.0, 1.0, 2.0])
    y1 = y0 + tau_true
    pol = compute_policy_value(tau_hat, y0, y1)
    assert pol["policy_value"] > pol["treat_none_value"]
    assert pol["policy_regret"] == 0.0  # Perfect policy decisions


def test_qini_curve():
    tau_hat = np.array([5.0, 4.0, 3.0, 1.0, 0.5])
    treatment = np.array([1, 1, 0, 1, 0])
    outcome = np.array([6.0, 5.0, 1.0, 2.0, 1.0])

    fractions, gains, auuc = compute_qini_curve(tau_hat, treatment, outcome, n_bins=3)
    assert len(fractions) > 1
    assert len(gains) == len(fractions)
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0


def test_ci_coverage():
    lower = np.array([1.0, 2.0, 1.5])
    upper = np.array([3.0, 4.0, 3.5])
    truth = np.array([2.0, 3.0, 2.5])

    res = compute_ci_coverage(lower, upper, truth)
    assert res["coverage_rate"] == 1.0
    assert res["mean_width"] == 2.0
