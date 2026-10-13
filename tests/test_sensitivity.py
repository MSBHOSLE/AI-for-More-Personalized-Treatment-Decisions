"""
Unit tests for causal sensitivity analysis methods.
"""

import pytest
import numpy as np
from src.sensitivity import rosenbaum_bounds, cinelli_hazlett_sensitivity, compute_oster_bounds


def test_rosenbaum_bounds():
    # Differences where treatment has strong positive effect
    rng = np.random.default_rng(42)
    diffs = rng.normal(loc=1.8, scale=0.6, size=50)

    res_df = rosenbaum_bounds(diffs, gamma_range=[1.0, 1.5, 2.0])
    assert len(res_df) == 3
    assert "p_value_upper" in res_df.columns
    # Gamma=1 should have very small p-value
    assert res_df.loc[res_df["gamma"] == 1.0, "p_value_upper"].values[0] < 0.01


def test_cinelli_hazlett_sensitivity():
    res = cinelli_hazlett_sensitivity(
        tau_estimate=2.5,
        standard_error=0.4,
        dof=400,
        q=1.0,
        alpha=0.05,
    )
    assert res["t_statistic"] > 5.0
    assert res["robustness_value_rv_q"] > 0.0
    assert res["robustness_value_rv_alpha"] > 0.0
    assert res["rv_q_percent"] <= 100.0


def test_oster_bounds():
    res = compute_oster_bounds(
        beta_uncontrolled=3.5,
        beta_controlled=2.2,
        r2_uncontrolled=0.15,
        r2_controlled=0.45,
        r_max=1.3,
    )
    assert "oster_delta" in res
    assert "is_robust" in res
