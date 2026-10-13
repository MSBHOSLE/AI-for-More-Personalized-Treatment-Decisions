"""
Unit tests for causal estimators (S-Learner, T-Learner, X-Learner, Causal Forest, Doubly Robust).
"""

import pytest
import numpy as np
from src.data.simulator import ClinicalCausalSimulator
from src.estimators.s_learner import SLearner
from src.estimators.t_learner import TLearner
from src.estimators.x_learner import XLearner
from src.estimators.causal_forest import CausalForest
from src.estimators.doubly_robust import DoublyRobustLearner


@pytest.fixture
def sample_data():
    sim = ClinicalCausalSimulator()
    df = sim.generate_data(n_samples=180, seed=42)
    cov_cols = ClinicalCausalSimulator.COVARIATE_COLUMNS
    X = df[cov_cols].values
    t = df["treatment"].values
    y = df["outcome"].values
    cate = df["true_cate"].values
    return X, t, y, cate


def test_s_learner(sample_data):
    X, t, y, _ = sample_data
    model = SLearner(random_state=42)
    model.fit(X, t, y)

    tau_hat = model.predict_cate(X)
    assert len(tau_hat) == len(X)
    assert not np.isnan(tau_hat).any()

    y0, y1 = model.predict_potential_outcomes(X)
    assert np.allclose(y1 - y0, tau_hat)

    ate = model.predict_ate(X)
    assert np.isclose(ate, np.mean(tau_hat))


def test_t_learner(sample_data):
    X, t, y, _ = sample_data
    model = TLearner(random_state=42)
    model.fit(X, t, y)

    tau_hat = model.predict_cate(X)
    assert len(tau_hat) == len(X)
    assert not np.isnan(tau_hat).any()

    y0, y1 = model.predict_potential_outcomes(X)
    assert np.allclose(y1 - y0, tau_hat)


def test_x_learner(sample_data):
    X, t, y, _ = sample_data
    model = XLearner(random_state=42)
    model.fit(X, t, y)

    tau_hat = model.predict_cate(X)
    assert len(tau_hat) == len(X)
    assert not np.isnan(tau_hat).any()

    y0, y1 = model.predict_potential_outcomes(X)
    assert len(y0) == len(X)
    assert len(y1) == len(X)


def test_causal_forest(sample_data):
    X, t, y, _ = sample_data
    model = CausalForest(n_estimators=15, max_depth=3, min_samples_leaf=5, random_state=42)
    model.fit(X, t, y)

    tau_hat = model.predict_cate(X)
    assert len(tau_hat) == len(X)
    assert not np.isnan(tau_hat).any()

    y0, y1 = model.predict_potential_outcomes(X)
    assert len(y0) == len(X)
    assert len(y1) == len(X)


def test_doubly_robust(sample_data):
    X, t, y, _ = sample_data
    model = DoublyRobustLearner(n_splits=3, random_state=42)
    model.fit(X, t, y)

    tau_hat = model.predict_cate(X)
    assert len(tau_hat) == len(X)
    assert not np.isnan(tau_hat).any()

    se = model.ate_standard_error()
    assert se > 0.0

    ate = model.predict_ate()
    assert not np.isnan(ate)
