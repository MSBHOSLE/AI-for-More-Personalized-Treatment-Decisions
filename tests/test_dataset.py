"""
Unit tests for data generation and dataset processing.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.simulator import ClinicalCausalSimulator
from src.data.dataset import CausalDataset
from src.evaluation.diagnostics import compute_smd_unweighted, compute_smd_weighted, check_overlap_diagnostics


def test_simulator_data_generation():
    sim = ClinicalCausalSimulator()
    df = sim.generate_data(n_samples=200, seed=123)

    assert len(df) == 200
    assert "patient_id" in df.columns
    assert "treatment" in df.columns
    assert "outcome" in df.columns
    assert "true_cate" in df.columns
    assert set(df["treatment"].unique()).issubset({0, 1})

    # Age bounds
    assert df["age"].min() >= 35.0
    assert df["age"].max() <= 88.0

    # Finite outcomes
    assert not df["outcome"].isna().any()
    assert not df["true_cate"].isna().any()


def test_causal_dataset_wrapper():
    sim = ClinicalCausalSimulator()
    df = sim.generate_data(n_samples=250, seed=456)

    dataset = CausalDataset(df)
    assert len(dataset) == 250
    assert len(dataset.feature_names) == 12

    X, t, y, cate = dataset.get_arrays()
    assert X.shape == (250, 12)
    assert len(t) == 250
    assert len(y) == 250
    assert len(cate) == 250

    # Propensity estimation
    ps = dataset.estimate_propensities()
    assert len(ps) == 250
    assert (ps >= 0.01).all() and (ps <= 0.99).all()

    # Diagnostics
    smd_df = compute_smd_weighted(X, t, ps, dataset.feature_names)
    assert len(smd_df) == 12
    assert "weighted_smd" in smd_df.columns

    overlap = check_overlap_diagnostics(ps, t)
    assert "positivity_violation_rate" in overlap
    assert overlap["nominal_n_treated"] > 0
