"""
Run Sensitivity Analyses for Unobserved Confounding.
Computes Rosenbaum's Gamma Bounds, Cinelli & Hazlett Robustness Values, and Oster's Delta.
Saves sensitivity figures to reports/figures/.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_yaml, setup_logger, ensure_dir
from src.data.dataset import CausalDataset
from src.sensitivity import rosenbaum_bounds, cinelli_hazlett_sensitivity, compute_oster_bounds
from src.visualization import plot_rosenbaum_sensitivity
from sklearn.linear_model import LinearRegression

logger = setup_logger("run_sensitivity")


def main():
    logger.info("Initializing Causal Sensitivity Analysis...")
    test_path = "data/processed/test_cohort.csv"
    if not os.path.exists(test_path):
        from scripts.prepare_data import main as prep_main
        prep_main()

    df_test = pd.read_csv(test_path)
    dataset = CausalDataset(df_test)
    ps = dataset.estimate_propensities()

    t1_indices = np.where(dataset.treatment == 1)[0]
    t0_indices = np.where(dataset.treatment == 0)[0]

    # Nearest neighbor matching on propensity score to obtain matched pair differences
    ps_t1 = ps[t1_indices]
    ps_t0 = ps[t0_indices]
    y_t1 = dataset.outcome[t1_indices]
    y_t0 = dataset.outcome[t0_indices]

    matched_diffs = []
    for i, p_val in enumerate(ps_t1):
        # Find nearest control
        nearest_idx = np.argmin(np.abs(ps_t0 - p_val))
        matched_diffs.append(y_t1[i] - y_t0[nearest_idx])

    matched_diffs = np.array(matched_diffs)
    gamma_range = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 3.0]

    # 1. Rosenbaum Bounds
    logger.info("Computing Rosenbaum sensitivity bounds across Gamma range...")
    rosen_df = rosenbaum_bounds(matched_diffs, gamma_range=gamma_range)

    res_dir = ensure_dir("experiments/results/sensitivity")
    fig_dir = ensure_dir("reports/figures")

    rosen_path = os.path.join(res_dir, "rosenbaum_bounds.csv")
    rosen_df.to_csv(rosen_path, index=False)
    logger.info(f"Saved Rosenbaum bounds to {rosen_path}")

    plot_rosenbaum_sensitivity(
        rosenbaum_df=rosen_df,
        save_path=os.path.join(fig_dir, "rosenbaum_sensitivity_plot.png"),
    )

    # 2. Cinelli & Hazlett Robustness Values
    logger.info("Computing Cinelli & Hazlett (2020) Robustness Values...")
    # Estimate linear regression with covariates to get SE and DOF
    X_reg = np.column_stack([dataset.treatment, dataset.X])
    lr = LinearRegression().fit(X_reg, dataset.outcome)
    y_pred = lr.predict(X_reg)
    residuals = dataset.outcome - y_pred
    dof = len(dataset) - X_reg.shape[1] - 1
    mse = np.sum(residuals**2) / dof
    # Standard error of treatment coefficient (first column)
    xtx_inv = np.linalg.pinv(np.dot(X_reg.T, X_reg))
    se_treat = np.sqrt(mse * xtx_inv[0, 0])
    tau_hat = lr.coef_[0]

    ch_res = cinelli_hazlett_sensitivity(
        tau_estimate=tau_hat,
        standard_error=se_treat,
        dof=dof,
        q=1.0,
        alpha=0.05,
    )

    ch_df = pd.DataFrame([ch_res])
    ch_df.to_csv(os.path.join(res_dir, "cinelli_hazlett_rv.csv"), index=False)
    logger.info(
        f"Cinelli-Hazlett RV_q: {ch_res['rv_q_percent']:.2f}% | "
        f"RV_alpha: {ch_res['rv_alpha_percent']:.2f}%"
    )

    # 3. Oster's Delta Bounds
    logger.info("Computing Oster's Delta selection bound...")
    # Uncontrolled model: outcome on treatment only
    lr_uncontrolled = LinearRegression().fit(dataset.treatment.reshape(-1, 1), dataset.outcome)
    r2_uncontrolled = float(lr_uncontrolled.score(dataset.treatment.reshape(-1, 1), dataset.outcome))
    beta_uncontrolled = float(lr_uncontrolled.coef_[0])

    r2_controlled = float(lr.score(X_reg, dataset.outcome))
    beta_controlled = float(tau_hat)

    oster_res = compute_oster_bounds(
        beta_uncontrolled=beta_uncontrolled,
        beta_controlled=beta_controlled,
        r2_uncontrolled=r2_uncontrolled,
        r2_controlled=r2_controlled,
        r_max=1.3,
    )
    oster_df = pd.DataFrame([oster_res])
    oster_df.to_csv(os.path.join(res_dir, "oster_bounds.csv"), index=False)
    logger.info(f"Oster's Delta: {oster_res['oster_delta']:.3f} (Robust: {oster_res['is_robust']})")

    logger.info("Sensitivity analyses completed successfully.")


if __name__ == "__main__":
    main()
