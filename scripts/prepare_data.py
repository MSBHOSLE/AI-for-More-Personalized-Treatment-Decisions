"""
Prepare clinical dataset for causal ML benchmark.
Generates cohorts, computes baseline diagnostics and save processed files.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_yaml, setup_logger, ensure_dir
from src.data.simulator import generate_and_save_data
from src.data.dataset import CausalDataset
from src.evaluation.diagnostics import compute_smd_weighted, check_overlap_diagnostics
from src.visualization import plot_love_plot

logger = setup_logger("prepare_data")


def main():
    config_path = "configs/dataset.yaml"
    logger.info(f"Loading configuration from {config_path}...")
    cfg = load_yaml(config_path)

    logger.info("Simulating clinical cohort data with ground truth counterfactuals...")
    df_train, df_test = generate_and_save_data(config_path)

    logger.info(f"Generated train cohort: {len(df_train)} rows, test cohort: {len(df_test)} rows.")

    # Wrap train cohort in CausalDataset for diagnostics
    train_dataset = CausalDataset(df=df_train)
    ps = train_dataset.estimate_propensities()
    overlap_info = check_overlap_diagnostics(ps, train_dataset.treatment)
    logger.info(f"Propensity overlap diagnostics: {overlap_info}")

    # Compute covariate balance (Love plot data)
    smd_df = compute_smd_weighted(
        X=train_dataset.X,
        treatment=train_dataset.treatment,
        propensity_scores=ps,
        feature_names=train_dataset.feature_names,
    )

    proc_dir = ensure_dir("data/processed")
    fig_dir = ensure_dir("reports/figures")

    smd_df.to_csv(os.path.join(proc_dir, "covariate_balance_smd.csv"), index=False)
    plot_love_plot(smd_df, save_path=os.path.join(fig_dir, "covariate_balance_love_plot.png"))
    logger.info(f"Covariate balance plot saved to {fig_dir}/covariate_balance_love_plot.png")

    logger.info("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
