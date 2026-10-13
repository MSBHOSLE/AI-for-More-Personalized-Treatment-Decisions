"""
Sample Size Experiment.
Evaluates estimator sample efficiency across varying cohort sizes:
N in [500, 1000, 2000, 3500, 5000].
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.data.simulator import ClinicalCausalSimulator
from src.estimators.s_learner import SLearner
from src.estimators.t_learner import TLearner
from src.estimators.x_learner import XLearner
from src.estimators.causal_forest import CausalForest
from src.estimators.doubly_robust import DoublyRobustLearner
from src.evaluation.metrics import evaluate_estimator_full
from src.utils import ensure_dir, setup_logger

logger = setup_logger("sample_size_experiment")


def run_sample_size_experiment(
    sample_sizes: List[int] = None,
    n_replications: int = 3,
    test_size: int = 1500,
    output_dir: str = "experiments/results/sample_size",
) -> pd.DataFrame:
    """Run experiment evaluating PEHE and ATE bias as training sample size grows."""
    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 3500, 5000]

    ensure_dir(output_dir)
    sim = ClinicalCausalSimulator()

    # Fixed independent evaluation set
    test_df = sim.generate_data(n_samples=test_size, seed=999)
    cov_cols = ClinicalCausalSimulator.COVARIATE_COLUMNS
    X_test = test_df[cov_cols].values
    y0_test = test_df["y0_true"].values
    y1_test = test_df["y1_true"].values
    tau_test = test_df["true_cate"].values
    t_test = test_df["treatment"].values
    y_test = test_df["outcome"].values

    records = []

    for n in sample_sizes:
        logger.info(f"Evaluating sample size: N = {n} ({n_replications} replications)...")
        for rep in range(n_replications):
            seed = 1000 * rep + n
            train_df = sim.generate_data(n_samples=n, seed=seed)

            X_train = train_df[cov_cols].values
            t_train = train_df["treatment"].values
            y_train = train_df["outcome"].values

            estimators = {
                "S-Learner": SLearner(random_state=seed),
                "T-Learner": TLearner(random_state=seed),
                "X-Learner": XLearner(random_state=seed),
                "Causal Forest": CausalForest(n_estimators=60, random_state=seed),
                "Doubly Robust": DoublyRobustLearner(random_state=seed),
            }

            for name, est in estimators.items():
                try:
                    est.fit(X_train, t_train, y_train)
                    tau_hat = est.predict_cate(X_test)
                    eval_metrics = evaluate_estimator_full(
                        estimator_name=name,
                        tau_hat=tau_hat,
                        tau_true=tau_test,
                        y0_true=y0_test,
                        y1_true=y1_test,
                        treatment=t_test,
                        outcome=y_test,
                    )
                    eval_metrics["sample_size"] = n
                    eval_metrics["replication"] = rep
                    records.append(eval_metrics)
                except Exception as e:
                    logger.warning(f"Failed {name} at N={n}, rep={rep}: {e}")

    results_df = pd.DataFrame(records)
    results_path = os.path.join(output_dir, "sample_size_results.csv")
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["sample_size", "estimator"])[["pehe", "ate_abs_error", "policy_value", "auuc"]]
        .mean()
        .reset_index()
    )
    summary_path = os.path.join(output_dir, "sample_size_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved sample size experiment results to {results_path}")
    return results_df
