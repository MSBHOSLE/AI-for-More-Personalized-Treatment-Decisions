"""
Treatment Imbalance Experiment.
Evaluates estimator stability across treatment allocation ratios:
prevalence in [0.05, 0.10, 0.20, 0.35, 0.50].
Demonstrates X-Learner and Doubly Robust superiority when treated units are sparse.
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

logger = setup_logger("treatment_imbalance_experiment")


def run_treatment_imbalance_experiment(
    treatment_prevalences: List[float] = None,
    n_samples: int = 3500,
    n_replications: int = 3,
    output_dir: str = "experiments/results/treatment_imbalance",
) -> pd.DataFrame:
    """Run experiment evaluating estimator performance under varying treatment proportions."""
    if treatment_prevalences is None:
        treatment_prevalences = [0.05, 0.10, 0.20, 0.35, 0.50]

    ensure_dir(output_dir)
    sim = ClinicalCausalSimulator()
    cov_cols = ClinicalCausalSimulator.COVARIATE_COLUMNS
    records = []

    # Balanced independent test set
    test_df = sim.generate_data(n_samples=1500, treatment_prevalence=0.35, seed=777)
    X_test = test_df[cov_cols].values
    y0_test = test_df["y0_true"].values
    y1_test = test_df["y1_true"].values
    tau_test = test_df["true_cate"].values
    t_test = test_df["treatment"].values
    y_test = test_df["outcome"].values

    for prev in treatment_prevalences:
        logger.info(f"Evaluating treatment prevalence: p = {prev:.2f}...")
        for rep in range(n_replications):
            seed = 3000 * rep + int(prev * 100)
            train_df = sim.generate_data(
                n_samples=n_samples,
                treatment_prevalence=prev,
                seed=seed,
            )

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
                    eval_metrics["treatment_prevalence"] = prev
                    eval_metrics["replication"] = rep
                    eval_metrics["actual_treated_count"] = int(np.sum(t_train == 1))
                    records.append(eval_metrics)
                except Exception as e:
                    logger.warning(f"Failed {name} at prev={prev}, rep={rep}: {e}")

    results_df = pd.DataFrame(records)
    results_path = os.path.join(output_dir, "treatment_imbalance_results.csv")
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["treatment_prevalence", "estimator"])[
            ["pehe", "ate_abs_error", "policy_value", "auuc"]
        ]
        .mean()
        .reset_index()
    )
    summary_path = os.path.join(output_dir, "treatment_imbalance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved treatment imbalance experiment results to {results_path}")
    return results_df
