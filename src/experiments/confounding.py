"""
Confounding Robustness Experiment.
Evaluates estimator sensitivity to unobserved confounding strength:
gamma in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0].
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

logger = setup_logger("confounding_experiment")


def run_confounding_experiment(
    confounding_strengths: List[float] = None,
    n_samples: int = 3000,
    n_replications: int = 3,
    output_dir: str = "experiments/results/confounding",
) -> pd.DataFrame:
    """Run experiment evaluating estimator degradation under unobserved confounding."""
    if confounding_strengths is None:
        confounding_strengths = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

    ensure_dir(output_dir)
    sim = ClinicalCausalSimulator()
    cov_cols = ClinicalCausalSimulator.COVARIATE_COLUMNS
    records = []

    for u_strength in confounding_strengths:
        logger.info(f"Evaluating unobserved confounding strength: gamma = {u_strength}...")
        for rep in range(n_replications):
            seed = 2000 * rep + int(u_strength * 100)
            train_df = sim.generate_data(
                n_samples=n_samples,
                unobserved_confounding_strength=u_strength,
                seed=seed,
            )
            test_df = sim.generate_data(
                n_samples=1500,
                unobserved_confounding_strength=u_strength,
                seed=seed + 500,
            )

            X_train = train_df[cov_cols].values
            t_train = train_df["treatment"].values
            y_train = train_df["outcome"].values

            X_test = test_df[cov_cols].values
            y0_test = test_df["y0_true"].values
            y1_test = test_df["y1_true"].values
            tau_test = test_df["true_cate"].values
            t_test = test_df["treatment"].values
            y_test = test_df["outcome"].values

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
                    eval_metrics["unobserved_confounding_strength"] = u_strength
                    eval_metrics["replication"] = rep
                    records.append(eval_metrics)
                except Exception as e:
                    logger.warning(f"Failed {name} at gamma={u_strength}, rep={rep}: {e}")

    results_df = pd.DataFrame(records)
    results_path = os.path.join(output_dir, "confounding_results.csv")
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["unobserved_confounding_strength", "estimator"])[
            ["pehe", "ate_abs_error", "ate_bias", "policy_regret"]
        ]
        .mean()
        .reset_index()
    )
    summary_path = os.path.join(output_dir, "confounding_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved confounding experiment results to {results_path}")
    return results_df
