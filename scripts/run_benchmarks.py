"""
Run complete causal machine learning benchmarks.
Trains S-Learner, T-Learner, X-Learner, Causal Forest, and Doubly Robust Learner.
Computes PEHE, ATE, Policy Value, AUUC, and Confidence Intervals.
Saves checkpoints and figures.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_yaml, setup_logger, ensure_dir
from src.data.dataset import CausalDataset
from src.estimators.s_learner import SLearner
from src.estimators.t_learner import TLearner
from src.estimators.x_learner import XLearner
from src.estimators.causal_forest import CausalForest
from src.estimators.doubly_robust import DoublyRobustLearner
from src.evaluation.metrics import evaluate_estimator_full, compute_qini_curve
from src.evaluation.confidence_intervals import compute_bootstrap_intervals, compute_ci_coverage
from src.visualization import (
    plot_cate_distribution,
    plot_qini_curves,
    plot_benchmark_metrics,
    plot_subgroup_forest,
)

logger = setup_logger("run_benchmarks")


def main():
    logger.info("Initializing Causal ML Benchmark...")
    train_path = "data/processed/train_cohort.csv"
    test_path = "data/processed/test_cohort.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.info("Processed data not found. Running prepare_data.py first...")
        from scripts.prepare_data import main as prep_main
        prep_main()

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    train_ds = CausalDataset(df_train)
    test_ds = CausalDataset(df_test)

    X_train, t_train, y_train, _ = train_ds.get_arrays()
    X_test, t_test, y_test, tau_true = test_ds.get_arrays()
    y0_true = df_test["y0_true"].values
    y1_true = df_test["y1_true"].values

    estimators = {
        "S-Learner": SLearner(random_state=42),
        "T-Learner": TLearner(random_state=42),
        "X-Learner": XLearner(random_state=42),
        "Causal Forest": CausalForest(n_estimators=100, max_depth=5, min_samples_leaf=12, random_state=42),
        "Doubly Robust": DoublyRobustLearner(random_state=42),
    }

    results = []
    predictions = {}
    qini_curves = {}
    checkpoint_dir = ensure_dir("experiments/checkpoints")
    results_dir = ensure_dir("experiments/results")
    fig_dir = ensure_dir("reports/figures")

    for name, est in estimators.items():
        logger.info(f"Training estimator: {name}...")
        est.fit(X_train, t_train, y_train)

        # Save checkpoint
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(est, os.path.join(checkpoint_dir, f"{safe_name}.joblib"))

        # Predict CATE
        tau_hat = est.predict_cate(X_test)
        predictions[name] = tau_hat

        # Evaluate performance
        eval_metrics = evaluate_estimator_full(
            estimator_name=name,
            tau_hat=tau_hat,
            tau_true=tau_true,
            y0_true=y0_true,
            y1_true=y1_true,
            treatment=t_test,
            outcome=y_test,
        )

        # Qini curve
        fractions, gains, auuc = compute_qini_curve(tau_hat, t_test, y_test)
        qini_curves[name] = (fractions, gains, auuc)

        # Confidence intervals (50 bootstraps for benchmark speed & stability)
        logger.info(f"Calculating confidence intervals for {name}...")
        ci_res = compute_bootstrap_intervals(
            estimator=est,
            X=X_train,
            treatment=t_train,
            outcome=y_train,
            X_eval=X_test,
            n_bootstraps=35,
            alpha=0.05,
            random_state=42,
        )
        cov_info = compute_ci_coverage(ci_res["cate_lower"], ci_res["cate_upper"], tau_true)

        eval_metrics["ci_coverage"] = cov_info["coverage_rate"]
        eval_metrics["ci_mean_width"] = cov_info["mean_width"]
        eval_metrics["ate_ci_lower"] = ci_res["ate_lower"]
        eval_metrics["ate_ci_upper"] = ci_res["ate_upper"]

        results.append(eval_metrics)
        logger.info(
            f"[{name}] PEHE: {eval_metrics['pehe']:.4f} | ATE Bias: {eval_metrics['ate_bias']:.4f} | "
            f"Policy Val: {eval_metrics['policy_value']:.4f} | CI Coverage: {eval_metrics['ci_coverage']:.2%}"
        )

    results_df = pd.DataFrame(results)
    benchmark_path = os.path.join(results_dir, "benchmark_summary.csv")
    results_df.to_csv(benchmark_path, index=False)
    logger.info(f"Saved benchmark summary table to {benchmark_path}")

    # Save predictions dataframe
    pred_df = pd.DataFrame({"patient_id": df_test["patient_id"], "true_cate": tau_true})
    for name, preds in predictions.items():
        pred_df[name] = preds
    pred_df.to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)

    # 1. Plot CATE Distribution
    plot_cate_distribution(
        predictions_dict=predictions,
        true_cate=tau_true,
        save_path=os.path.join(fig_dir, "cate_distribution_comparison.png"),
    )

    # 2. Plot Qini Curves
    plot_qini_curves(
        qini_dict=qini_curves,
        save_path=os.path.join(fig_dir, "uplift_qini_curves.png"),
    )

    # 3. Plot Benchmark Bar Chart
    plot_benchmark_metrics(
        benchmark_df=results_df,
        save_path=os.path.join(fig_dir, "benchmark_metrics_barchart.png"),
    )

    # 4. Generate Subgroup Analysis Forest Plot based on best estimator
    best_estimator_name = results_df.sort_values(by="pehe").iloc[0]["estimator"]
    best_tau = predictions[best_estimator_name]

    # Subgroups definition
    subgroups = [
        ("Overall Cohort", np.ones(len(df_test), dtype=bool)),
        ("Age >= 65", df_test["age"] >= 65),
        ("Age < 65", df_test["age"] < 65),
        ("HbA1c >= 8.5%", df_test["hba1c"] >= 8.5),
        ("HbA1c < 8.5%", df_test["hba1c"] < 8.5),
        ("Prior CVD: Yes", df_test["prior_cvd"] == 1),
        ("Prior CVD: No", df_test["prior_cvd"] == 0),
        ("eGFR < 60 (CKD)", df_test["egfr"] < 60),
        ("eGFR >= 60", df_test["egfr"] >= 60),
        ("Females", df_test["female"] == 1),
        ("Males", df_test["female"] == 0),
    ]

    subgroup_records = []
    for s_name, mask in subgroups:
        sub_tau = best_tau[mask]
        mean_eff = float(np.mean(sub_tau))
        se_eff = float(np.std(sub_tau, ddof=1) / np.sqrt(max(1, len(sub_tau))))
        subgroup_records.append(
            {
                "subgroup": s_name,
                "n_patients": int(np.sum(mask)),
                "mean_cate": mean_eff,
                "ci_lower": mean_eff - 1.96 * se_eff,
                "ci_upper": mean_eff + 1.96 * se_eff,
            }
        )

    subgroups_df = pd.DataFrame(subgroup_records)
    subgroups_df.to_csv(os.path.join(results_dir, "subgroup_analysis.csv"), index=False)
    plot_subgroup_forest(
        subgroups_df=subgroups_df,
        save_path=os.path.join(fig_dir, "subgroup_analysis_forest_plot.png"),
    )

    logger.info("Benchmark execution completed successfully.")


if __name__ == "__main__":
    main()
