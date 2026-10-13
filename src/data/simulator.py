"""
Clinical Causal Data Simulator.
Generates realistic cardiovascular/metabolic observational cohorts with:
- Clinically correlated covariates
- Known ground-truth Individualized Treatment Effects (CATE / ITE)
- Observational selection bias (confounding)
- Controllable unobserved confounding for sensitivity analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.special import expit


class ClinicalCausalSimulator:
    """Simulator generating synthetic clinical trials and observational cohorts with known CATE."""

    COVARIATE_COLUMNS = [
        "age",
        "female",
        "bmi",
        "systolic_bp",
        "heart_rate",
        "hba1c",
        "egfr",
        "ldl_cholesterol",
        "crp",
        "prior_cvd",
        "diabetes_duration_years",
        "smoker",
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        sim_cfg = self.config.get("simulation", {})
        self.n_samples = sim_cfg.get("n_samples", 5000)
        self.test_size = sim_cfg.get("test_size", 0.25)
        self.random_seed = sim_cfg.get("random_seed", 42)
        self.confounding_strength = sim_cfg.get("confounding_strength", 1.5)
        self.treatment_prevalence = sim_cfg.get("treatment_prevalence", 0.35)
        self.unobserved_confounding_strength = sim_cfg.get("unobserved_confounding_variance", 0.25)

    def generate_data(
        self,
        n_samples: int = None,
        confounding_strength: float = None,
        treatment_prevalence: float = None,
        unobserved_confounding_strength: float = None,
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic clinical cohort with known ground truth counterfactuals.
        """
        n = n_samples if n_samples is not None else self.n_samples
        cs = confounding_strength if confounding_strength is not None else self.confounding_strength
        prev = treatment_prevalence if treatment_prevalence is not None else self.treatment_prevalence
        u_strength = (
            unobserved_confounding_strength
            if unobserved_confounding_strength is not None
            else self.unobserved_confounding_strength
        )
        rng = np.random.default_rng(seed if seed is not None else self.random_seed)

        # 1. Base demographics and correlated covariates
        age = np.clip(rng.normal(loc=62.0, scale=10.5, size=n), 35.0, 88.0)
        female = rng.binomial(n=1, p=0.48, size=n)
        bmi = np.clip(rng.normal(loc=28.5, scale=5.2, size=n), 18.0, 48.0)

        # Correlated vitals & biomarkers
        systolic_bp = np.clip(
            rng.normal(loc=135.0, scale=14.0, size=n)
            + 0.15 * (age - 62.0)
            + 0.35 * (bmi - 28.5),
            90.0,
            205.0,
        )
        heart_rate = np.clip(rng.normal(loc=74.0, scale=11.0, size=n), 45.0, 120.0)

        hba1c = np.clip(
            rng.normal(loc=8.0, scale=1.3, size=n) + 0.05 * (bmi - 28.5),
            5.5,
            14.5,
        )

        egfr = np.clip(
            rng.normal(loc=70.0, scale=17.0, size=n)
            - 0.40 * (age - 62.0)
            - 1.10 * (hba1c - 8.0),
            15.0,
            135.0,
        )

        ldl = np.clip(rng.normal(loc=115.0, scale=28.0, size=n), 40.0, 240.0)

        log_crp = rng.normal(loc=0.8, scale=0.5, size=n)
        crp = np.clip(np.exp(log_crp), 0.2, 20.0)

        smoker = rng.binomial(n=1, p=0.22, size=n)
        diabetes_duration = np.clip(
            rng.gamma(shape=2.5, scale=3.0, size=n), 0.5, 32.0
        )

        # Prior CVD history with logistic risk
        cvd_logits = (
            -1.3
            + 0.035 * (age - 62.0)
            + 0.015 * (systolic_bp - 138.0)
            + 0.18 * (crp - 3.0)
            + 0.40 * smoker
        )
        prior_cvd = rng.binomial(n=1, p=expit(cvd_logits), size=n)

        # 2. Unobserved confounder (frailty / microvascular burden)
        u_unobserved = rng.normal(loc=0.0, scale=1.0, size=n)

        # 3. Propensity score / Treatment assignment
        # Target logit intercept calibrated to prevalence
        base_logit = np.log(prev / (1.0 - prev + 1e-8))
        ps_logits = (
            base_logit
            + cs
            * (
                0.030 * (age - 62.0)
                + 0.350 * (hba1c - 8.0)
                + 0.850 * prior_cvd
                - 0.025 * (egfr - 68.0)
                + 0.250 * smoker
            )
            + u_strength * u_unobserved
        )
        propensity_score = expit(ps_logits)
        # Ensure numerical safety
        propensity_score = np.clip(propensity_score, 0.005, 0.995)
        treatment = rng.binomial(n=1, p=propensity_score, size=n)

        # 4. Prognostic baseline outcome mu_0(X)
        mu_0 = (
            4.5
            + 0.025 * (age - 62.0)
            + 0.320 * (hba1c - 8.0)
            + 0.018 * (systolic_bp - 135.0)
            + 0.750 * prior_cvd
            - 0.015 * (egfr - 70.0)
            + 0.120 * crp
            + 0.50 * u_strength * u_unobserved
        )

        # 5. True Individualized Treatment Effect tau(X) [Heterogeneous effect function]
        tau = (
            2.2
            + 0.70 * np.maximum(0.0, hba1c - 7.5)
            + 1.35 * prior_cvd
            + 0.85 * (egfr < 60.0).astype(float)
            - 0.035 * (age - 62.0)
            - 0.025 * (bmi - 28.5)
            + 0.45 * female * prior_cvd
        )

        # Residual noise
        noise_0 = rng.normal(loc=0.0, scale=1.1, size=n)
        noise_1 = rng.normal(loc=0.0, scale=1.1, size=n)

        y0 = mu_0 + noise_0
        y1 = mu_0 + tau + noise_1

        # Observed factual outcome
        outcome = np.where(treatment == 1, y1, y0)

        patient_ids = [f"PT-{i+1:06d}" for i in range(n)]

        df = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "age": age,
                "female": female,
                "bmi": bmi,
                "systolic_bp": systolic_bp,
                "heart_rate": heart_rate,
                "hba1c": hba1c,
                "egfr": egfr,
                "ldl_cholesterol": ldl,
                "crp": crp,
                "prior_cvd": prior_cvd,
                "diabetes_duration_years": diabetes_duration,
                "smoker": smoker,
                "unobserved_u": u_unobserved,
                "treatment": treatment,
                "propensity_score": propensity_score,
                "true_cate": tau,
                "y0_true": y0,
                "y1_true": y1,
                "outcome": outcome,
            }
        )
        return df


def generate_and_save_data(
    config_path: str = "configs/dataset.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates train and test datasets based on config and saves to CSV."""
    from src.utils import load_yaml, ensure_dir

    cfg = load_yaml(config_path)
    sim = ClinicalCausalSimulator(config=cfg)

    n_total = cfg["simulation"]["n_samples"]
    test_frac = cfg["simulation"]["test_size"]
    seed = cfg["simulation"]["random_seed"]

    df_all = sim.generate_data(n_samples=n_total, seed=seed)

    # Train / test split by patient id index
    n_test = int(n_total * test_frac)
    n_train = n_total - n_test

    df_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
    df_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

    raw_dir = ensure_dir(cfg["paths"]["raw_dir"])
    proc_dir = ensure_dir(cfg["paths"]["processed_dir"])

    df_all.to_csv(f"{raw_dir}/clinical_cohort_full.csv", index=False)
    df_train.to_csv(cfg["paths"]["train_data"], index=False)
    df_test.to_csv(cfg["paths"]["test_data"], index=False)

    ground_truth = df_test[["patient_id", "true_cate", "y0_true", "y1_true", "treatment", "outcome"]]
    ground_truth.to_csv(cfg["paths"]["synthetic_ground_truth"], index=False)

    return df_train, df_test
