"""
Dataset abstraction and preprocessing for Causal Medical AI.
Manages covariate extraction, propensity estimation, and common support trimming.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class CausalDataset:
    """Encapsulates observational clinical cohort data for causal analysis."""

    DEFAULT_COVARIATES = [
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

    def __init__(
        self,
        df: pd.DataFrame,
        covariate_cols: Optional[List[str]] = None,
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        cate_col: Optional[str] = "true_cate",
        scale_features: bool = False,
    ):
        self.df = df.copy()
        self.covariate_cols = covariate_cols or [
            col for col in self.DEFAULT_COVARIATES if col in self.df.columns
        ]
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.cate_col = cate_col

        self.X = self.df[self.covariate_cols].values.astype(np.float64)
        self.treatment = self.df[self.treatment_col].values.astype(np.int32)
        self.outcome = self.df[self.outcome_col].values.astype(np.float64)

        if self.cate_col and self.cate_col in self.df.columns:
            self.true_cate = self.df[self.cate_col].values.astype(np.float64)
        else:
            self.true_cate = None

        self.scaler = None
        if scale_features:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)

        self._propensity_scores = None

    def __len__(self) -> int:
        return len(self.df)

    @property
    def feature_names(self) -> List[str]:
        return list(self.covariate_cols)

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return (X, treatment, outcome, true_cate)."""
        return self.X, self.treatment, self.outcome, self.true_cate

    def estimate_propensities(
        self,
        clip_bounds: Tuple[float, float] = (0.01, 0.99),
        penalty: str = "l2",
        c_param: float = 1.0,
    ) -> np.ndarray:
        """Estimate propensity scores e(X) using logistic regression."""
        clf = LogisticRegression(C=c_param, max_iter=1000, random_state=42)
        clf.fit(self.X, self.treatment)
        probs = clf.predict_proba(self.X)[:, 1]
        probs = np.clip(probs, clip_bounds[0], clip_bounds[1])
        self._propensity_scores = probs
        return probs

    def trim_common_support(
        self,
        lower_quantile: float = 0.02,
        upper_quantile: float = 0.98,
    ) -> "CausalDataset":
        """
        Enforce strict positivity by trimming observations outside the common support
        of treatment and control propensity scores.
        """
        if self._propensity_scores is None:
            self.estimate_propensities()

        treated_ps = self._propensity_scores[self.treatment == 1]
        control_ps = self._propensity_scores[self.treatment == 0]

        min_support = max(np.min(treated_ps), np.quantile(control_ps, lower_quantile))
        max_support = min(np.max(control_ps), np.quantile(treated_ps, upper_quantile))

        mask = (self._propensity_scores >= min_support) & (self._propensity_scores <= max_support)
        trimmed_df = self.df[mask].reset_index(drop=True)

        trimmed_dataset = CausalDataset(
            df=trimmed_df,
            covariate_cols=self.covariate_cols,
            treatment_col=self.treatment_col,
            outcome_col=self.outcome_col,
            cate_col=self.cate_col,
        )
        trimmed_dataset._propensity_scores = self._propensity_scores[mask]
        return trimmed_dataset

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "CausalDataset":
        """Load dataset directly from CSV."""
        df = pd.read_csv(path)
        return cls(df=df, **kwargs)
