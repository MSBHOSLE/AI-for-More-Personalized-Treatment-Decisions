"""
X-Learner (Künzel et al. 2019 Meta-Learner).
Particularly effective under extreme treatment imbalance.
Imputes counterfactual residuals in two stages and weights using propensity scores:
tau(X) = e(X) * tau_0(X) + (1 - e(X)) * tau_1(X).
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


class XLearner:
    """
    X-Learner meta-estimator for heterogeneous treatment effect estimation.
    Stage 1: Fits mu_0 and mu_1 on control and treated groups.
    Stage 2: Imputes counterfactual effects D_1 = Y_1 - mu_0(X_1) and D_0 = mu_1(X_0) - Y_0.
             Fits tau_1 on D_1 and tau_0 on D_0.
    Stage 3: Weights predictions by propensity score e(X).
    """

    def __init__(
        self,
        base_stage1_model=None,
        base_stage2_model=None,
        propensity_model=None,
        random_state: int = 42,
    ):
        self.random_state = random_state

        default_reg = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            min_samples_leaf=10,
            random_state=random_state,
        )

        self.mu_0 = clone(base_stage1_model) if base_stage1_model is not None else clone(default_reg)
        self.mu_1 = clone(base_stage1_model) if base_stage1_model is not None else clone(default_reg)
        self.tau_0 = clone(base_stage2_model) if base_stage2_model is not None else clone(default_reg)
        self.tau_1 = clone(base_stage2_model) if base_stage2_model is not None else clone(default_reg)

        if propensity_model is None:
            self.propensity_model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        else:
            self.propensity_model = clone(propensity_model)

        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "XLearner":
        """Fit 2-stage X-Learner pipeline."""
        mask_0 = treatment == 0
        mask_1 = treatment == 1

        if np.sum(mask_0) < 5 or np.sum(mask_1) < 5:
            raise ValueError("Insufficient observations in one of the treatment arms.")

        X_0, y_0 = X[mask_0], outcome[mask_0]
        X_1, y_1 = X[mask_1], outcome[mask_1]

        # Stage 1: Fit outcome surfaces
        self.mu_0.fit(X_0, y_0)
        self.mu_1.fit(X_1, y_1)

        # Propensity model
        self.propensity_model.fit(X, treatment)

        # Stage 2: Impute counterfactual contrasts
        # For treated units: imputed treatment effect = factual Y_1 - counterfactual mu_0(X_1)
        d_1 = y_1 - self.mu_0.predict(X_1)
        # For control units: imputed treatment effect = counterfactual mu_1(X_0) - factual Y_0
        d_0 = self.mu_1.predict(X_0) - y_0

        # Fit second-stage effect estimators
        self.tau_1.fit(X_1, d_1)
        self.tau_0.fit(X_0, d_0)

        self.is_fitted = True
        return self

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """Estimate propensity scores e(X)."""
        probs = self.propensity_model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.01, 0.99)

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate Conditional Average Treatment Effect tau(X).
        tau(X) = e(X) * tau_0(X) + (1 - e(X)) * tau_1(X).
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")

        e = self.predict_propensity(X)
        tau_0_hat = self.tau_0.predict(X)
        tau_1_hat = self.tau_1.predict(X)

        cate = e * tau_0_hat + (1.0 - e) * tau_1_hat
        return cate

    def predict_potential_outcomes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate expected potential outcomes (Y(0), Y(1))."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")

        y0 = self.mu_0.predict(X)
        cate = self.predict_cate(X)
        y1 = y0 + cate
        return y0, y1

    def predict_ate(self, X: np.ndarray) -> float:
        """Estimate Average Treatment Effect (ATE)."""
        cate = self.predict_cate(X)
        return float(np.mean(cate))
