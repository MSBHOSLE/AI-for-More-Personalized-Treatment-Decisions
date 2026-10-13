"""
Doubly Robust Learner (AIPW / Augmented Inverse Probability Weighting with Cross-Fitting).
Combines outcome regression and propensity score inverse weighting:
Consistent if either the outcome model OR propensity model is correctly specified.
Applies K-fold cross-fitting for Neyman orthogonality (Chernozhukov et al. 2018).
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


class DoublyRobustLearner:
    """
    Doubly Robust (AIPW) estimator for ATE and CATE.
    Computes cross-fitted doubly robust pseudo-outcomes Gamma_i:
      Gamma_i = mu_1(X_i) - mu_0(X_i) + T_i * (Y_i - mu_1(X_i)) / e(X_i) - (1 - T_i) * (Y_i - mu_0(X_i)) / (1 - e(X_i))
    and fits a second-stage regressor predicting Gamma from X for heterogeneous effects.
    """

    def __init__(
        self,
        outcome_model=None,
        propensity_model=None,
        final_model=None,
        n_splits: int = 5,
        clip_bounds: Tuple[float, float] = (0.02, 0.98),
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.clip_bounds = clip_bounds
        self.random_state = random_state

        default_reg = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            min_samples_leaf=10,
            random_state=random_state,
        )

        self.outcome_model = clone(outcome_model) if outcome_model is not None else clone(default_reg)
        self.propensity_model = (
            clone(propensity_model)
            if propensity_model is not None
            else LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        )
        self.final_model = clone(final_model) if final_model is not None else clone(default_reg)

        self.gamma_scores = None
        self.fitted_mu0 = None
        self.fitted_mu1 = None
        self.fitted_propensity = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "DoublyRobustLearner":
        """Fit using K-fold cross-fitting to generate unconfounded pseudo-outcomes."""
        n = len(X)
        gamma = np.zeros(n, dtype=np.float64)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X):
            X_tr, t_tr, y_tr = X[train_idx], treatment[train_idx], outcome[train_idx]
            X_val, t_val, y_val = X[val_idx], treatment[val_idx], outcome[val_idx]

            # Fit propensity model on fold train
            ps_model = clone(self.propensity_model)
            ps_model.fit(X_tr, t_tr)
            e_val = ps_model.predict_proba(X_val)[:, 1]
            e_val = np.clip(e_val, self.clip_bounds[0], self.clip_bounds[1])

            # Fit outcome models on fold train
            mask_tr_0 = t_tr == 0
            mask_tr_1 = t_tr == 1

            mu0_model = clone(self.outcome_model)
            mu1_model = clone(self.outcome_model)

            mu0_model.fit(X_tr[mask_tr_0], y_tr[mask_tr_0])
            mu1_model.fit(X_tr[mask_tr_1], y_tr[mask_tr_1])

            mu0_val = mu0_model.predict(X_val)
            mu1_val = mu1_model.predict(X_val)

            # Compute AIPW pseudo-outcome for validation fold
            gamma_val = (
                (mu1_val - mu0_val)
                + (t_val * (y_val - mu1_val) / e_val)
                - ((1.0 - t_val) * (y_val - mu0_val) / (1.0 - e_val))
            )
            gamma[val_idx] = gamma_val

        self.gamma_scores = gamma

        # Fit final stage model predicting pseudo-outcome from X
        self.final_model.fit(X, self.gamma_scores)

        # Fit full outcome models for potential outcomes prediction
        self.fitted_mu0 = clone(self.outcome_model)
        self.fitted_mu1 = clone(self.outcome_model)
        self.fitted_mu0.fit(X[treatment == 0], outcome[treatment == 0])
        self.fitted_mu1.fit(X[treatment == 1], outcome[treatment == 1])

        self.is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect tau(X)."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")
        return self.final_model.predict(X)

    def predict_ate(self, X: np.ndarray = None) -> float:
        """
        Estimate Average Treatment Effect (ATE).
        Can return empirical mean of AIPW scores if X is None, or mean predicted CATE on X.
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")
        if X is None:
            return float(np.mean(self.gamma_scores))
        return float(np.mean(self.predict_cate(X)))

    def ate_standard_error(self) -> float:
        """Compute asymptotic standard error for ATE estimate using score variance."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted.")
        n = len(self.gamma_scores)
        return float(np.std(self.gamma_scores, ddof=1) / np.sqrt(n))

    def predict_potential_outcomes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate expected potential outcomes (Y(0), Y(1))."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")
        y0 = self.fitted_mu0.predict(X)
        cate = self.predict_cate(X)
        y1 = y0 + cate
        return y0, y1
