"""
S-Learner (Single Model Meta-Learner).
Fits a unified response surface mu(X, T) and estimates:
tau(X) = mu(X, 1) - mu(X, 0).
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


class SLearner:
    """
    S-Learner meta-estimator for Individualized Treatment Effect (ITE / CATE).
    Trains a single model predicting outcome Y from covariates X and treatment indicator T.
    """

    def __init__(
        self,
        base_model=None,
        include_interactions: bool = True,
        random_state: int = 42,
    ):
        self.include_interactions = include_interactions
        self.random_state = random_state

        if base_model is None:
            self.model = GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                min_samples_leaf=10,
                random_state=random_state,
            )
        else:
            self.model = clone(base_model)

        self.is_fitted = False

    def _build_features(self, X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Construct [X, T] or [X, T, X * T] design matrix."""
        t_col = treatment.reshape(-1, 1)
        if self.include_interactions:
            interactions = X * t_col
            return np.hstack([X, t_col, interactions])
        else:
            return np.hstack([X, t_col])

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "SLearner":
        """
        Fit single response surface model.
        Args:
            X: Covariate matrix (n_samples, n_features)
            treatment: Binary treatment vector (n_samples,)
            outcome: Observed clinical outcome vector (n_samples,)
        """
        X_design = self._build_features(X, treatment)
        self.model.fit(X_design, outcome)
        self.is_fitted = True
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict expected outcomes (Y(0), Y(1)) for each observation."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")

        n = len(X)
        zeros = np.zeros(n, dtype=int)
        ones = np.ones(n, dtype=int)

        X_ctrl = self._build_features(X, zeros)
        X_treat = self._build_features(X, ones)

        y0_pred = self.model.predict(X_ctrl)
        y1_pred = self.model.predict(X_treat)

        return y0_pred, y1_pred

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect tau(X) = E[Y(1) - Y(0) | X]."""
        y0_pred, y1_pred = self.predict_potential_outcomes(X)
        return y1_pred - y0_pred

    def predict_ate(self, X: np.ndarray) -> float:
        """Estimate Average Treatment Effect (ATE)."""
        cate = self.predict_cate(X)
        return float(np.mean(cate))
