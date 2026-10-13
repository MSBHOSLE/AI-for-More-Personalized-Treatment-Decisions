"""
T-Learner (Two-Model Meta-Learner).
Fits separate response surfaces mu_0(X) for control and mu_1(X) for treated units:
tau(X) = mu_1(X) - mu_0(X).
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor


class TLearner:
    """
    T-Learner meta-estimator for Individualized Treatment Effect (ITE / CATE).
    Trains two distinct models:
      Model 0 on control subset {i : T_i = 0}
      Model 1 on treated subset {i : T_i = 1}
    """

    def __init__(
        self,
        base_model_0=None,
        base_model_1=None,
        random_state: int = 42,
    ):
        self.random_state = random_state

        default_model = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            min_samples_leaf=10,
            random_state=random_state,
        )

        self.model_0 = clone(base_model_0) if base_model_0 is not None else clone(default_model)
        self.model_1 = clone(base_model_1) if base_model_1 is not None else clone(default_model)
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "TLearner":
        """
        Fit two separate response surfaces for control and treated groups.
        """
        mask_0 = treatment == 0
        mask_1 = treatment == 1

        if np.sum(mask_0) < 5 or np.sum(mask_1) < 5:
            raise ValueError("Insufficient observations in one of the treatment arms.")

        self.model_0.fit(X[mask_0], outcome[mask_0])
        self.model_1.fit(X[mask_1], outcome[mask_1])
        self.is_fitted = True
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict expected outcomes (Y(0), Y(1)) for each observation."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")

        y0_pred = self.model_0.predict(X)
        y1_pred = self.model_1.predict(X)
        return y0_pred, y1_pred

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect tau(X) = E[Y(1) - Y(0) | X]."""
        y0_pred, y1_pred = self.predict_potential_outcomes(X)
        return y1_pred - y0_pred

    def predict_ate(self, X: np.ndarray) -> float:
        """Estimate Average Treatment Effect (ATE)."""
        cate = self.predict_cate(X)
        return float(np.mean(cate))
