"""
Causal Forest (Athey & Wager 2019, Generalized Random Forests adaptation).
Constructs an ensemble of honest causal decision trees where:
1. Splits maximize variance of treatment effect differences between child leaves.
2. Honest estimation: disjoint samples for tree construction and leaf effect calculation.
3. Asymptotically normal treatment effect estimates with forest variance.
"""

import numpy as np
from typing import Optional, Tuple, List


class HonestCausalTree:
    """A single honest causal decision tree."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 15,
        max_features: Optional[float] = 0.8,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.tree_structure = None

    def _find_best_split(
        self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, feature_indices: np.ndarray
    ) -> Optional[Tuple[int, float]]:
        """Find split feature and threshold maximizing treatment effect variance."""
        n_samples = len(X)
        if n_samples < 2 * self.min_samples_leaf:
            return None

        best_score = -np.inf
        best_split = None

        for feat_idx in feature_indices:
            values = X[:, feat_idx]
            # Sample candidate split quantiles for efficiency
            quantiles = np.quantile(values, np.linspace(0.1, 0.9, 9))
            unique_thresholds = np.unique(quantiles)

            for thresh in unique_thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                t_l, t_r = treatment[left_mask], treatment[right_mask]
                y_l, y_r = outcome[left_mask], outcome[right_mask]

                # Require both arms in both leaves
                if np.sum(t_l == 1) < 2 or np.sum(t_l == 0) < 2 or np.sum(t_r == 1) < 2 or np.sum(t_r == 0) < 2:
                    continue

                tau_l = np.mean(y_l[t_l == 1]) - np.mean(y_l[t_l == 0])
                tau_r = np.mean(y_r[t_r == 1]) - np.mean(y_r[t_r == 0])

                # Criterion: heterogeneity between children
                score = (n_l * n_r / (n_samples**2)) * ((tau_l - tau_r) ** 2)

                if score > best_score:
                    best_score = score
                    best_split = (feat_idx, thresh)

        return best_split

    def _build_tree(
        self,
        X_split: np.ndarray,
        t_split: np.ndarray,
        y_split: np.ndarray,
        depth: int = 0,
    ) -> dict:
        """Recursively build tree splits."""
        n_feats = X_split.shape[1]
        n_sub_feats = max(1, int(self.max_features * n_feats)) if self.max_features else n_feats
        feat_sub = self.rng.choice(n_feats, size=n_sub_feats, replace=False)

        if depth >= self.max_depth or len(X_split) < 2 * self.min_samples_leaf:
            return {"is_leaf": True, "node_id": None}

        split = self._find_best_split(X_split, t_split, y_split, feat_sub)
        if split is None:
            return {"is_leaf": True, "node_id": None}

        feat_idx, thresh = split
        left_mask = X_split[:, feat_idx] <= thresh
        right_mask = ~left_mask

        return {
            "is_leaf": False,
            "feature": feat_idx,
            "threshold": thresh,
            "left": self._build_tree(X_split[left_mask], t_split[left_mask], y_split[left_mask], depth + 1),
            "right": self._build_tree(X_split[right_mask], t_split[right_mask], y_split[right_mask], depth + 1),
        }

    def _populate_leaf_effects(
        self,
        node: dict,
        X_eval: np.ndarray,
        t_eval: np.ndarray,
        y_eval: np.ndarray,
    ):
        """Honest evaluation: populate leaf values using the independent evaluation partition."""
        if node["is_leaf"]:
            t1_mask = t_eval == 1
            t0_mask = t_eval == 0
            y1_mean = np.mean(y_eval[t1_mask]) if np.sum(t1_mask) > 0 else 0.0
            y0_mean = np.mean(y_eval[t0_mask]) if np.sum(t0_mask) > 0 else 0.0

            node["tau"] = y1_mean - y0_mean if (np.sum(t1_mask) > 0 and np.sum(t0_mask) > 0) else 0.0
            node["y0"] = y0_mean
            node["y1"] = y1_mean
            node["n_eval"] = len(y_eval)
            return

        feat_idx = node["feature"]
        thresh = node["threshold"]
        left_mask = X_eval[:, feat_idx] <= thresh
        right_mask = ~left_mask

        self._populate_leaf_effects(node["left"], X_eval[left_mask], t_eval[left_mask], y_eval[left_mask])
        self._populate_leaf_effects(node["right"], X_eval[right_mask], t_eval[right_mask], y_eval[right_mask])

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, honesty: bool = True) -> "HonestCausalTree":
        """Fit honest causal tree."""
        n = len(X)
        if honesty:
            # 50/50 partition for split determination and leaf estimation
            indices = self.rng.permutation(n)
            split_idx = indices[: n // 2]
            eval_idx = indices[n // 2 :]

            X_split, t_split, y_split = X[split_idx], treatment[split_idx], outcome[split_idx]
            X_eval, t_eval, y_eval = X[eval_idx], treatment[eval_idx], outcome[eval_idx]
        else:
            X_split, t_split, y_split = X, treatment, outcome
            X_eval, t_eval, y_eval = X, treatment, outcome

        self.tree_structure = self._build_tree(X_split, t_split, y_split, depth=0)
        self._populate_leaf_effects(self.tree_structure, X_eval, t_eval, y_eval)
        return self

    def _predict_row(self, row: np.ndarray, node: dict) -> Tuple[float, float, float]:
        """Traverse tree to predict for a single sample."""
        if node["is_leaf"]:
            return node.get("tau", 0.0), node.get("y0", 0.0), node.get("y1", 0.0)
        feat_idx = node["feature"]
        thresh = node["threshold"]
        if row[feat_idx] <= thresh:
            return self._predict_row(row, node["left"])
        else:
            return self._predict_row(row, node["right"])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict (tau, y0, y1) for all rows."""
        tau_preds, y0_preds, y1_preds = [], [], []
        for row in X:
            t_pred, y0_p, y1_p = self._predict_row(row, self.tree_structure)
            tau_preds.append(t_pred)
            y0_preds.append(y0_p)
            y1_preds.append(y1_p)
        return np.array(tau_preds), np.array(y0_preds), np.array(y1_preds)


class CausalForest:
    """
    Causal Forest ensemble of honest causal trees.
    Aggregates honest causal trees to provide robust heterogeneous treatment effect estimation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 15,
        subsample_ratio: float = 0.75,
        honesty: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample_ratio = subsample_ratio
        self.honesty = honesty
        self.random_state = random_state
        self.trees: List[HonestCausalTree] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "CausalForest":
        """Train ensemble of honest causal trees with bootstrap/subsampling."""
        n_samples = len(X)
        subsample_size = int(self.subsample_ratio * n_samples)
        rng = np.random.default_rng(self.random_state)

        self.trees = []
        for i in range(self.n_estimators):
            sub_idx = rng.choice(n_samples, size=subsample_size, replace=False)
            tree = HonestCausalTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i * 17,
            )
            tree.fit(
                X[sub_idx],
                treatment[sub_idx],
                outcome[sub_idx],
                honesty=self.honesty,
            )
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect tau(X)."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")

        tree_preds = np.array([tree.predict(X)[0] for tree in self.trees])
        # Average across trees
        return np.mean(tree_preds, axis=0)

    def predict_potential_outcomes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate expected potential outcomes (Y(0), Y(1))."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict.")

        y0_preds = np.array([tree.predict(X)[1] for tree in self.trees])
        y1_preds = np.array([tree.predict(X)[2] for tree in self.trees])
        return np.mean(y0_preds, axis=0), np.mean(y1_preds, axis=0)

    def predict_ate(self, X: np.ndarray) -> float:
        """Estimate Average Treatment Effect (ATE)."""
        cate = self.predict_cate(X)
        return float(np.mean(cate))
