"""Дерево решений с критериями Information gain, Gain ratio и Gini index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Criterion = Literal["information_gain", "gain_ratio", "gini_index"]

CRITERION_ALIASES = {
    "information_gain": "information_gain",
    "info_gain": "information_gain",
    "entropy": "information_gain",
    "gain_ratio": "gain_ratio",
    "gain-ratio": "gain_ratio",
    "gini": "gini_index",
    "gini_index": "gini_index",
    "gini-index": "gini_index",
}


def normalize_criterion(name: str) -> Criterion:
    key = name.strip().lower().replace(" ", "_")
    if key not in CRITERION_ALIASES:
        raise ValueError(
            "Критерий: information_gain | gain_ratio | gini_index"
        )
    return CRITERION_ALIASES[key]  # type: ignore[return-value]


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    return float(1.0 - (probs * probs).sum())


def _class_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y, minlength=n_classes).astype(float)


@dataclass
class TreeNode:
    feature: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None
    prediction: int | None = None
    n_samples: int = 0
    class_counts: list[int] = field(default_factory=list)
    impurity: float = 0.0


class DecisionTreeClassifier:
    """Бинарное дерево решений (CART-подобное) с тремя критериями выбора разбиения."""

    def __init__(
        self,
        criterion: str = "gini_index",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_thresholds: int = 32,
        random_state: int | None = 42,
    ) -> None:
        self.criterion: Criterion = normalize_criterion(criterion)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.tree_: TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_features_ = X.shape[1]
        self.n_classes_ = int(y.max()) + 1
        self._gain_sum = np.zeros(self.n_features_, dtype=float)
        self.tree_ = self._build(X, y, depth=0)
        total = self._gain_sum.sum()
        if total > 0:
            self.feature_importances_ = self._gain_sum / total
        else:
            self.feature_importances_ = np.zeros(self.n_features_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(self.tree_, row) for row in X], dtype=int)

    def get_depth(self) -> int:
        return self._depth(self.tree_)

    def get_n_leaves(self) -> int:
        return self._n_leaves(self.tree_)

    def _predict_one(self, node: TreeNode, row: np.ndarray) -> int:
        while node.prediction is None:
            assert node.feature is not None and node.threshold is not None
            assert node.left is not None and node.right is not None
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return int(node.prediction)

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        counts = _class_counts(y, self.n_classes_)
        impurity = (
            _gini(counts)
            if self.criterion == "gini_index"
            else _entropy(counts)
        )
        node = TreeNode(
            n_samples=len(y),
            class_counts=counts.astype(int).tolist(),
            impurity=impurity,
            prediction=int(np.argmax(counts)),
        )

        pure = np.count_nonzero(counts) <= 1
        depth_limited = self.max_depth is not None and depth >= self.max_depth
        too_small = len(y) < self.min_samples_split
        if pure or depth_limited or too_small:
            return node

        split = self._best_split(X, y, counts)
        if split is None:
            return node

        feature, threshold, gain = split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return node

        self._gain_sum[feature] += gain * len(y)
        node.feature = feature
        node.threshold = float(threshold)
        node.prediction = None
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _candidate_thresholds(self, column: np.ndarray) -> np.ndarray:
        values = np.unique(column)
        if len(values) <= 1:
            return np.array([])
        if len(values) <= self.max_thresholds:
            return (values[:-1] + values[1:]) / 2.0
        qs = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
        return np.unique(np.quantile(column, qs))

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_counts: np.ndarray,
    ) -> tuple[int, float, float] | None:
        parent_impurity = (
            _gini(parent_counts)
            if self.criterion == "gini_index"
            else _entropy(parent_counts)
        )
        n = len(y)
        best_score = -1.0
        best: tuple[int, float, float] | None = None

        for feature in range(self.n_features_):
            thresholds = self._candidate_thresholds(X[:, feature])
            col = X[:, feature]
            for threshold in thresholds:
                left_mask = col <= threshold
                n_left = int(left_mask.sum())
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_counts = _class_counts(y[left_mask], self.n_classes_)
                right_counts = parent_counts - left_counts

                if self.criterion == "gini_index":
                    left_imp = _gini(left_counts)
                    right_imp = _gini(right_counts)
                    weighted = (n_left * left_imp + n_right * right_imp) / n
                    score = parent_impurity - weighted  # delta gini
                else:
                    left_imp = _entropy(left_counts)
                    right_imp = _entropy(right_counts)
                    weighted = (n_left * left_imp + n_right * right_imp) / n
                    info_gain = parent_impurity - weighted
                    if self.criterion == "information_gain":
                        score = info_gain
                    else:
                        # gain ratio
                        p_l, p_r = n_left / n, n_right / n
                        split_info = 0.0
                        for p in (p_l, p_r):
                            if p > 0:
                                split_info -= p * np.log2(p)
                        if split_info <= 1e-12:
                            continue
                        score = info_gain / split_info

                if score > best_score + 1e-15:
                    best_score = score
                    best = (feature, float(threshold), float(score))

        if best is None or best_score <= 1e-15:
            return None
        return best

    def _depth(self, node: TreeNode | None) -> int:
        if node is None or node.prediction is not None:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def _n_leaves(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        if node.prediction is not None:
            return 1
        return self._n_leaves(node.left) + self._n_leaves(node.right)
