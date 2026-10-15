"""Метрики качества бинарной/мультиклассовой классификации."""

from __future__ import annotations

import numpy as np


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    fp = int(np.sum((y_true != positive) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred != positive)))
    tn = int(np.sum((y_true != positive) & (y_pred != positive)))
    return tp, fp, fn, tn


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    tp, fp, _, _ = _binary_counts(y_true, y_pred, positive)
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    tp, _, fn, _ = _binary_counts(y_true, y_pred, positive)
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    p = precision_score(y_true, y_pred, positive)
    r = recall_score(y_true, y_pred, positive)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive: int = 1,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, positive),
        "recall": recall_score(y_true, y_pred, positive),
        "f1": f1_score(y_true, y_pred, positive),
    }
