"""Иерархическая кластеризация: single, complete, average, ward."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

METHODS = ("single", "complete", "average", "ward")

METHOD_LABELS = {
    "single": "Single linkage",
    "complete": "Complete linkage",
    "average": "Group average",
    "ward": "Ward",
}

ALIASES = {
    "single": "single",
    "single_linkage": "single",
    "complete": "complete",
    "complete_linkage": "complete",
    "average": "average",
    "group_average": "average",
    "group-average": "average",
    "ward": "ward",
    "ward_distance": "ward",
}


def normalize_method(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key not in ALIASES:
        raise ValueError("Мера: single | complete | average | ward")
    return ALIASES[key]


def cluster(X: np.ndarray, method: str):
    """Евклидово расстояние между объектами, выбранная мера между кластерами."""
    method = normalize_method(method)
    return linkage(np.asarray(X, dtype=float), method=method, metric="euclidean")


def cut(Z: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k должно быть >= 1")
    return fcluster(Z, t=k, criterion="maxclust").astype(int)


def color_threshold(Z: np.ndarray, k: int) -> float:
    """Высота, ниже которой дендрограмма окрашена в k цветов."""
    n_merges = Z.shape[0]
    # строка n - k : слияние, после которого кластеров становится k - 1
    index = n_merges - (k - 1)
    index = min(max(index, 0), n_merges - 1)
    return float(Z[index, 2])
