"""k-Means: несколько запусков, инициализация k-means++."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    error: float
    iterations: int


def _pairwise_sq(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1)
    return np.maximum(x2 + c2 - 2.0 * X @ centers.T, 0.0)


def _init_kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    first = int(rng.integers(n))
    centers = [X[first]]
    closest = _pairwise_sq(X, np.vstack(centers)).ravel()
    for _ in range(1, k):
        total = closest.sum()
        if total <= 1e-12:
            centers.append(X[int(rng.integers(n))])
        else:
            probs = closest / total
            centers.append(X[int(rng.choice(n, p=probs))])
        closest = _pairwise_sq(X, np.vstack(centers)).min(axis=1)
    return np.vstack(centers)


def _fit_once(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
) -> ClusterResult:
    centers = _init_kmeans_pp(X, k, rng)
    labels = np.zeros(X.shape[0], dtype=int)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        distances = _pairwise_sq(X, centers)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members) == 0:
                farthest = int(distances.min(axis=1).argmax())
                updated[cluster] = X[farthest]
            else:
                updated[cluster] = members.mean(axis=0)
        shift = np.linalg.norm(updated - centers)
        centers = updated
        if shift <= tol:
            break
    error = float(_pairwise_sq(X, centers)[np.arange(len(X)), labels].sum())
    return ClusterResult(labels=labels, centers=centers, error=error, iterations=iterations)


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> ClusterResult:
    X = np.asarray(X, dtype=float)
    if k < 1 or k > len(X):
        raise ValueError("k должно быть от 1 до числа объектов")
    rng = np.random.default_rng(random_state)
    best: ClusterResult | None = None
    for _ in range(n_init):
        seed = int(rng.integers(0, 2**31 - 1))
        result = _fit_once(X, k, np.random.default_rng(seed), max_iter, tol)
        if best is None or result.error < best.error:
            best = result
    assert best is not None
    return best
