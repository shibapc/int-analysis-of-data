"""k-Medoids: медоид — объект кластера с минимальной суммой расстояний."""

from __future__ import annotations

import numpy as np

from kmeans import ClusterResult, _init_kmeans_pp


def _distance_matrix(X: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    sq = np.maximum(x2 + x2.T - 2.0 * X @ X.T, 0.0)
    np.fill_diagonal(sq, 0.0)
    return np.sqrt(sq)


def _fit_once(
    distances: np.ndarray,
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
) -> ClusterResult:
    n = distances.shape[0]
    seeds = _init_kmeans_pp(X, k, rng)
    # ближайший реальный объект к каждому стартовому центру
    seed_dist = ((X[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    medoids = np.empty(k, dtype=int)
    used: set[int] = set()
    for cluster, order in enumerate(np.argsort(seed_dist, axis=0).T):
        for index in order:
            index = int(index)
            if index not in used:
                medoids[cluster] = index
                used.add(index)
                break
    labels = np.zeros(n, dtype=int)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        labels = distances[:, medoids].argmin(axis=1)
        new_medoids = medoids.copy()
        taken: set[int] = set()
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster)
            if len(members) == 0:
                farthest = int(np.argsort(distances[:, medoids].min(axis=1))[-1])
                if farthest in taken:
                    continue
                new_medoids[cluster] = farthest
                taken.add(farthest)
                continue
            costs = distances[np.ix_(members, members)].sum(axis=1)
            choice = int(members[int(costs.argmin())])
            new_medoids[cluster] = choice
            taken.add(choice)
        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            medoids = new_medoids
            break
        medoids = new_medoids
    labels = distances[:, medoids].argmin(axis=1)
    error = float(distances[np.arange(n), medoids[labels]].sum())
    return ClusterResult(
        labels=labels,
        centers=X[medoids].copy(),
        error=error,
        iterations=iterations,
    )


def kmedoids(
    X: np.ndarray,
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 50,
    random_state: int = 42,
    distances: np.ndarray | None = None,
) -> ClusterResult:
    X = np.asarray(X, dtype=float)
    if k < 1 or k > len(X):
        raise ValueError("k должно быть от 1 до числа объектов")
    if distances is None:
        distances = _distance_matrix(X)
    rng = np.random.default_rng(random_state)
    best: ClusterResult | None = None
    for _ in range(n_init):
        seed = int(rng.integers(0, 2**31 - 1))
        result = _fit_once(distances, X, k, np.random.default_rng(seed), max_iter)
        if best is None or result.error < best.error:
            best = result
    assert best is not None
    return best
