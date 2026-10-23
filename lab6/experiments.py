"""Четыре меры схожести, дендрограммы и сравнение с работой 5 при k = 6."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from dataset import load_customers
from hierarchical import METHODS, METHOD_LABELS, cluster, color_threshold, cut
from visualize import save_dendrogram, save_dendrogram_grid, save_scatter_grid

sys.path.insert(0, str(ROOT.parent / "lab5"))
from kmeans import kmeans  # noqa: E402
from kmedoids import kmedoids  # noqa: E402

sys.path.pop(0)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "figure.dpi": 120,
        "savefig.dpi": 170,
    }
)

COMPARE_K = 6


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    _, a_inv = np.unique(a, return_inverse=True)
    _, b_inv = np.unique(b, return_inverse=True)
    cont = np.zeros((a_inv.max() + 1, b_inv.max() + 1), dtype=np.int64)
    np.add.at(cont, (a_inv, b_inv), 1)

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2.0

    sum_ij = comb2(cont).sum()
    sum_a = comb2(cont.sum(axis=1)).sum()
    sum_b = comb2(cont.sum(axis=0)).sum()
    n = len(a)
    total = n * (n - 1) / 2.0
    expected = sum_a * sum_b / total if total else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    denom = maximum - expected
    if denom <= 1e-12:
        return 1.0
    return float((sum_ij - expected) / denom)


def relabel_by_income(labels: np.ndarray, income: np.ndarray) -> np.ndarray:
    order = sorted(set(int(c) for c in labels), key=lambda c: float(income[labels == c].mean()))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def profiles(frame, labels: np.ndarray) -> list[dict]:
    rows = []
    for cluster_id in sorted(set(int(c) for c in labels)):
        part = frame.loc[labels == cluster_id]
        row = {"cluster": cluster_id, "n": int(len(part))}
        for column in frame.columns:
            row[column] = float(part[column].mean())
        rows.append(row)
    return rows


def fit_pca(X: np.ndarray):
    mean = X.mean(axis=0)
    _, singular, vt = np.linalg.svd(X - mean, full_matrices=False)
    components = vt[:2]
    variance = (singular ** 2) / max(len(X) - 1, 1)
    ratio = variance / variance.sum()
    scores = (X - mean) @ components.T
    return scores, ratio[:2]


def run() -> dict:
    data = load_customers(ROOT / "customers" / "customers.csv")
    income = data.frame["Income"].to_numpy()
    scores, ratio = fit_pca(data.X)
    axis = (
        f"ГК 1 ({ratio[0] * 100:.0f}% дисперсии)",
        f"ГК 2 ({ratio[1] * 100:.0f}% дисперсии)",
    )

    km = kmeans(data.X, COMPARE_K, random_state=42)
    md = kmedoids(data.X, COMPARE_K, random_state=42)
    km_labels = relabel_by_income(km.labels, income)
    md_labels = relabel_by_income(md.labels, income)

    summary = {
        "n": int(len(data.X)),
        "compare_k": COMPARE_K,
        "pca_ratio": [float(ratio[0]), float(ratio[1])],
        "axis": list(axis),
        "kmeans_sizes": [int(np.sum(km_labels == c)) for c in range(COMPARE_K)],
        "kmedoids_sizes": [int(np.sum(md_labels == c)) for c in range(COMPARE_K)],
        "kmeans_sse": float(km.error),
        "kmedoids_error": float(md.error),
        "methods": {},
    }

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    grid = []
    scatter_panels = []

    for method in METHODS:
        Z = cluster(data.X, method)
        raw = cut(Z, COMPARE_K)
        labels = relabel_by_income(raw, income)
        threshold = color_threshold(Z, COMPARE_K)
        sizes = [int(np.sum(labels == c)) for c in range(COMPARE_K)]
        entry = {
            "label": METHOD_LABELS[method],
            "last_height": float(Z[-1, 2]),
            "cut_height": threshold,
            "sizes": sizes,
            "ari_kmeans": adjusted_rand(labels, km_labels),
            "ari_kmedoids": adjusted_rand(labels, md_labels),
            "profiles": profiles(data.frame, labels),
        }
        summary["methods"][method] = entry
        print(
            f"{method}: height={entry['last_height']:.2f} "
            f"sizes={sizes} ARI(km)={entry['ari_kmeans']:.3f} ARI(md)={entry['ari_kmedoids']:.3f}"
        )
        save_dendrogram(
            Z,
            fig_dir / f"dendrogram_{method}.png",
            f"{METHOD_LABELS[method]}",
            threshold,
        )
        grid.append((method, Z, threshold))
        scatter_panels.append((METHOD_LABELS[method], labels))

    save_dendrogram_grid(grid, fig_dir / "dendrograms.png")
    save_scatter_grid(
        scores,
        scatter_panels,
        axis,
        fig_dir / "scatter_k6.png",
        f"Разрез дендрограммы на {COMPARE_K} кластеров",
    )

    out = ROOT / "results" / "experiment_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return summary


if __name__ == "__main__":
    run()
