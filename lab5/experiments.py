"""Эксперименты k-Means и k-Medoids на customers, рисунки и сводка."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import FEATURE_LABELS, add_noise, load_customers, to_original
from kmeans import kmeans
from kmedoids import _distance_matrix, kmedoids
from visualize import error_by_noise, error_curves, scatter_grid, scatter_pair

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

KS = list(range(3, 10))
NOISE_LEVELS = (0.0, 0.01, 0.03, 0.05, 0.10)
SEED = 42


def fit_pca(X: np.ndarray):
    mean = X.mean(axis=0)
    centered = X - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    variance = (singular ** 2) / max(len(X) - 1, 1)
    ratio = variance / variance.sum()
    return mean, components, ratio[:2]


def project(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T


def relabel(labels: np.ndarray, income: np.ndarray, centers: np.ndarray):
    order = sorted(set(int(c) for c in labels), key=lambda c: float(income[labels == c].mean()))
    mapping = {old: new for new, old in enumerate(order)}
    new_labels = np.array([mapping[int(label)] for label in labels], dtype=int)
    new_centers = centers[order]
    return new_labels, new_centers


def profiles(frame: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    rows = []
    for cluster in sorted(set(int(c) for c in labels)):
        part = frame.loc[labels == cluster]
        row = {"cluster": cluster, "n": int(len(part))}
        for column in frame.columns:
            row[column] = float(part[column].mean())
        rows.append(row)
    return rows


def elbow_k(ks: list[int], errors: list[float]) -> int:
    x = np.asarray(ks, dtype=float)
    y = np.asarray(errors, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    span = y.max() - y.min()
    y_n = (y - y.min()) / span if span > 0 else np.zeros_like(y)
    line = y_n[0] + (y_n[-1] - y_n[0]) * x_n
    return int(ks[int(np.argmax(line - y_n))])


def run() -> dict:
    data = load_customers(ROOT / "customers" / "customers.csv")
    pca_mean, components, ratio = fit_pca(data.X)
    loadings = {
        name: [float(components[0, i]), float(components[1, i])]
        for i, name in enumerate(data.feature_names)
    }
    income = data.frame["Income"].to_numpy()
    clean_scores = project(data.X, pca_mean, components)
    axis = (
        f"ГК 1 ({ratio[0] * 100:.0f}% дисперсии)",
        f"ГК 2 ({ratio[1] * 100:.0f}% дисперсии)",
    )

    summary: dict = {
        "n": int(len(data.X)),
        "features": data.feature_names,
        "feature_labels": FEATURE_LABELS,
        "pca_ratio": [float(ratio[0]), float(ratio[1])],
        "loadings": loadings,
        "axis": list(axis),
        "levels": {},
    }
    stored = {}

    for fraction in NOISE_LEVELS:
        rng = np.random.default_rng(SEED + int(round(fraction * 1000)))
        features, noisy_index = add_noise(data.X, fraction, rng)
        distances = _distance_matrix(features)
        level = {
            "n_noisy": int(len(noisy_index)),
            "kmeans": {},
            "kmedoids": {},
        }
        stored[fraction] = {"features": features, "kmeans": {}, "kmedoids": {}}
        for k in KS:
            km = kmeans(features, k, random_state=SEED)
            md = kmedoids(features, k, random_state=SEED, distances=distances)
            km_labels, km_centers = relabel(km.labels, income, km.centers)
            md_labels, md_centers = relabel(md.labels, income, md.centers)
            level["kmeans"][str(k)] = km.error
            level["kmedoids"][str(k)] = md.error
            stored[fraction]["kmeans"][k] = (km_labels, km_centers, km.error)
            stored[fraction]["kmedoids"][k] = (md_labels, md_centers, md.error)
            print(
                f"noise={fraction:.0%} k={k}: "
                f"SSE={km.error:.1f} medoid={md.error:.1f}"
            )
        summary["levels"][f"{fraction:.2f}"] = level

    clean_errors = [summary["levels"]["0.00"]["kmeans"][str(k)] for k in KS]
    chosen = elbow_k(KS, clean_errors)
    summary["chosen_k"] = chosen

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    error_curves(
        KS,
        [("k-Means", clean_errors, "#1f4e79")],
        "SSE",
        "k-Means: ошибка кластеризации",
        fig_dir / "error_kmeans.png",
        mark_k=chosen,
    )
    error_curves(
        KS,
        [("k-Medoids", [summary["levels"]["0.00"]["kmedoids"][str(k)] for k in KS], "#c26e4a")],
        "Сумма расстояний до медоида",
        "k-Medoids: ошибка кластеризации",
        fig_dir / "error_kmedoids.png",
        mark_k=chosen,
    )

    km_labels, km_centers, _ = stored[0.0]["kmeans"][chosen]
    md_labels, md_centers, _ = stored[0.0]["kmedoids"][chosen]
    scatter_pair(
        clean_scores,
        km_labels,
        md_labels,
        project(km_centers, pca_mean, components),
        project(md_centers, pca_mean, components),
        axis,
        f"Кластеры клиентов при k = {chosen}",
        fig_dir / "scatter_clean.png",
    )

    for algo, ylabel, filename in (
        ("kmeans", "SSE", "error_noise_kmeans.png"),
        ("kmedoids", "Сумма расстояний до медоида", "error_noise_kmedoids.png"),
    ):
        curves = []
        for fraction in NOISE_LEVELS:
            key = f"{fraction:.2f}"
            values = [summary["levels"][key][algo][str(k)] for k in KS]
            label = "0%" if fraction == 0 else f"{fraction:.0%}"
            curves.append((label, values))
        error_by_noise(
            KS,
            curves,
            ylabel,
            "Ошибка при разной доле зашумленных объектов",
            fig_dir / filename,
        )

    for algo, filename, title in (
        ("kmeans", "scatter_noise_kmeans.png", "k-Means"),
        ("kmedoids", "scatter_noise_kmedoids.png", "k-Medoids"),
    ):
        panels = []
        for fraction in (0.01, 0.03, 0.05, 0.10):
            labels, center, _ = stored[fraction][algo][chosen]
            scores = project(stored[fraction]["features"], pca_mean, components)
            panels.append(
                (
                    f"шум {fraction:.0%}",
                    scores,
                    labels,
                    project(center, pca_mean, components),
                )
            )
        scatter_grid(
            panels,
            axis,
            f"{title}, k = {chosen}. Цвет точки — номер кластера",
            fig_dir / filename,
        )

    # профили и исходные координаты центров
    def pack(algo: str, fraction: float) -> dict:
        labels, centers, error = stored[fraction][algo][chosen]
        original = to_original(data, centers)
        return {
            "error": error,
            "sizes": [int(np.sum(labels == c)) for c in range(chosen)],
            "profiles": profiles(data.frame, labels),
            "centers_original": original.tolist(),
        }

    summary["clean_detail"] = {
        "kmeans": pack("kmeans", 0.0),
        "kmedoids": pack("kmedoids", 0.0),
    }
    summary["noise_detail"] = {
        f"{fraction:.2f}": {
            "kmeans": pack("kmeans", fraction),
            "kmedoids": pack("kmedoids", fraction),
        }
        for fraction in NOISE_LEVELS
        if fraction > 0
    }

    out = ROOT / "results" / "experiment_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"chosen_k={chosen}")
    print(f"saved {out}")
    return summary


if __name__ == "__main__":
    run()
