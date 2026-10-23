"""Дендрограммы и сравнение разреза с разделительной кластеризацией."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from hierarchical import METHOD_LABELS

COLORS = ["#1f4e79", "#c26e4a", "#2a9d8f", "#7b6b8a", "#d4a017", "#264653", "#e76f51", "#6b9080"]


def save_dendrogram(
    Z: np.ndarray,
    path: Path,
    title: str,
    color_threshold: float,
    truncate: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=truncate,
        color_threshold=color_threshold,
        above_threshold_color="#4a4a4a",
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        count_sort=True,
    )
    ax.axhline(color_threshold, color="#888", linestyle="--", linewidth=1, label="разрез на 6 кластеров")
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("Кластер (в скобках — число исходных клиентов)")
    ax.set_ylabel("Расстояние объединения")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dendrogram_grid(
    panels: list[tuple[str, np.ndarray, float]],
    path: Path,
    truncate: int = 16,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))
    for ax, (name, Z, threshold) in zip(axes.ravel(), panels):
        dendrogram(
            Z,
            ax=ax,
            truncate_mode="lastp",
            p=truncate,
            color_threshold=threshold,
            above_threshold_color="#4a4a4a",
            leaf_rotation=90,
            leaf_font_size=8,
            show_contracted=True,
            count_sort=True,
        )
        ax.axhline(threshold, color="#888", linestyle="--", linewidth=1)
        ax.grid(False)
        ax.set_title(METHOD_LABELS.get(name, name))
        ax.set_xlabel("Кластер (в скобках — число клиентов)")
        ax.set_ylabel("Расстояние")
    fig.suptitle("Дендрограммы при четырёх мерах схожести кластеров", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_scatter_grid(
    scores: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    axis_labels: tuple[str, str],
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0), sharex=True, sharey=True)
    for ax, (name, labels) in zip(axes.ravel(), panels):
        order = np.argsort(labels)
        colors = [COLORS[int(label) % len(COLORS)] for label in labels[order]]
        ax.scatter(
            scores[order, 0],
            scores[order, 1],
            c=colors,
            s=22,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(name)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
