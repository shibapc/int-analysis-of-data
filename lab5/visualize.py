"""Графики кластеров и ошибки."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "#1f4e79",
    "#c26e4a",
    "#2a9d8f",
    "#7b6b8a",
    "#d4a017",
    "#264653",
    "#e76f51",
    "#6b9080",
    "#3d5a80",
]


def _colors(labels: np.ndarray) -> list[str]:
    return [COLORS[int(label) % len(COLORS)] for label in labels]


def scatter_pair(
    scores: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_centers: np.ndarray,
    right_centers: np.ndarray,
    axis_labels: tuple[str, str],
    title: str,
    path: Path,
    left_name: str = "k-Means",
    right_name: str = "k-Medoids",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharex=True, sharey=True)
    panels = (
        (axes[0], left_labels, left_centers, left_name),
        (axes[1], right_labels, right_centers, right_name),
    )
    for ax, labels, centers, name in panels:
        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            c=_colors(labels),
            s=28,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c="white",
            edgecolors="#222",
            marker="X",
            s=90,
            linewidths=1.1,
            zorder=4,
            label="центр кластера",
        )
        ax.set_title(name)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.legend(frameon=False, loc="best")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scatter_grid(
    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    axis_labels: tuple[str, str],
    title: str,
    path: Path,
) -> None:
    """panels: (название, координаты n×2, метки, центры n_clusters×2)."""
    n = len(panels)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.0, 4.6 * rows), sharex=True, sharey=True)
    flat = np.atleast_1d(axes).ravel()
    for ax, (name, scores, labels, center) in zip(flat, panels):
        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            c=_colors(labels),
            s=22,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            center[:, 0],
            center[:, 1],
            c="white",
            edgecolors="#222",
            marker="X",
            s=80,
            linewidths=1.0,
            zorder=4,
        )
        ax.set_title(name)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def error_curves(
    ks: list[int],
    series: list[tuple[str, list[float], str]],
    ylabel: str,
    title: str,
    path: Path,
    mark_k: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, values, color in series:
        ax.plot(ks, values, marker="o", linewidth=2.2, color=color, label=name)
        for k, value in zip(ks, values):
            ax.annotate(
                f"{value:.0f}",
                (k, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=color,
            )
    if mark_k is not None:
        ax.axvline(mark_k, color="#888", linestyle="--", linewidth=1, label=f"k = {mark_k}")
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def error_by_noise(
    ks: list[int],
    curves: list[tuple[str, list[float]]],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    palette = ["#1f4e79", "#2a9d8f", "#c26e4a", "#7b6b8a", "#d4a017"]
    for (name, values), color in zip(curves, palette):
        ax.plot(ks, values, marker="o", linewidth=2.2, color=color, label=name)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)
    ax.set_title(title)
    ax.legend(frameon=False, title="Доля шума")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
