"""Диаграмма качества ансамбля и линии одиночного дерева решений."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ensemble import TECHNIQUE_LABELS

TECHNIQUE_ORDER = ("bagging", "random_forest", "boosting")
TECHNIQUE_COLORS = {
    "bagging": "#1f4e79",
    "random_forest": "#c26e4a",
    "boosting": "#6b9080",
}
BASELINE_COLOR = "#7b6b8a"

METRIC_PANELS = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F-мера"),
)


def plot_metrics_vs_estimators(
    rows: list[dict],
    baseline: dict,
    path: str | Path,
) -> Path:
    """Четыре метрики против числа участников; пунктир — дерево решений."""
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 9.4), sharex=True)
    xs_ticks = sorted({int(r["n_estimators"]) for r in rows})

    for ax, (key, title) in zip(axes.ravel(), METRIC_PANELS):
        values = [float(baseline[key])]
        for technique in TECHNIQUE_ORDER:
            series = sorted(
                (r for r in rows if r["technique"] == technique),
                key=lambda r: r["n_estimators"],
            )
            xs = [r["n_estimators"] for r in series]
            ys = [r[key] for r in series]
            values.extend(ys)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=TECHNIQUE_COLORS[technique],
                label=TECHNIQUE_LABELS[technique],
            )
        tree_value = float(baseline[key])
        ax.axhline(
            tree_value,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.6,
            label=f"Дерево решений, gini ({tree_value:.3f})",
        )
        lo, hi = min(values), max(values)
        pad = max(0.008, (hi - lo) * 0.35)
        top_pad = pad * 2.4 if key in {"recall", "f1"} else pad
        ax.set_ylim(lo - pad, hi + top_pad)
        ax.set_title(title)
        ax.set_xlabel("Число участников ансамбля")
        ax.set_ylabel(title)
        ax.set_xticks(xs_ticks)
        legend_loc = {
            "accuracy": "upper left",
            "precision": "lower right",
            "recall": "upper right",
            "f1": "upper left",
        }[key]
        ax.legend(frameon=False, fontsize=8, loc=legend_loc)

    fig.suptitle(
        "Качество классификации в зависимости от размера ансамбля",
        fontsize=13,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
