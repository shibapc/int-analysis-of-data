"""Визуализация дерева решений."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from decision_tree import DecisionTreeClassifier, TreeNode


def _node_label(
    node: TreeNode,
    feature_names: list[str],
    target_names: list[str],
) -> str:
    total = max(node.n_samples, 1)
    dist = ", ".join(
        f"{name}: {cnt}" for name, cnt in zip(target_names, node.class_counts)
    )
    pred = target_names[node.prediction] if node.prediction is not None else "?"
    if node.prediction is not None and node.feature is None:
        return f"класс = {pred}\nn = {node.n_samples}\n{dist}"
    assert node.feature is not None and node.threshold is not None
    fname = feature_names[node.feature]
    return (
        f"{fname} ≤ {node.threshold:.3g}\n"
        f"n = {node.n_samples}\n{dist}"
    )


def _layout(node: TreeNode, depth: int = 0, x: float = 0.5, dx: float = 0.25, positions=None, id_counter=None):
    if positions is None:
        positions = {}
    if id_counter is None:
        id_counter = [0]
    node_id = id_counter[0]
    id_counter[0] += 1
    positions[node_id] = (x, -depth, node)
    if node.feature is not None:
        child_dx = dx / 1.7
        if node.left is not None:
            _layout(node.left, depth + 1, x - dx, child_dx, positions, id_counter)
        if node.right is not None:
            _layout(node.right, depth + 1, x + dx, child_dx, positions, id_counter)
    return positions


def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    target_names: list[str],
    title: str,
    max_depth: int | None = 4,
    figsize: tuple[float, float] = (16, 9),
) -> plt.Figure:
    """Нарисовать дерево (обрезая по max_depth для читаемости)."""
    if model.tree_ is None:
        raise RuntimeError("Модель не обучена")

    def truncate(node: TreeNode, depth: int) -> TreeNode:
        if node.feature is None or (max_depth is not None and depth >= max_depth):
            return TreeNode(
                prediction=node.prediction if node.prediction is not None else int(
                    max(range(len(node.class_counts)), key=lambda i: node.class_counts[i])
                ),
                n_samples=node.n_samples,
                class_counts=list(node.class_counts),
                impurity=node.impurity,
            )
        return TreeNode(
            feature=node.feature,
            threshold=node.threshold,
            left=truncate(node.left, depth + 1) if node.left else None,
            right=truncate(node.right, depth + 1) if node.right else None,
            n_samples=node.n_samples,
            class_counts=list(node.class_counts),
            impurity=node.impurity,
        )

    root = truncate(model.tree_, 0)
    positions = _layout(root, dx=0.35)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=12)

    # рёбра
    id_by_obj = {id(pos[2]): nid for nid, pos in positions.items()}

    def walk(node: TreeNode):
        nid = id_by_obj[id(node)]
        x, y, _ = positions[nid]
        if node.left is not None:
            cid = id_by_obj[id(node.left)]
            x2, y2, _ = positions[cid]
            ax.plot([x, x2], [y - 0.08, y2 + 0.08], color="#666", lw=1.2)
            ax.text((x + x2) / 2 - 0.02, (y + y2) / 2, "да", fontsize=7, color="#1f4e79")
            walk(node.left)
        if node.right is not None:
            cid = id_by_obj[id(node.right)]
            x2, y2, _ = positions[cid]
            ax.plot([x, x2], [y - 0.08, y2 + 0.08], color="#666", lw=1.2)
            ax.text((x + x2) / 2 + 0.01, (y + y2) / 2, "нет", fontsize=7, color="#c26e4a")
            walk(node.right)

    walk(root)

    for nid, (x, y, node) in positions.items():
        is_leaf = node.feature is None
        face = "#e8f0e8" if is_leaf else "#e8eef6"
        box = FancyBboxPatch(
            (x - 0.08, y - 0.09),
            0.16,
            0.16,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#1f4e79",
            facecolor=face,
            transform=ax.transData,
            clip_on=False,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            _node_label(node, feature_names, target_names),
            ha="center",
            va="center",
            fontsize=6.5,
            linespacing=1.2,
        )

    ys = [p[1] for p in positions.values()]
    xs = [p[0] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.15, max(xs) + 0.15)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.25)
    leaf = mpatches.Patch(color="#e8f0e8", label="лист")
    internal = mpatches.Patch(color="#e8eef6", label="узел разбиения")
    # lower left наезжает на крайний лист и обрезает подпись класса
    ax.legend(
        handles=[internal, leaf],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def save_tree_plot(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    target_names: list[str],
    path: str | Path,
    title: str,
    max_depth: int | None = 4,
) -> None:
    fig = plot_decision_tree(model, feature_names, target_names, title, max_depth=max_depth)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
