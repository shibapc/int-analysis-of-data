#!/usr/bin/env python3
"""Иерархическая кластеризация. Параметры: набор данных и мера схожести кластеров."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dataset import load_customers
from hierarchical import METHOD_LABELS, cluster, color_threshold, cut, normalize_method

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Иерархическая кластеризация. Мера между объектами — евклидово расстояние. "
            "Мера между кластерами: single, complete, average, ward."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "customers" / "customers.csv"),
        help="CSV с объектами",
    )
    parser.add_argument(
        "-m",
        "--method",
        required=True,
        help="single | complete | average | ward",
    )
    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=6,
        help="На сколько кластеров разрезать дерево для раскраски и размеров (по умолчанию 6)",
    )
    parser.add_argument(
        "--plot",
        default="",
        help="Куда сохранить дендрограмму. По умолчанию results/figures/dendrogram_<method>.png",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    method = normalize_method(args.method)
    data = load_customers(args.input)
    Z = cluster(data.X, method)
    labels = cut(Z, args.clusters)
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]

    print(f"Мера схожести кластеров: {METHOD_LABELS[method]} ({method})")
    print("Мера схожести объектов: euclidean")
    print(f"Объектов: {len(data.X)}")
    print(f"Слияний в дереве: {len(Z)}")
    print(f"Высота последнего слияния: {Z[-1, 2]:.4f}")
    print(f"Разрез на {args.clusters} кластеров, размеры: " + ", ".join(str(int(n)) for n in sorted(sizes, reverse=True)))

    from visualize import save_dendrogram

    plot = Path(args.plot) if args.plot else ROOT / "results" / "figures" / f"dendrogram_{method}.png"
    save_dendrogram(
        Z,
        plot,
        f"{METHOD_LABELS[method]}: дендрограмма",
        color_threshold(Z, args.clusters),
    )
    print(f"Дендрограмма: {plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
