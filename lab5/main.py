#!/usr/bin/env python3
"""Кластеризация k-Means и k-Medoids. Параметры: набор, k, доля шума."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dataset import add_noise, load_customers, to_original
from kmeans import kmeans
from kmedoids import kmedoids

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Кластеризация набора данных алгоритмами k-Means и k-Medoids. "
            "Печатает координаты объектов, номер кластера и ошибку."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "customers" / "customers.csv"),
        help="CSV с объектами",
    )
    parser.add_argument("-k", "--clusters", type=int, required=True, help="Число кластеров")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Доля зашумленных объектов, от 0 до 1 (0.05 = 5%%). По умолчанию 0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save",
        help="Сохранить назначения в CSV",
    )
    return parser


def _print_result(name: str, error_name: str, frame, labels, error: float) -> None:
    print(f"Алгоритм: {name}")
    print(f"Число кластеров: {len(set(labels.tolist()))}")
    print(f"Ошибка кластеризации ({error_name}): {error:.4f}")
    header = " ".join(f"{name:>16}" for name in frame.columns) + f" {'cluster':>8}"
    print(header)
    for index, row in frame.iterrows():
        values = " ".join(f"{float(row[col]):16.4f}" for col in frame.columns)
        print(f"{values} {int(labels[index]):8d}")
    sizes = np.bincount(labels)
    print("Размеры кластеров: " + ", ".join(f"{i} → {int(n)}" for i, n in enumerate(sizes)))
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_customers(args.input)
    features = data.X
    if args.noise > 0:
        features, noisy_index = add_noise(
            features,
            args.noise,
            np.random.default_rng(args.seed),
        )
        print(
            f"Шум: {args.noise:.0%} объектов ({len(noisy_index)} из {len(features)}), "
            "1–3 координаты, амплитуда 1.5σ"
        )
    elif args.noise < 0:
        raise SystemExit("Доля шума не может быть отрицательной")

    km = kmeans(features, args.clusters, random_state=args.seed)
    md = kmedoids(features, args.clusters, random_state=args.seed)
    _print_result("k-Means", "SSE, сумма квадратов расстояний до центроида", data.frame, km.labels, km.error)
    _print_result("k-Medoids", "сумма расстояний до медоида", data.frame, md.labels, md.error)

    centers = to_original(data, km.centers)
    medoids = to_original(data, md.centers)
    print("Центроиды k-Means в исходных единицах:")
    for index, row in enumerate(centers):
        pretty = ", ".join(f"{name}={value:.2f}" for name, value in zip(data.feature_names, row))
        print(f"  {index}: {pretty}")
    print("Медоиды k-Medoids в исходных единицах:")
    for index, row in enumerate(medoids):
        pretty = ", ".join(f"{name}={value:.2f}" for name, value in zip(data.feature_names, row))
        print(f"  {index}: {pretty}")

    if args.save:
        import pandas as pd

        out = data.frame.copy()
        out["kmeans"] = km.labels
        out["kmedoids"] = md.labels
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        print(f"Назначения: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
