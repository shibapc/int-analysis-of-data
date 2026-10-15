#!/usr/bin/env python3
"""CLI: ансамблевая классификация (Census Income / Adult)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset import load_adult
from ensemble import TECHNIQUE_LABELS, build_model, resolve_spec
from metrics import classification_metrics

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Классификация набора данных ансамблем. "
            "Техники: bagging, random_forest, boosting."
        )
    )
    p.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "data" / "adult.data.csv"),
        help="Обучающий CSV (adult.data)",
    )
    p.add_argument(
        "--test",
        default=str(ROOT / "data" / "adult.test.csv"),
        help="Тестовый CSV (adult.test)",
    )
    p.add_argument(
        "-t",
        "--technique",
        required=True,
        help="bagging | random_forest | boosting",
    )
    p.add_argument(
        "-n",
        "--n-estimators",
        type=int,
        required=True,
        help="Число участников ансамбля",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Глубина базового дерева. По умолчанию 12 (бэггинг, лес) или 3 (бустинг)",
    )
    p.add_argument(
        "--min-samples-leaf",
        type=int,
        default=None,
        help="Минимум объектов в листе. По умолчанию 20",
    )
    p.add_argument(
        "--max-samples",
        type=float,
        default=None,
        help="Доля объектов в бутстреп-выборке бэггинга (0, 1]. По умолчанию 1.0",
    )
    p.add_argument(
        "--max-features",
        default=None,
        help="Число признаков в участнике: число, доля, sqrt, log2. "
        "По умолчанию 1.0 для бэггинга и sqrt для случайного леса",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Темп обучения бустинга. По умолчанию 0.1",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--metrics-json",
        help="Сохранить метрики в JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = resolve_spec(
        args.technique,
        args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )

    train, test, _, _ = load_adult(args.input, args.test)
    if test is None:
        raise SystemExit("Нужен тестовый набор: передайте --test")

    positive = list(train.target_names).index(">50K") if ">50K" in train.target_names else 1
    model = build_model(spec)
    model.fit(train.X, train.y)
    pred = model.predict(test.X)
    metrics = classification_metrics(test.y, pred, positive=positive)
    applied = spec.public_params()

    print(f"Техника: {TECHNIQUE_LABELS[spec.technique]} ({spec.technique})")
    print("Параметры: " + ", ".join(f"{k}={v}" for k, v in applied.items() if k != "technique"))
    print("Разбиение: train=100% adult.data, test=adult.test")
    print(f"Обучающих объектов: {len(train.y)}")
    print(f"Тестовых объектов: {len(test.y)}")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")

    if args.metrics_json:
        Path(args.metrics_json).write_text(
            json.dumps(
                {
                    **applied,
                    "n_train": int(len(train.y)),
                    "n_test": int(len(test.y)),
                    **metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
