#!/usr/bin/env python3
"""CLI: классификация деревом решений (Census Income / Adult)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset import load_adult, load_dataset_for_split, split_xy
from decision_tree import DecisionTreeClassifier, normalize_criterion
from metrics import classification_metrics
from visualize import save_tree_plot

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Классификация набора данных деревом решений. "
            "Критерии: information_gain, gain_ratio, gini_index."
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
        help="Тестовый CSV (adult.test). Используется, если --train-ratio=1",
    )
    p.add_argument(
        "-c",
        "--criterion",
        default="gini_index",
        help="information_gain | gain_ratio | gini_index",
    )
    p.add_argument(
        "-r",
        "--train-ratio",
        type=float,
        default=1.0,
        help="Доля обучающей выборки от набора (0..1). "
        "1.0 = 100%% adult.data, тест = adult.test",
    )
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument(
        "--tree-plot",
        help="Путь для сохранения рисунка дерева",
    )
    p.add_argument(
        "--plot-depth",
        type=int,
        default=4,
        help="Глубина дерева на рисунке",
    )
    p.add_argument(
        "--metrics-json",
        help="Сохранить метрики в JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    criterion = normalize_criterion(args.criterion)

    if args.train_ratio >= 1.0:
        train, test, _, _ = load_adult(args.input, args.test)
        assert test is not None
        X_train, y_train = train.X, train.y
        X_test, y_test = test.X, test.y
        feature_names = train.feature_names
        target_names = train.target_names
        split_desc = "train=100% adult.data, test=adult.test"
    else:
        data = load_dataset_for_split(args.input)
        X_train, X_test, y_train, y_test = split_xy(data, args.train_ratio)
        feature_names = data.feature_names
        target_names = data.target_names
        split_desc = f"train={args.train_ratio:.0%}, test={1 - args.train_ratio:.0%} (от adult.data)"

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # положительный класс — >50K
    positive = list(target_names).index(">50K") if ">50K" in target_names else 1
    metrics = classification_metrics(y_test, pred, positive=positive)

    print(f"Критерий: {criterion}")
    print(f"Разбиение: {split_desc}")
    print(f"Обучающих объектов: {len(y_train)}")
    print(f"Тестовых объектов: {len(y_test)}")
    print(f"Глубина дерева: {model.get_depth()}")
    print(f"Число листьев: {model.get_n_leaves()}")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")

    if args.tree_plot:
        save_tree_plot(
            model,
            feature_names,
            target_names,
            args.tree_plot,
            title=f"Дерево решений ({criterion})",
            max_depth=args.plot_depth,
        )
        print(f"Рисунок дерева: {args.tree_plot}")

    if args.metrics_json:
        Path(args.metrics_json).write_text(
            json.dumps(
                {
                    "criterion": criterion,
                    "train_ratio": args.train_ratio,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "depth": model.get_depth(),
                    "n_leaves": model.get_n_leaves(),
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
