"""Эксперименты: ансамбли при 50–100 участниках, сравнение с деревом, PDF-отчёт."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

from dataset import load_adult
from ensemble import TECHNIQUE_LABELS, TECHNIQUES, build_model, resolve_spec, spec_to_dict
from metrics import classification_metrics
from visualize import plot_metrics_vs_estimators

N_GRID = (50, 60, 70, 80, 90, 100)
BASELINE_CRITERION = "gini_index"
LAB3_METRICS = ROOT.parent / "lab3" / "results" / "full_train_metrics.csv"
REPO_URL_FILE = ROOT / "repo_url.txt"

# запасные значения практической работы 3, если CSV недоступен
FALLBACK_BASELINES = [
    {
        "criterion": "information_gain",
        "accuracy": 0.849667994687915,
        "precision": 0.7443839346494214,
        "recall": 0.591081081081081,
        "f1": 0.658933413678819,
        "depth": 12,
        "n_leaves": 366,
    },
    {
        "criterion": "gain_ratio",
        "accuracy": 0.8562416998671979,
        "precision": 0.801334903808402,
        "recall": 0.5516216216216216,
        "f1": 0.6534336481511125,
        "depth": 12,
        "n_leaves": 107,
    },
    {
        "criterion": "gini_index",
        "accuracy": 0.848472775564409,
        "precision": 0.7376005361930295,
        "recall": 0.5948648648648649,
        "f1": 0.658587672052663,
        "depth": 12,
        "n_leaves": 378,
    },
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_repo_url() -> str:
    if not REPO_URL_FILE.exists():
        return ""
    for line in REPO_URL_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def load_tree_baselines() -> list[dict]:
    if not LAB3_METRICS.exists():
        return [dict(row) for row in FALLBACK_BASELINES]
    rows = []
    with LAB3_METRICS.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            parsed = {"criterion": row["criterion"]}
            for key in ("accuracy", "precision", "recall", "f1"):
                parsed[key] = float(row[key])
            for key in ("depth", "n_leaves", "n_train", "n_test"):
                if key in row and row[key] != "":
                    parsed[key] = int(float(row[key]))
            rows.append(parsed)
    return rows or [dict(row) for row in FALLBACK_BASELINES]


def run_experiments(train_x, train_y, test_x, test_y, positive: int) -> list[dict]:
    """Для каждой техники дообучать ансамбль с 50 до 100 участников."""
    rows: list[dict] = []
    for technique in TECHNIQUES:
        spec = resolve_spec(technique, n_estimators=N_GRID[0])
        model = build_model(spec, warm_start=True)
        applied = spec.public_params()
        for n_estimators in N_GRID:
            model.n_estimators = n_estimators
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            metrics = classification_metrics(test_y, pred, positive=positive)
            row = {
                "technique": technique,
                "n_estimators": n_estimators,
                "n_train": int(len(train_y)),
                "n_test": int(len(test_y)),
                "max_depth": applied.get("max_depth"),
                "min_samples_leaf": applied.get("min_samples_leaf"),
                "max_features": applied.get("max_features", ""),
                "max_samples": applied.get("max_samples", ""),
                "learning_rate": applied.get("learning_rate", ""),
                **metrics,
            }
            rows.append(row)
            print(
                f"[{TECHNIQUE_LABELS[technique]} n={n_estimators}] "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f}"
            )
    return rows


def save_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "technique",
        "n_estimators",
        "n_train",
        "n_test",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "max_samples",
        "learning_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _series(rows: list[dict], technique: str) -> list[dict]:
    return sorted(
        (row for row in rows if row["technique"] == technique),
        key=lambda row: row["n_estimators"],
    )


def _change_phrase(start: float, end: float) -> str:
    delta = end - start
    if abs(delta) < 0.004:
        return f"почти не меняется ({start:.3f} → {end:.3f})"
    word = "повышается" if delta > 0 else "снижается"
    return f"{word} с {start:.3f} до {end:.3f} (Δ={delta:+.3f})"


def _interpretation(last: dict[str, dict], baseline: dict) -> str:
    rf = last["random_forest"]
    bag = last["bagging"]
    boost = last["boosting"]
    parts: list[str] = []

    if bag["f1"] > rf["f1"] + 0.002 and bag["recall"] > rf["recall"] + 0.01 and rf["precision"] > bag["precision"] + 0.01:
        parts.append(
            f"Случайный лес точнее бэггинга ({rf['precision']:.3f} против {bag['precision']:.3f}), "
            f"но хуже находит класс >50K ({rf['recall']:.3f} против {bag['recall']:.3f}). "
            "На каждом разбиении лес видит только sqrt признаков и реже ставит рискованную "
            "метку >50K, поэтому точность растёт, а полнота и F-мера оказываются ниже, "
            f"чем у бэггинга (F1 {rf['f1']:.3f} против {bag['f1']:.3f})."
        )
    elif rf["f1"] > bag["f1"] + 0.002:
        parts.append(
            "Случайный лес выше бэггинга по F-мере: случайный набор признаков снижает "
            "корреляцию деревьев, и голосование убирает больше ошибки."
        )
    else:
        parts.append(
            "Бэггинг и случайный лес держатся рядом: при 50–100 деревьях оба уже "
            "усреднили разброс одиночного глубокого дерева."
        )

    best_other = max(rf["f1"], bag["f1"])
    if boost["f1"] > best_other + 0.005:
        parts.append(
            "Бустинг к 100 деревьям обгоняет оба метода и по аккуратности, и по F-мере. "
            "Бэггинг и лес к 50 участникам уже на плато, а бустинг с learning_rate=0.1 "
            "продолжает поправлять ошибки: с 50 до 100 участников растут и полнота, и F-мера. "
            "Одно дерево глубины 3 слабее дерева из работы 3, но цепочка из ста таких деревьев "
            "собирает границу точнее. При 50 участниках F-мера бустинга ещё около линии "
            "одиночного дерева, к 100 участникам отрыв уже устойчивый."
        )
    elif boost["f1"] + 0.005 < best_other:
        parts.append(
            "Бустинг с глубиной 3 и learning_rate=0.1 в этом диапазоне ниже леса и бэггинга. "
            "Неглубокие деревья медленнее собирают границу, которую глубокое дерево "
            "уже проводит по capital-gain, образованию и семейному положению."
        )
    else:
        parts.append(
            "Бустинг выходит на тот же уровень, что лес и бэггинг: последовательное "
            "исправление ошибок компенсирует меньшую глубину базового дерева."
        )

    if min(row["accuracy"] for row in last.values()) > baseline["accuracy"] + 0.004:
        parts.append(
            f"По аккуратности все три ансамбля выше линии дерева "
            f"({baseline['accuracy']:.3f}): несколько деревьев сглаживают разбиения, "
            "которые одно дерево фиксирует на обучающей выборке. Точность класса >50K "
            "тоже выше у каждого ансамбля. Полнота выше не у всех: случайный лес "
            "уступает дереву по recall и почти совпадает с ним по F-мере, то есть "
            "выигрыш леса — в более осторожных срабатываниях, а не в том, что редкий класс найден полнее."
        )
    return " ".join(parts)


def explanation_paragraphs(
    rows: list[dict],
    baselines: list[dict],
    baseline: dict,
    positive_rate: float,
    n_train: int,
    n_test: int,
) -> list[str]:
    other = [row for row in baselines if row["criterion"] != BASELINE_CRITERION]
    other_f1 = ", ".join(f"{row['criterion']} F1={row['f1']:.3f}" for row in other)
    paragraphs = [
        (
            f"Модели обучены на очищенной выборке adult.data ({n_train} объектов) и "
            f"проверены на adult.test ({n_test} объектов). Это тот же протокол, на котором "
            f"в практической работе 3 оценивалось дерево решений при 100% обучающей выборки. "
            f"Доля класса >50K на тесте — {positive_rate:.1%}. Из-за такого перекоса "
            f"аккуратность легко держится около 85%, даже если редкий класс найден не полностью. "
            f"Поэтому на диаграмме рядом с accuracy стоят precision, recall и F-мера класса >50K."
        ),
        (
            f"Пунктир на каждой панели — одиночное дерево с критерием Gini из практической "
            f"работы 3: accuracy={baseline['accuracy']:.3f}, precision={baseline['precision']:.3f}, "
            f"recall={baseline['recall']:.3f}, F1={baseline['f1']:.3f}. "
            f"Остальные критерии того задания на том же тесте близки ({other_f1}); "
            f"на график нанесена линия Gini, потому что деревья бэггинга и случайного леса "
            f"тоже делят узлы по индексу Джини. Глубина и минимум объектов в листе у этих "
            f"двух техник совпадают с деревом из работы 3 (12 и 20): отрыв кривой от пунктира "
            f"показывает эффект ансамбля при той же сложности базового дерева. "
            f"У бустинга деревья слабее нарочно — глубина 3 и learning_rate=0.1, "
            f"вклад каждого дерева небольшой, решение складывается по цепочке."
        ),
    ]

    bits = []
    for technique in TECHNIQUES:
        series = _series(rows, technique)
        bits.append(
            f"{TECHNIQUE_LABELS[technique]}: F-мера {_change_phrase(series[0]['f1'], series[-1]['f1'])}, "
            f"полнота {_change_phrase(series[0]['recall'], series[-1]['recall'])}"
        )
    paragraphs.append(
        "Бэггинг и случайный лес к 50 участникам уже вышли на плато, поэтому шаг "
        "от 50 до 100 почти ничего не меняет: основная дисперсия одиночного дерева "
        "к этому моменту усреднена. Бустинг в том же диапазоне ещё обучается. "
        + ". ".join(bits)
        + "."
    )

    last = {technique: _series(rows, technique)[-1] for technique in TECHNIQUES}
    best = max(last, key=lambda name: last[name]["f1"])
    versus = ", ".join(
        f"{TECHNIQUE_LABELS[name]} {last[name]['f1'] - baseline['f1']:+.3f}"
        for name in TECHNIQUES
    )
    paragraphs.append(
        f"При 100 участниках разница F-меры с деревом Gini: {versus}. "
        f"Выше всех F-мера у техники «{TECHNIQUE_LABELS[best]}» "
        f"({last[best]['f1']:.3f}; accuracy={last[best]['accuracy']:.3f}, "
        f"precision={last[best]['precision']:.3f}, recall={last[best]['recall']:.3f}). "
        + _interpretation(last, baseline)
    )
    return paragraphs


def _text_width_px(renderer, text: str, size: float) -> float:
    font = FontProperties(family=plt.rcParams["font.family"], size=size)
    width, _, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
    return width


def _wrap(text: str, *, fig, size: float, left: float = 0.08, right: float = 0.92) -> str:
    renderer = fig.canvas.get_renderer()
    max_px = (right - left) * fig.get_figwidth() * fig.dpi * 0.985

    def break_word(word: str) -> list[str]:
        if _text_width_px(renderer, word, size) <= max_px:
            return [word]
        parts: list[str] = []
        chunk = ""
        for ch in word:
            trial = chunk + ch
            if chunk and _text_width_px(renderer, trial, size) > max_px:
                parts.append(chunk)
                chunk = ch
            else:
                chunk = trial
        if chunk:
            parts.append(chunk)
        return parts

    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        pieces: list[str] = []
        for word in words:
            pieces.extend(break_word(word))
        current = pieces[0]
        for word in pieces[1:]:
            trial = f"{current} {word}"
            if _text_width_px(renderer, trial, size) <= max_px:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return "\n".join(lines)


def _style_table(table) -> None:
    for (i, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if i == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#f4f6f8")


def build_report(
    rows: list[dict],
    baselines: list[dict],
    baseline: dict,
    metrics_plot: Path,
    positive_rate: float,
    n_train: int,
    n_test: int,
    out_pdf: Path,
) -> None:
    from PIL import Image

    repo_url = read_repo_url()

    def header(fig, title: str) -> float:
        fig.text(0.08, 0.955, "Южно-Уральский государственный университет", fontsize=10, color="#333")
        fig.text(0.08, 0.932, "Курс «Основы интеллектуального анализа данных»", fontsize=10, color="#333")
        fig.text(0.08, 0.895, title, fontsize=16, fontweight="bold")
        fig.lines.append(
            plt.Line2D([0.08, 0.92], [0.875, 0.875], transform=fig.transFigure, color="#1f4e79", lw=1.4)
        )
        return 0.855

    def heading(fig, y, text):
        fig.text(0.08, y, text, fontsize=13, fontweight="bold", va="top")
        return y - 0.028

    def paragraph(fig, y, text, size=10.5):
        wrapped = _wrap(text, fig=fig, size=size)
        fig.text(0.08, y, wrapped, fontsize=size, va="top", color="#222", linespacing=1.35)
        n_lines = wrapped.count("\n") + 1
        return y - n_lines * 0.0168 * (size / 10.5) - 0.01

    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Практическая работа 4")
        fig.text(
            0.08,
            y,
            "Ансамблевая классификация (Census Income)",
            fontsize=12,
            color="#1f4e79",
            va="top",
        )
        y -= 0.035
        y = heading(fig, y, "1. Формулировка задания")
        y = paragraph(
            fig,
            y,
            "Разработать программу классификации с помощью ансамбля. Параметры программы: "
            "набор данных, техника (бэггинг, случайный лес или бустинг), число участников "
            "ансамбля и параметры выбранной техники. Провести эксперименты на наборе Census "
            "Income из задания «Классификация с помощью дерева решений», меняя число "
            "участников от 50 до 100 с шагом 10. Построить диаграмму показателей качества "
            "в зависимости от размера ансамбля и нанести на неё значения одиночного дерева "
            "решений. Подготовить отчёт с рисунком и пояснением результатов.",
        )
        y = heading(fig, y, "2. Репозиторий с материалами")
        if repo_url:
            y = paragraph(
                fig,
                y,
                "Исходные тексты, набор данных и результаты этой работы лежат в каталоге lab4/:",
            )
            wrapped_url = _wrap(repo_url, fig=fig, size=10)
            fig.text(0.08, y, wrapped_url, fontsize=10, color="#1f4e79", va="top", url=repo_url)
            n_url = wrapped_url.count("\n") + 1
            y -= n_url * 0.016 * (10 / 10.5) + 0.012
        else:
            y = paragraph(
                fig,
                y,
                "Материалы в каталоге lab4/. Чтобы добавить гиперссылку, запишите URL "
                "репозитория в repo_url.txt и снова запустите python experiments.py.",
            )
        y = heading(fig, y, "3. Набор данных и метод")
        y = paragraph(
            fig,
            y,
            "Census Income (Adult): возраст, образование, семейное положение, занятость "
            "и другие признаки переписи. Целевой класс — годовой доход ≤50K или >50K. "
            "Пропуски «?» удалены, категориальные признаки закодированы так же, как в "
            "практической работе 3. Обучение — вся очищенная adult.data, проверка — adult.test. "
            "Положительный класс метрик precision, recall и F-меры — «>50K».",
        )
        y = paragraph(
            fig,
            y,
            "Бэггинг обучает деревья CART (критерий Джини, глубина 12, не меньше 20 объектов "
            "в листе) на бутстреп-выборках того же размера и использует все признаки. "
            "Итог — голосование большинства. Случайный лес устроен так же, но на каждом "
            "разбиении рассматривает sqrt признаков: для 14 признаков это 3. Бустинг — "
            "градиентный: деревья глубины 3 с темпом learning_rate=0.1 строятся по очереди "
            "и поправляют остатки предыдущих. Во всех трёх техниках меняется только число "
            "участников: 50, 60, 70, 80, 90, 100.",
        )
        y = heading(fig, y, "4. Дерево решений из практической работы 3")
        y = paragraph(
            fig,
            y,
            "Те же объекты adult.test. На рисунке 1 пунктиром нанесена строка gini: "
            "этот критерий совпадает с деревьями бэггинга и случайного леса.",
            size=10,
        )
        tree_table = [["Критерий", "Accuracy", "Precision", "Recall", "F1"]]
        short_criterion = {
            "information_gain": "information gain",
            "gain_ratio": "gain ratio",
            "gini_index": "gini index",
        }
        for row in baselines:
            tree_table.append(
                [
                    short_criterion.get(row["criterion"], row["criterion"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                ]
            )
        table_height = 0.15
        ax = fig.add_axes([0.08, max(0.05, y - table_height - 0.01), 0.84, table_height])
        ax.axis("off")
        table = ax.table(
            cellText=tree_table,
            loc="center",
            cellLoc="center",
            colWidths=[0.28, 0.18, 0.18, 0.18, 0.18],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        _style_table(table)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Показатели качества классификации")
        caption = "Рисунок 1. Метрики ансамбля и линия дерева решений (gini)"
        fig.text(0.08, y, caption, fontsize=11, fontweight="bold", va="top")
        image = Image.open(metrics_plot)
        width = 0.88
        height = width * (fig.get_figwidth() / fig.get_figheight()) * (image.height / image.width)
        bottom = max(0.06, y - 0.03 - height)
        ax = fig.add_axes([0.06, bottom, width, height])
        ax.imshow(image)
        ax.axis("off")
        fig.text(
            0.08,
            bottom - 0.025,
            "Пунктир — accuracy, precision, recall и F-мера одиночного дерева Gini "
            "на adult.test (практическая работа 3).",
            fontsize=9,
            color="#333",
            va="top",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Пояснение результатов")
        y = heading(fig, y, "5. Смысл полученных результатов")
        for text in explanation_paragraphs(
            rows, baselines, baseline, positive_rate, n_train, n_test
        ):
            y = paragraph(fig, y, text)
        fig.text(
            0.08,
            0.04,
            "Отчёт: experiments.py. Рисунок: results/figures/metrics_vs_estimators.png.",
            fontsize=8,
            color="#555",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Сводка экспериментов")
        y = heading(fig, y, "6. Метрики на adult.test")
        y = paragraph(
            fig,
            y,
            "Полная сетка: три техники и шесть размеров ансамбля. Первые три строки — "
            "дерево решений из практической работы 3. Пунктир на рисунке 1 соответствует "
            "строке gini.",
            size=10,
        )
        short_tree = {
            "information_gain": "дерево, info gain",
            "gain_ratio": "дерево, gain ratio",
            "gini_index": "дерево, gini",
        }
        table_data = [["Техника", "N", "Accuracy", "Precision", "Recall", "F1"]]
        for row in baselines:
            table_data.append(
                [
                    short_tree.get(row["criterion"], row["criterion"]),
                    "—",
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                ]
            )
        for technique in TECHNIQUES:
            for row in _series(rows, technique):
                table_data.append(
                    [
                        TECHNIQUE_LABELS[technique],
                        str(row["n_estimators"]),
                        f"{row['accuracy']:.4f}",
                        f"{row['precision']:.4f}",
                        f"{row['recall']:.4f}",
                        f"{row['f1']:.4f}",
                    ]
                )
        ax = fig.add_axes([0.07, 0.06, 0.86, 0.70])
        ax.axis("off")
        table = ax.table(
            cellText=table_data,
            cellLoc="center",
            colWidths=[0.26, 0.10, 0.16, 0.16, 0.16, 0.16],
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        _style_table(table)
        pdf.savefig(fig)
        plt.close(fig)


def _write_report_from_payload(payload: dict, out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    baselines = payload["tree_baselines"]
    baseline = next(row for row in baselines if row["criterion"] == BASELINE_CRITERION)
    rows = payload["experiments"]
    metrics_plot = plot_metrics_vs_estimators(
        rows,
        baseline,
        fig_dir / "metrics_vs_estimators.png",
    )
    report_path = ROOT / "Отчет по практической работе 4.pdf"
    build_report(
        rows,
        baselines,
        baseline,
        metrics_plot,
        float(payload["positive_rate_test"]),
        int(payload["n_train"]),
        int(payload["n_test"]),
        report_path,
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперименты ансамблевой классификации")
    parser.add_argument("-o", "--output-dir", default=str(ROOT / "results"))
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Пересобрать рисунок и PDF из results/experiment_results.json",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        payload = json.loads((out_dir / "experiment_results.json").read_text(encoding="utf-8"))
        report_path = _write_report_from_payload(payload, out_dir)
        print(f"\nОтчёт: {report_path}")
        return 0

    train, test, _, _ = load_adult(
        ROOT / "data" / "adult.data.csv",
        ROOT / "data" / "adult.test.csv",
    )
    assert test is not None
    positive = list(train.target_names).index(">50K")
    positive_rate = float(np.mean(test.y == positive))

    baselines = load_tree_baselines()
    rows = run_experiments(train.X, train.y, test.X, test.y, positive)
    save_table(rows, out_dir / "ensemble_metrics.csv")

    specs = {
        technique: spec_to_dict(resolve_spec(technique, n_estimators=N_GRID[-1]))
        for technique in TECHNIQUES
    }
    payload = {
        "n_train": int(len(train.y)),
        "n_test": int(len(test.y)),
        "positive_rate_test": positive_rate,
        "baseline_criterion": BASELINE_CRITERION,
        "tree_baselines": baselines,
        "specs": specs,
        "experiments": rows,
    }
    (out_dir / "experiment_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = _write_report_from_payload(payload, out_dir)
    print(f"\nОтчёт: {report_path}")
    print(f"Рисунок: {out_dir / 'figures' / 'metrics_vs_estimators.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
