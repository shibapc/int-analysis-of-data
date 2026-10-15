"""Эксперименты: деревья решений, метрики при разных долях train, PDF-отчёт."""

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

from dataset import load_adult, load_dataset_for_split, split_xy
from decision_tree import DecisionTreeClassifier
from metrics import classification_metrics
from visualize import save_tree_plot

CRITERIA = ("information_gain", "gain_ratio", "gini_index")
TRAIN_RATIOS = (0.6, 0.7, 0.8, 0.9)
# для требуемого эксперимента «фиксируем критерий» — основной ряд на графиках
FIXED_CRITERION = "gini_index"
MAX_DEPTH = 12
MIN_LEAF = 20
PLOT_DEPTH = 3
REPO_URL_FILE = ROOT / "repo_url.txt"

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


def run_full_train_trees(out_dir: Path) -> list[dict]:
    """100% adult.data → дерево; оценка на adult.test; рисунки деревьев."""
    train, test, _, _ = load_adult(
        ROOT / "data" / "adult.data.csv",
        ROOT / "data" / "adult.test.csv",
    )
    assert test is not None
    positive = list(train.target_names).index(">50K")
    rows = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for criterion in CRITERIA:
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_LEAF,
        )
        model.fit(train.X, train.y)
        pred = model.predict(test.X)
        metrics = classification_metrics(test.y, pred, positive=positive)
        plot_path = fig_dir / f"tree_{criterion}.png"
        save_tree_plot(
            model,
            train.feature_names,
            train.target_names,
            plot_path,
            title=f"Дерево решений: {criterion} (train=100% adult.data)",
            max_depth=PLOT_DEPTH,
        )
        row = {
            "criterion": criterion,
            "train_ratio": 1.0,
            "n_train": int(len(train.y)),
            "n_test": int(len(test.y)),
            "depth": model.get_depth(),
            "n_leaves": model.get_n_leaves(),
            "tree_plot": str(plot_path),
            **metrics,
        }
        rows.append(row)
        print(
            f"[100% train] {criterion}: acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} depth={model.get_depth()}"
        )
    return rows


def run_ratio_experiments(out_dir: Path) -> list[dict]:
    """Фиксированный критерий + все критерии × доли 60..90%."""
    data = load_dataset_for_split(ROOT / "data" / "adult.data.csv")
    positive = list(data.target_names).index(">50K")
    rows = []
    for criterion in CRITERIA:
        for ratio in TRAIN_RATIOS:
            X_train, X_test, y_train, y_test = split_xy(data, ratio)
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_LEAF,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = classification_metrics(y_test, pred, positive=positive)
            row = {
                "criterion": criterion,
                "train_ratio": ratio,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "depth": model.get_depth(),
                "n_leaves": model.get_n_leaves(),
                **metrics,
            }
            rows.append(row)
            print(
                f"[{ratio:.0%}/{1-ratio:.0%}] {criterion}: "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}"
            )
    return rows


def save_tables(full_rows: list[dict], ratio_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "full_train_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "criterion", "n_train", "n_test", "depth", "n_leaves",
                "accuracy", "precision", "recall", "f1",
            ],
        )
        writer.writeheader()
        for row in full_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with (out_dir / "ratio_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "criterion", "train_ratio", "n_train", "n_test", "depth", "n_leaves",
                "accuracy", "precision", "recall", "f1",
            ],
        )
        writer.writeheader()
        for row in ratio_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def plot_metrics(ratio_rows: list[dict], fig_dir: Path, criterion: str = FIXED_CRITERION) -> Path:
    rows = [r for r in ratio_rows if r["criterion"] == criterion]
    xs = [r["train_ratio"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    series = [
        ("accuracy", "Accuracy", "#1f4e79"),
        ("precision", "Precision", "#c26e4a"),
        ("recall", "Recall", "#6b9080"),
        ("f1", "F-мера", "#7b6b8a"),
    ]
    for key, label, color in series:
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label, color=color)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=color)
    ax.set_xlabel("Доля обучающей выборки, %")
    ax.set_ylabel("Значение метрики")
    ax.set_title(f"Качество классификации ({criterion})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}:{100-int(x)}" for x in xs])
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = fig_dir / f"metrics_{criterion}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_metrics_all_criteria(ratio_rows: list[dict], fig_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F-мера"),
    ]
    colors = {
        "information_gain": "#1f4e79",
        "gain_ratio": "#c26e4a",
        "gini_index": "#6b9080",
    }
    for ax, (key, title) in zip(axes.ravel(), metric_keys):
        for criterion in CRITERIA:
            rows = [r for r in ratio_rows if r["criterion"] == criterion]
            xs = [r["train_ratio"] * 100 for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, marker="o", label=criterion, color=colors[criterion], lw=2)
        ax.set_title(title)
        ax.set_xlabel("Доля train, %")
        ax.set_ylabel(title)
        ax.set_xticks([60, 70, 80, 90])
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Метрики при разных критериях и долях обучающей выборки", fontsize=13)
    fig.tight_layout()
    path = fig_dir / "metrics_all_criteria.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _split_label(train_ratio: float) -> str:
    """Доля train:test в процентах. int((1 - 0.8) * 100) даёт 19, не 20."""
    train_pct = int(round(train_ratio * 100))
    return f"{train_pct}:{100 - train_pct}"


def _text_width_px(renderer, text: str, size: float) -> float:
    font = FontProperties(family=plt.rcParams["font.family"], size=size)
    width, _, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
    return width


def _wrap(text: str, *, fig, size: float, left: float = 0.08, right: float = 0.92) -> str:
    """Перенос по ширине глифов, а не по числу символов.

    Фиксированная длина строки для DejaVu Serif вылезает за правое поле A4:
    кириллица и смесь с латиницей шире, чем предполагает textwrap.
    """
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


def build_report(
    full_rows: list[dict],
    ratio_rows: list[dict],
    metrics_plot: Path,
    metrics_all_plot: Path,
    out_pdf: Path,
) -> None:
    repo_url = read_repo_url()
    fig_dir = ROOT / "results" / "figures"

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
        n = wrapped.count("\n") + 1
        return y - n * 0.0168 * (size / 10.5) - 0.01

    import io
    from PIL import Image

    def embed(fig, image_path: Path, left, bottom, width, height):
        image = Image.open(image_path)
        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(image)
        ax.axis("off")

    with PdfPages(out_pdf) as pdf:
        # page 1
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Лабораторная работа № 2")
        fig.text(0.08, y, "Классификация деревом решений (Census Income)", fontsize=12, color="#1f4e79", va="top")
        y -= 0.035
        y = heading(fig, y, "1. Формулировка задания")
        y = paragraph(
            fig,
            y,
            "Разработать программу классификации с помощью дерева решений. Параметры: "
            "набор данных и критерий выбора атрибута разбиения (Information gain, "
            "Gain ratio, Gini index). Провести эксперименты на Census Income: сначала "
            "построить дерево на 100% обучающей выборки и визуализировать его. Затем "
            "добавить параметр доли обучающей выборки и метрики accuracy, precision, "
            "recall, F-мера; варьировать соотношение train:test от 60:40 до 90:10 "
            "с шагом 10%. Подготовить отчёт с рисунками и пояснениями.",
        )
        y = heading(fig, y, "2. Репозиторий с материалами")
        if repo_url:
            y = paragraph(fig, y, "Исходные тексты, данные и результаты:")
            wrapped_url = _wrap(repo_url, fig=fig, size=10)
            fig.text(0.08, y, wrapped_url, fontsize=10, color="#1f4e79", va="top", url=repo_url)
            n_url = wrapped_url.count("\n") + 1
            y -= n_url * 0.016 * (10 / 10.5) + 0.008
        else:
            y = paragraph(
                fig,
                y,
                "Материалы в каталоге lab2/. Чтобы добавить гиперссылку, запишите URL "
                "репозитория в repo_url.txt и снова запустите python experiments.py.",
            )
        y = heading(fig, y, "3. Набор данных и метод")
        y = paragraph(
            fig,
            y,
            "Census Income (Adult): признаки социально-демографического профиля "
            "(возраст, образование, семейное положение, занятость и др.), целевой "
            "класс — годовой доход ≤50K или >50K. Пропуски «?» удалены. Категориальные "
            "признаки кодируются порядковым кодированием; дерево строит бинарные "
            "разбиения «признак ≤ порог».",
        )
        y = paragraph(
            fig,
            y,
            "Критерии: Information gain — прирост информации по энтропии; "
            "Gain ratio — information gain, нормированный на split information "
            "(штраф за большое число ветвей); Gini index — уменьшение индекса Джини. "
            "Положительный класс для precision/recall/F1 — «>50K».",
        )
        y = heading(fig, y, "4. Метрики на 100% обучающей выборки (тест — adult.test)")
        table = [["Критерий", "Accuracy", "Precision", "Recall", "F1", "Глубина"]]
        for r in full_rows:
            table.append([
                r["criterion"],
                f"{r['accuracy']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                str(r["depth"]),
            ])
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.22])
        ax.axis("off")
        t = ax.table(cellText=table, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.4)
        for (i, _), cell in t.get_celld().items():
            cell.set_edgecolor("#d0d5dd")
            if i == 0:
                cell.set_facecolor("#1f4e79")
                cell.set_text_props(color="white", fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#f4f6f8")
        pdf.savefig(fig)
        plt.close(fig)

        # page 2 — trees
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Визуализация деревьев решений")
        fig.text(0.08, y, "Рисунок 1–3. Деревья для трёх критериев (глубина отображения 3)", fontsize=11, fontweight="bold", va="top")
        y -= 0.02
        positions = [
            (fig_dir / "tree_information_gain.png", 0.58),
            (fig_dir / "tree_gain_ratio.png", 0.33),
            (fig_dir / "tree_gini_index.png", 0.08),
        ]
        for path, bottom in positions:
            if path.exists():
                embed(fig, path, 0.06, bottom, 0.88, 0.23)
        pdf.savefig(fig)
        plt.close(fig)

        # page 3 — metrics plots
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Показатели качества классификации")
        fig.text(
            0.08,
            y,
            f"Рисунок 4. Метрики при фиксированном критерии {FIXED_CRITERION}",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        embed(fig, metrics_plot, 0.08, 0.52, 0.84, 0.28)
        fig.text(
            0.08,
            0.48,
            "Рисунок 5. Сравнение всех критериев",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        embed(fig, metrics_all_plot, 0.08, 0.08, 0.84, 0.36)
        pdf.savefig(fig)
        plt.close(fig)

        # page 4 — analysis
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        y = header(fig, "Пояснение результатов")
        y = heading(fig, y, "5. Смысл полученных результатов")
        best_full = max(full_rows, key=lambda r: r["f1"])
        fixed_rows = [r for r in ratio_rows if r["criterion"] == FIXED_CRITERION]
        y = paragraph(
            fig,
            y,
            f"На полной обучающей выборке adult.data (после очистки ~{full_rows[0]['n_train']} "
            f"объектов) и тесте adult.test все три критерия дают близкую аккуратность "
            f"около {full_rows[0]['accuracy']:.1%}–{full_rows[-1]['accuracy']:.1%}. "
            f"По F-мере лучше всего себя показал критерий {best_full['criterion']} "
            f"(F1={best_full['f1']:.3f}). Это согласуется с известными результатами "
            "для Adult: дерево решений уверенно отделяет доход >50K по образованию, "
            "семейному положению, capital-gain и часам работы.",
        )
        y = paragraph(
            fig,
            y,
            "Визуализация верхних уровней дерева показывает типичные разбиения: "
            "часто первыми идут capital-gain / marital-status / education-num — "
            "признаки, сильно связанные с доходом. Листья с доминирующим классом "
            "≤50K соответствуют более частой доле населения с доходом ниже порога.",
        )
        acc_line = ", ".join(
            f"{_split_label(r['train_ratio'])} → "
            f"acc={r['accuracy']:.3f}, F1={r['f1']:.3f}"
            for r in fixed_rows
        )
        y = paragraph(
            fig,
            y,
            f"При фиксированном критерии {FIXED_CRITERION} и изменении доли обучения "
            f"от 60% до 90%: {acc_line}. "
            "Рост доли обучения обычно слегка повышает устойчивость оценок, но на Adult "
            "разница невелика: выборка большая, стратифицированное разбиение сохраняет "
            "баланс классов. Precision и recall для класса >50K ниже accuracy — класс "
            "дохода выше 50K реже (~24%), поэтому ошибка «пропустить богатого» заметнее "
            "в recall, а ложные срабатывания бьют по precision.",
        )
        y = paragraph(
            fig,
            y,
            "Сравнение критериев на одних и тех же долях train показывает, что "
            "Information gain и Gini дают очень похожие кривые; Gain ratio иногда "
            "строит более компактные деревья за счёт штрафа за «широкие» разбиения. "
            "Для практики на Adult любой из трёх критериев приемлем; выбор влияет "
            "больше на форму дерева, чем на итоговую аккуратность.",
        )
        y = heading(fig, y, "6. Сводка по долям train (критерий " + FIXED_CRITERION + ")")
        table = [["Train:Test", "Accuracy", "Precision", "Recall", "F1"]]
        for r in fixed_rows:
            table.append([
                _split_label(r["train_ratio"]),
                f"{r['accuracy']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
            ])
        ax = fig.add_axes([0.08, max(0.08, y - 0.28), 0.84, 0.24])
        ax.axis("off")
        t = ax.table(cellText=table, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.45)
        for (i, _), cell in t.get_celld().items():
            cell.set_edgecolor("#d0d5dd")
            if i == 0:
                cell.set_facecolor("#1f4e79")
                cell.set_text_props(color="white", fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#f4f6f8")
        fig.text(
            0.08,
            0.04,
            "Отчёт: experiments.py. Рисунки: results/figures/.",
            fontsize=8,
            color="#555",
        )
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперименты дерева решений")
    parser.add_argument("-o", "--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full_rows = run_full_train_trees(out_dir)
    ratio_rows = run_ratio_experiments(out_dir)
    save_tables(full_rows, ratio_rows, out_dir)

    metrics_plot = plot_metrics(ratio_rows, fig_dir, FIXED_CRITERION)
    metrics_all = plot_metrics_all_criteria(ratio_rows, fig_dir)

    payload = {"full_train": full_rows, "ratio_experiments": ratio_rows}
    (out_dir / "experiment_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = ROOT / "report_lab2.pdf"
    build_report(full_rows, ratio_rows, metrics_plot, metrics_all, report_path)
    print(f"\nОтчёт: {report_path}")
    print(f"Рисунки: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
