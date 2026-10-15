# Лабораторная работа № 2. Дерево решений (Census Income)

Классификация дохода ≤50K / >50K деревом решений.

## Критерии разбиения

- `information_gain` — прирост информации (энтропия)
- `gain_ratio` — отношение прироста информации
- `gini_index` — индекс Джини

## Запуск

```bash
cd lab2

# дерево на 100% adult.data, оценка на adult.test
python main.py -c gini_index -r 1 --tree-plot results/figures/demo_tree.png

# доля обучающей выборки 80%
python main.py -c information_gain -r 0.8

# все эксперименты + PDF-отчёт
python experiments.py
```

## Данные

| Файл | Описание |
|------|----------|
| `data/adult.data.csv` | обучающая выборка |
| `data/adult.test.csv` | тестовая выборка |
| `data/adult.names.txt` | описание признаков |

## Результаты

- `results/figures/tree_*.png` — визуализация деревьев
- `results/figures/metrics_*.png` — графики метрик
- `results/ratio_metrics.csv` — таблица экспериментов
- `report_lab2.pdf` — отчёт
