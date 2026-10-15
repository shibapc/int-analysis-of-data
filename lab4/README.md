# Практическая работа 4. Ансамблевая классификация (Census Income)

Классификация дохода ≤50K / >50K бэггингом, случайным лесом или градиентным бустингом.

## Техники

- `bagging` — бутстреп объектов, деревья глубины 12, все признаки
- `random_forest` — то же дерево и `max_features=sqrt`
- `boosting` — градиентный бустинг, глубина 3, `learning_rate=0.1`

Число участников задаётся параметром `-n`.

## Запуск

```bash
cd lab4
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python main.py -t random_forest -n 100
python main.py -t bagging -n 50 --max-samples 0.8
python main.py -t boosting -n 80 --learning-rate 0.05 --max-depth 3

python experiments.py
```

`experiments.py` строит модели для 50…100 участников с шагом 10, сохраняет график и PDF-отчёт.

## Данные

| Файл | Описание |
|------|----------|
| `data/adult.data.csv` | обучающая выборка |
| `data/adult.test.csv` | тестовая выборка |
| `data/adult.names (1).txt` | описание признаков |

Тот же набор, что в практической работе 3 (дерево решений).

## Результаты

- `results/figures/metrics_vs_estimators.png` — метрики и линия дерева решений
- `results/ensemble_metrics.csv` — таблица экспериментов
- `Отчет по практической работе 4.pdf` — отчёт
