# Практическая работа 6. Иерархическая кластеризация (customers)

Дендрограммы Single linkage, Complete linkage, Group average и Ward на тех же клиентах банка, что в работе 5.

## Запуск

```bash
cd lab6
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python main.py -m ward
python main.py -m single
python experiments.py
python report.py
```

`main.py` строит дендрограмму для выбранной меры. `experiments.py` считает все четыре меры, разрезает дерево на 6 кластеров и сравнивает разбиение с k-Means и k-Medoids.

## Данные

| Файл | Описание |
|------|----------|
| `customers/customers.csv` | клиенты банка |
| `customers/custmers.txt` | описание признаков |

## Результаты

- `results/figures/dendrograms.png` — четыре дендрограммы
- `results/figures/scatter_k6.png` — разрез на 6 кластеров
- `Отчет по практической работе 6.pdf` — отчёт
