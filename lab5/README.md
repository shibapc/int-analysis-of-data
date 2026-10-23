# Практическая работа 5. Кластеризация (customers)

k-Means и k-Medoids на сведениях о клиентах банка.

## Запуск

```bash
cd lab5
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python main.py -k 6
python main.py -k 6 --noise 0.05
python experiments.py
python report.py
```

`main.py` печатает координаты объектов, номер кластера и ошибку обоих алгоритмов.
`experiments.py` считает сетку k = 3…9 и доли шума 1%, 3%, 5%, 10%, сохраняет рисунки.
`report.py` собирает отчёт DOCX по готовым результатам.

## Данные

| Файл | Описание |
|------|----------|
| `customers/customers.csv` | клиенты банка |
| `customers/custmers.txt` | описание признаков |

## Результаты

- `results/figures/` — точечные графики и кривые ошибки
- `Отчет по практической работе 5.docx` — отчёт
