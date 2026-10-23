"""Загрузка и стандартизация набора customers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Defaulted пропущен у 150 из 850 клиентов, поэтому в расстояние не входит.
FEATURES = [
    "Age",
    "Education",
    "YearsEmployed",
    "Income",
    "CardDebt",
    "OtherDebt",
    "DebtIncomeRatio",
]

FEATURE_LABELS = {
    "Age": "возраст, лет",
    "Education": "образование, код",
    "YearsEmployed": "стаж, лет",
    "Income": "доход, тыс. $",
    "CardDebt": "долг по карте",
    "OtherDebt": "прочие долги",
    "DebtIncomeRatio": "долг / доход, %",
}


@dataclass
class CustomersData:
    frame: pd.DataFrame
    X: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]


def load_customers(path: str | Path) -> CustomersData:
    frame = pd.read_csv(path)
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"В файле нет столбцов: {', '.join(missing)}")
    data = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Пропуски в признаках кластеризации: {', '.join(bad)}")
    values = data.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    scaled = (values - mean) / scale
    return CustomersData(
        frame=data.reset_index(drop=True),
        X=scaled,
        mean=mean,
        scale=scale,
        feature_names=list(FEATURES),
    )


def to_original(data: CustomersData, scaled: np.ndarray) -> np.ndarray:
    return scaled * data.scale + data.mean


def add_noise(
    X: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
    amplitude: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Изменить долю объектов: к 1–3 координатам прибавить U(−a, a).

    X ожидается уже стандартизованным, поэтому амплитуда измеряется
    в стандартных отклонениях исходных признаков.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Доля шума должна быть в [0, 1]")
    noisy = np.array(X, dtype=float, copy=True)
    n, n_features = noisy.shape
    count = int(round(n * fraction))
    if count == 0:
        return noisy, np.array([], dtype=int)
    chosen = rng.choice(n, size=count, replace=False)
    max_coords = min(3, n_features)
    for index in chosen:
        n_coords = int(rng.integers(1, max_coords + 1))
        columns = rng.choice(n_features, size=n_coords, replace=False)
        noisy[index, columns] += rng.uniform(-amplitude, amplitude, size=n_coords)
    return noisy, np.sort(chosen)
