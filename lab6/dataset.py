"""Загрузка customers — те же признаки и стандартизация, что в работе 5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "Age",
    "Education",
    "YearsEmployed",
    "Income",
    "CardDebt",
    "OtherDebt",
    "DebtIncomeRatio",
]


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
