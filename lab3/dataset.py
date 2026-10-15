"""Загрузка и подготовка Census Income (Adult)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

FEATURE_NAMES = NUMERIC + CATEGORICAL


@dataclass
class PreparedData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    n_raw: int
    n_clean: int


def _read_adult_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # adult.test начинается со строки-комментария
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    skiprows = 1 if first.startswith("|") else 0
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=skiprows,
    )
    df = df.dropna(how="all")
    return df


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace("?", np.nan)
    out = out.dropna().reset_index(drop=True)
    out["income"] = (
        out["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    )
    return out


def load_adult(
    train_path: str | Path,
    test_path: str | Path | None = None,
) -> tuple[PreparedData, PreparedData | None, OrdinalEncoder, LabelEncoder]:
    """Загрузить train (и опционально test), закодировать признаки одинаково."""
    train_df = _clean_frame(_read_adult_csv(train_path))
    test_df = _clean_frame(_read_adult_csv(test_path)) if test_path else None

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    label_encoder = LabelEncoder()

    def transform(df: pd.DataFrame, fit: bool) -> PreparedData:
        y_raw = df["income"].to_numpy()
        if fit:
            y = label_encoder.fit_transform(y_raw)
            cat = encoder.fit_transform(df[CATEGORICAL])
        else:
            y = label_encoder.transform(y_raw)
            cat = encoder.transform(df[CATEGORICAL])
        num = df[NUMERIC].to_numpy(dtype=float)
        X = np.hstack([num, cat])
        return PreparedData(
            X=X,
            y=y,
            feature_names=list(FEATURE_NAMES),
            target_names=list(label_encoder.classes_),
            n_raw=len(df),
            n_clean=len(df),
        )

    train = transform(train_df, fit=True)
    test = transform(test_df, fit=False) if test_df is not None else None
    # поправим n_raw — перечитаем без dropna для статистики при желании; достаточно clean
    return train, test, encoder, label_encoder


def load_dataset_for_split(train_path: str | Path) -> PreparedData:
    """Один набор (adult.data) для разбиения train/test по доле."""
    train, _, _, _ = load_adult(train_path, None)
    return train


def split_xy(
    data: PreparedData,
    train_ratio: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio должен быть в (0, 1)")
    return train_test_split(
        data.X,
        data.y,
        train_size=train_ratio,
        stratify=data.y,
        random_state=random_state,
    )
