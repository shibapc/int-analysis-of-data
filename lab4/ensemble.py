"""Ансамблевая классификация: бэггинг, случайный лес, градиентный бустинг."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

Technique = str

TECHNIQUES = ("bagging", "random_forest", "boosting")

TECHNIQUE_ALIASES = {
    "bagging": "bagging",
    "бэггинг": "bagging",
    "bag": "bagging",
    "random_forest": "random_forest",
    "random-forest": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",
    "forest": "random_forest",
    "случайный_лес": "random_forest",
    "boosting": "boosting",
    "бустинг": "boosting",
    "gradient_boosting": "boosting",
    "gbdt": "boosting",
    "gb": "boosting",
}

TECHNIQUE_LABELS = {
    "bagging": "Бэггинг",
    "random_forest": "Случайный лес",
    "boosting": "Бустинг",
}


def normalize_technique(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in TECHNIQUE_ALIASES:
        raise ValueError("Техника: bagging | random_forest | boosting")
    return TECHNIQUE_ALIASES[key]


def parse_max_features(value: str | float | int | None):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"sqrt", "log2"}:
        return lowered
    if lowered in {"none", "all"}:
        return None if lowered == "none" else 1.0
    if any(ch in text for ch in ".eE"):
        return float(text)
    return int(text)


@dataclass
class EnsembleSpec:
    technique: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_features: str | float | int | None
    max_samples: float
    learning_rate: float
    random_state: int = 42
    n_jobs: int = -1

    def public_params(self) -> dict:
        """Параметры, которые реально относятся к выбранной технике."""
        common = {
            "technique": self.technique,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        if self.technique == "bagging":
            return {
                **common,
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": "gini",
            }
        if self.technique == "random_forest":
            return {
                **common,
                "max_features": self.max_features,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": "gini",
            }
        return {
            **common,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }


def resolve_spec(
    technique: str,
    n_estimators: int,
    *,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    max_features: str | float | int | None = None,
    max_samples: float | None = None,
    learning_rate: float | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> EnsembleSpec:
    technique = normalize_technique(technique)
    if n_estimators < 1:
        raise ValueError("n_estimators должен быть >= 1")

    defaults = {
        "bagging": dict(
            max_depth=12,
            min_samples_leaf=20,
            max_features=1.0,
            max_samples=1.0,
            learning_rate=0.1,
        ),
        "random_forest": dict(
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",
            max_samples=1.0,
            learning_rate=0.1,
        ),
        "boosting": dict(
            max_depth=3,
            min_samples_leaf=20,
            max_features=1.0,
            max_samples=1.0,
            learning_rate=0.1,
        ),
    }[technique]

    if max_depth is not None:
        defaults["max_depth"] = max_depth
    if min_samples_leaf is not None:
        defaults["min_samples_leaf"] = min_samples_leaf
    if max_features is not None:
        defaults["max_features"] = parse_max_features(max_features)
    if max_samples is not None:
        defaults["max_samples"] = max_samples
    if learning_rate is not None:
        defaults["learning_rate"] = learning_rate

    if not 0 < float(defaults["max_samples"]) <= 1:
        raise ValueError("max_samples должен быть в (0, 1]")
    if float(defaults["learning_rate"]) <= 0:
        raise ValueError("learning_rate должен быть > 0")

    return EnsembleSpec(
        technique=technique,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        **defaults,
    )


def build_model(spec: EnsembleSpec, *, warm_start: bool = False):
    """Собрать ансамбль. При warm_start=True fit() дообучает новые деревья."""
    if spec.technique == "bagging":
        base = DecisionTreeClassifier(
            criterion="gini",
            max_depth=spec.max_depth,
            min_samples_leaf=spec.min_samples_leaf,
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=spec.n_estimators,
            max_samples=spec.max_samples,
            max_features=spec.max_features,
            bootstrap=True,
            bootstrap_features=False,
            warm_start=warm_start,
            n_jobs=spec.n_jobs,
            random_state=spec.random_state,
        )
    if spec.technique == "random_forest":
        return RandomForestClassifier(
            n_estimators=spec.n_estimators,
            criterion="gini",
            max_depth=spec.max_depth,
            min_samples_leaf=spec.min_samples_leaf,
            max_features=spec.max_features,
            bootstrap=True,
            warm_start=warm_start,
            n_jobs=spec.n_jobs,
            random_state=spec.random_state,
        )
    return GradientBoostingClassifier(
        n_estimators=spec.n_estimators,
        learning_rate=spec.learning_rate,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        warm_start=warm_start,
        random_state=spec.random_state,
    )


def spec_to_dict(spec: EnsembleSpec) -> dict:
    data = asdict(spec)
    data["applied"] = spec.public_params()
    return data
