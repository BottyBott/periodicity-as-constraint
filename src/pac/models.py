from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

CATEGORICAL_FEATURES: Sequence[str] = ("group", "period", "block")
DEFAULT_PROPERTIES: Sequence[str] = (
    "first_ionization_energy_ev",
    "pauling_en",
    "electron_affinity_ev",
    "melting_point_k",
    "boiling_point_k",
)


@dataclass
class EvaluationResult:
    property_name: str
    n_samples: int
    n_splits: int
    smooth_mae_cv: float
    smooth_bic: float
    constraint_mae_cv: float
    constraint_bic: float


def evaluate_property(
    df: pd.DataFrame,
    property_name: str,
    *,
    n_knots: int = 12,
    spline_degree: int = 3,
    n_splits: int = 5,
    random_state: int = 0,
) -> Optional[EvaluationResult]:
    """
    Compare spline(Z) vs. group/period/block models for a single property.
    Returns None when the property has too few non-null samples.
    """

    subset = _prepare_subset(df, property_name)
    n_samples = len(subset)
    if n_samples < max(8, spline_degree + 2):
        return None

    target = subset[property_name].to_numpy(dtype=float)

    smooth_features = subset[["Z"]].to_numpy(dtype=float)
    constraint_features = subset[list(CATEGORICAL_FEATURES)].astype("string").fillna("NA")

    n_knots = _resolve_n_knots(n_samples, n_knots, spline_degree)

    smooth_pipeline = Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=spline_degree,
                    include_bias=False,
                ),
            ),
            ("regressor", LinearRegression()),
        ]
    )
    smooth_pipeline.fit(smooth_features, target)
    smooth_design = smooth_pipeline.named_steps["spline"].transform(smooth_features)
    smooth_bic = _bic(target, smooth_pipeline.predict(smooth_features), smooth_design.shape[1] + 1)
    smooth_mae_cv = _cross_validated_mae(
        smooth_pipeline,
        smooth_features,
        target,
        n_splits=n_splits,
        random_state=random_state,
    )

    constraint_pipeline = Pipeline(
        [
            (
                "encoder",
                _make_one_hot_encoder(),
            ),
            ("regressor", LinearRegression()),
        ]
    )
    constraint_pipeline.fit(constraint_features, target)
    constraint_design = constraint_pipeline.named_steps["encoder"].transform(constraint_features)
    constraint_bic = _bic(
        target,
        constraint_pipeline.predict(constraint_features),
        constraint_design.shape[1] + 1,
    )
    constraint_mae_cv = _cross_validated_mae(
        constraint_pipeline,
        constraint_features,
        target,
        n_splits=n_splits,
        random_state=random_state,
    )

    used_splits = min(n_splits, n_samples) if n_samples >= 2 else 0

    return EvaluationResult(
        property_name=property_name,
        n_samples=n_samples,
        n_splits=used_splits,
        smooth_mae_cv=smooth_mae_cv,
        smooth_bic=smooth_bic,
        constraint_mae_cv=constraint_mae_cv,
        constraint_bic=constraint_bic,
    )


def evaluate_properties(
    df: pd.DataFrame,
    properties: Iterable[str] = DEFAULT_PROPERTIES,
    *,
    n_knots: int = 12,
    spline_degree: int = 3,
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Evaluate multiple properties and return the metrics as a DataFrame."""
    records: List[EvaluationResult] = []
    for prop in properties:
        result = evaluate_property(
            df,
            prop,
            n_knots=n_knots,
            spline_degree=spline_degree,
            n_splits=n_splits,
            random_state=random_state,
        )
        if result is not None:
            records.append(result)

    if not records:
        return pd.DataFrame(
            columns=[
                "property",
                "n_samples",
                "n_splits",
                "smooth_mae_cv",
                "smooth_bic",
                "constraint_mae_cv",
                "constraint_bic",
            ]
        )

    return pd.DataFrame(
        {
            "property": [r.property_name for r in records],
            "n_samples": [r.n_samples for r in records],
            "n_splits": [r.n_splits for r in records],
            "smooth_mae_cv": [r.smooth_mae_cv for r in records],
            "smooth_bic": [r.smooth_bic for r in records],
            "constraint_mae_cv": [r.constraint_mae_cv for r in records],
            "constraint_bic": [r.constraint_bic for r in records],
        }
    )


def build_prediction_frame(
    df: pd.DataFrame,
    property_name: str,
    *,
    n_knots: int = 12,
    spline_degree: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fit both models on the full data for a property and return a DataFrame with predictions.
    """

    subset = _prepare_subset(df, property_name)
    if subset.empty:
        return None

    values = subset[property_name].to_numpy(dtype=float)
    smooth_features = subset[["Z"]].to_numpy(dtype=float)
    constraint_features = subset[list(CATEGORICAL_FEATURES)].astype("string").fillna("NA")

    n_knots = _resolve_n_knots(len(subset), n_knots, spline_degree)

    smooth_pipeline = Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=spline_degree,
                    include_bias=False,
                ),
            ),
            ("regressor", LinearRegression()),
        ]
    )
    smooth_pipeline.fit(smooth_features, values)

    constraint_pipeline = Pipeline(
        [
            (
                "encoder",
                _make_one_hot_encoder(),
            ),
            ("regressor", LinearRegression()),
        ]
    )
    constraint_pipeline.fit(constraint_features, values)

    subset = subset.copy()
    subset["smooth_prediction"] = smooth_pipeline.predict(smooth_features)
    subset["constraint_prediction"] = constraint_pipeline.predict(constraint_features)
    subset["residual_smooth"] = subset[property_name] - subset["smooth_prediction"]
    subset["residual_constraint"] = subset[property_name] - subset["constraint_prediction"]

    return subset.sort_values("Z").reset_index(drop=True)


def _prepare_subset(df: pd.DataFrame, property_name: str) -> pd.DataFrame:
    columns = ["Z", property_name, *CATEGORICAL_FEATURES]
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Dataframe missing required columns: {sorted(missing)}")

    subset = df[columns].dropna(subset=[property_name]).copy()
    subset["Z"] = pd.to_numeric(subset["Z"], errors="coerce")
    subset = subset.dropna(subset=["Z"])
    subset["Z"] = subset["Z"].astype(int)
    subset = subset.sort_values("Z")
    return subset


def _resolve_n_knots(n_samples: int, requested: int, spline_degree: int) -> int:
    if n_samples <= spline_degree + 1:
        return spline_degree + 1

    max_reasonable = max(spline_degree + 1, n_samples // 2)
    n_knots = min(requested, max_reasonable)
    return max(spline_degree + 1, n_knots)


def _bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    n = y_true.size
    if n == 0:
        return np.nan
    residual = y_true - y_pred
    rss = np.sum(residual**2)
    rss = max(rss, np.finfo(float).eps)
    return n * np.log(rss / n) + n_params * np.log(n)


def _cross_validated_mae(
    pipeline: Pipeline,
    features: np.ndarray | pd.DataFrame,
    target: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> float:
    splits = min(n_splits, len(target))
    if splits < 2:
        return np.nan

    cv = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        pipeline,
        features,
        target,
        scoring="neg_mean_absolute_error",
        cv=cv,
        error_score="raise",
    )
    return float(-scores.mean())


def _make_one_hot_encoder() -> OneHotEncoder:
    parameters = inspect.signature(OneHotEncoder).parameters
    kwargs = {"handle_unknown": "ignore", "dtype": float}
    if "sparse_output" in parameters:
        kwargs["sparse_output"] = False  # type: ignore[assignment]
    else:
        kwargs["sparse"] = False  # type: ignore[assignment]
    return OneHotEncoder(**kwargs)
