from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DEFAULT_PROPERTIES: Sequence[str] = (
    "first_ionization_energy_ev",
    "pauling_en",
    "electron_affinity_ev",
    "melting_point_k",
    "boiling_point_k",
)  # mirror pac.models defaults


@dataclass
class BoundaryJumpResult:
    property_name: str
    boundary_values: np.ndarray
    interior_values: np.ndarray
    boundary_median: float
    interior_median: float
    u_statistic: float
    p_value: float

    @property
    def count_boundary(self) -> int:
        return int(self.boundary_values.size)

    @property
    def count_interior(self) -> int:
        return int(self.interior_values.size)

    @property
    def median_ratio(self) -> float:
        if self.interior_median == 0:
            return np.inf
        return self.boundary_median / self.interior_median


def boundary_jump_result(df: pd.DataFrame, property_name: str) -> Optional[BoundaryJumpResult]:
    """Compute boundary jump statistics for a single property."""
    subset = (
        df[["Z", "period", property_name]]
        .dropna(subset=[property_name, "period"])
        .sort_values("Z")
        .reset_index(drop=True)
    )

    if subset.empty or len(subset) < 3:
        return None

    values = subset[property_name].to_numpy(dtype=float)
    periods = subset["period"].to_numpy(dtype=int)

    diffs = np.abs(np.diff(values))
    boundaries = periods[:-1] != periods[1:]
    interior_mask = ~boundaries

    boundary_values = diffs[boundaries]
    interior_values = diffs[interior_mask]

    if boundary_values.size == 0 or interior_values.size == 0:
        return None

    u_stat, p_value = mannwhitneyu(boundary_values, interior_values, alternative="greater")

    return BoundaryJumpResult(
        property_name=property_name,
        boundary_values=boundary_values,
        interior_values=interior_values,
        boundary_median=float(np.median(boundary_values)),
        interior_median=float(np.median(interior_values)),
        u_statistic=float(u_stat),
        p_value=float(p_value),
    )


def evaluate_boundary_jumps(
    df: pd.DataFrame,
    properties: Iterable[str] = DEFAULT_PROPERTIES,
) -> Tuple[pd.DataFrame, List[BoundaryJumpResult]]:
    """Evaluate boundary jumps across properties, returning a summary table and raw results."""
    records: List[BoundaryJumpResult] = []
    for prop in properties:
        result = boundary_jump_result(df, prop)
        if result is not None:
            records.append(result)

    if not records:
        return (
            pd.DataFrame(
                columns=[
                    "property",
                    "boundary_median",
                    "interior_median",
                    "median_ratio",
                    "count_boundary",
                    "count_interior",
                    "u_statistic",
                    "p_value",
                ]
            ),
            records,
        )

    table = pd.DataFrame(
        {
            "property": [r.property_name for r in records],
            "boundary_median": [r.boundary_median for r in records],
            "interior_median": [r.interior_median for r in records],
            "median_ratio": [r.median_ratio for r in records],
            "count_boundary": [r.count_boundary for r in records],
            "count_interior": [r.count_interior for r in records],
            "u_statistic": [r.u_statistic for r in records],
            "p_value": [r.p_value for r in records],
        }
    ).sort_values("property").reset_index(drop=True)

    return table, records


__all__ = ["BoundaryJumpResult", "boundary_jump_result", "evaluate_boundary_jumps"]
