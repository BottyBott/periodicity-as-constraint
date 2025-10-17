from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELEMENTS_CSV = _ROOT / "data" / "elements.csv"

_REQUIRED_COLUMNS: Iterable[str] = (
    "Z",
    "symbol",
    "name",
    "group",
    "period",
    "block",
    "first_ionization_energy_ev",
    "pauling_en",
    "electron_affinity_ev",
    "melting_point_k",
    "boiling_point_k",
)

_FLOAT_COLUMNS = (
    "first_ionization_energy_ev",
    "pauling_en",
    "electron_affinity_ev",
    "melting_point_k",
    "boiling_point_k",
)


def load_elements(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical elements table with light validation."""
    csv_path = Path(path) if path is not None else DEFAULT_ELEMENTS_CSV
    df = pd.read_csv(csv_path)

    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df = df.copy()
    df["Z"] = pd.to_numeric(df["Z"], errors="raise").astype(int)

    for col in ("group", "period"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in _FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Z").reset_index(drop=True)

    if df["Z"].isna().any() or df["Z"].duplicated().any():
        raise ValueError("Atomic numbers must be unique and non-null.")

    return df
