from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELEMENTS_CSV = _ROOT / "data" / "elements.csv"

BLOCK_VALUES = {"s", "p", "d", "f"}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    allow_null: bool = False
    allowed_values: set[str] | None = None
    min_value: float | None = None
    max_value: float | None = None


_COLUMN_SPECS: Iterable[ColumnSpec] = (
    ColumnSpec("Z", "int", allow_null=False, min_value=1),
    ColumnSpec("symbol", "string", allow_null=False),
    ColumnSpec("name", "string", allow_null=False),
    ColumnSpec("group", "int_nullable", allow_null=True, min_value=1, max_value=18),
    ColumnSpec("period", "int", allow_null=False, min_value=1, max_value=9),
    ColumnSpec("block", "string", allow_null=False, allowed_values=BLOCK_VALUES),
    ColumnSpec("first_ionization_energy_ev", "float", allow_null=True, min_value=0.0),
    ColumnSpec("pauling_en", "float", allow_null=True, min_value=0.0),
    ColumnSpec("electron_affinity_ev", "float", allow_null=True),
    ColumnSpec("melting_point_k", "float", allow_null=True, min_value=0.0),
    ColumnSpec("boiling_point_k", "float", allow_null=True, min_value=0.0),
)

UNCERTAINTY_NOTES: Mapping[tuple[str, str], str] = {
    ("He", "melting_point_k"): "Helium only solidifies under pressure (~2–3 MPa).",
    ("Cn", "boiling_point_k"): "Adsorption-derived estimate; 357−108+112 K (Los Alamos).",
    ("Fl", "melting_point_k"): "Predicted value from transactinide models.",
    ("Fl", "boiling_point_k"): "Predicted value from transactinide models.",
    ("Og", "boiling_point_k"): "Predicted from relativistic calculations; high uncertainty.",
    ("Hs", "melting_point_k"): "Predicted melting point from theoretical studies.",
}


def load_elements(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical elements table with schema and uncertainty validation."""
    csv_path = Path(path) if path is not None else DEFAULT_ELEMENTS_CSV
    df = pd.read_csv(csv_path)

    expected_columns = {spec.name for spec in _COLUMN_SPECS}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df = df.copy()
    for spec in _COLUMN_SPECS:
        if spec.dtype == "int":
            df[spec.name] = pd.to_numeric(df[spec.name], errors="raise").astype(int)
        elif spec.dtype == "int_nullable":
            df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce").astype("Int64")
        elif spec.dtype == "float":
            df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce")
        elif spec.dtype == "string":
            df[spec.name] = df[spec.name].astype("string").str.strip()
        else:
            raise ValueError(f"Unsupported dtype specifier: {spec.dtype}")

        if not spec.allow_null and df[spec.name].isna().any():
            raise ValueError(f"Column '{spec.name}' contains null values but allow_null is False.")

        if spec.allowed_values is not None:
            invalid_mask = ~df[spec.name].isna() & ~df[spec.name].isin(spec.allowed_values)
            if invalid_mask.any():
                bad_values = sorted(df.loc[invalid_mask, spec.name].unique())
                raise ValueError(f"Column '{spec.name}' contains invalid values: {bad_values}")

        if spec.min_value is not None:
            below = df[spec.name].dropna() < spec.min_value
            if below.any():
                bad_rows = df.loc[below.index[below], ["Z", spec.name]]
                raise ValueError(
                    f"Column '{spec.name}' has values below {spec.min_value}: {bad_rows.to_dict('records')}"
                )

        if spec.max_value is not None:
            above = df[spec.name].dropna() > spec.max_value
            if above.any():
                bad_rows = df.loc[above.index[above], ["Z", spec.name]]
                raise ValueError(
                    f"Column '{spec.name}' has values above {spec.max_value}: {bad_rows.to_dict('records')}"
                )

    df = df.sort_values("Z").reset_index(drop=True)

    if df["Z"].duplicated().any():
        dupes = df.loc[df["Z"].duplicated(), "Z"].tolist()
        raise ValueError(f"Atomic numbers must be unique; duplicates found: {dupes}")

    if df["symbol"].duplicated().any():
        dupes = df.loc[df["symbol"].duplicated(), "symbol"].tolist()
        raise ValueError(f"Element symbols must be unique; duplicates found: {dupes}")

    if not df["Z"].is_monotonic_increasing:
        raise ValueError("Atomic numbers are not strictly increasing after sorting.")

    _validate_uncertainties(df)

    df.attrs["schema_version"] = 1
    df.attrs["uncertainty_notes"] = UNCERTAINTY_NOTES
    df.attrs["column_specs"] = _COLUMN_SPECS

    return df


def _validate_uncertainties(df: pd.DataFrame) -> None:
    for (symbol, column), note in UNCERTAINTY_NOTES.items():
        mask = df["symbol"] == symbol
        if not mask.any():
            raise ValueError(f"Expected uncertainty note for symbol '{symbol}' but no such row was found.")
        value = df.loc[mask, column]
        if value.empty or value.isna().all():
            raise ValueError(
                f"Uncertainty note '{note}' expects column '{column}' to have a value for symbol '{symbol}'."
            )
