from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pac import build_prediction_frame, evaluate_properties, load_elements

DEFAULT_PROPERTIES = (
    "first_ionization_energy_ev",
    "pauling_en",
    "electron_affinity_ev",
    "melting_point_k",
    "boiling_point_k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare smooth spline(Z) fits against group/period/block constraint models."
    )
    parser.add_argument(
        "--property",
        dest="properties",
        action="append",
        help="Property column to evaluate (can be repeated). Defaults to a curated set.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("reports/smooth_vs_steps_metrics.csv"),
        help="Path to write the metrics table (CSV).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/smooth_vs_steps.png"),
        help="Path to save the diagnostic figure.",
    )
    parser.add_argument(
        "--n-knots",
        type=int,
        default=12,
        help="Number of spline knots for the smooth model.",
    )
    parser.add_argument(
        "--spline-degree",
        type=int,
        default=3,
        help="Spline degree for the smooth model.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the CSV or figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after generating it.",
    )
    return parser.parse_args()


def create_diagnostic_figure(
    frame: pd.DataFrame,
    property_name: str,
    *,
    save_path: Path | None,
    show: bool,
) -> None:
    if frame is None or frame.empty:
        return

    spans = _compute_period_spans(frame)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 8), sharex=True)

    ax_top, ax_bottom = axes

    ax_top.scatter(frame["Z"], frame[property_name], color="#264653", s=25, label="Observed")
    ax_top.plot(
        frame["Z"],
        frame["smooth_prediction"],
        color="#2a9d8f",
        linewidth=1.6,
        label="Smooth spline(Z)",
    )
    ax_top.plot(
        frame["Z"],
        frame["constraint_prediction"],
        color="#b23a48",
        linewidth=1.4,
        label="Constraint (group/period/block)",
    )

    ax_top.set_ylabel(property_name)
    ax_top.set_title("Smooth vs. constraint models")
    ax_top.legend(loc="upper left")
    ax_top.grid(alpha=0.2)

    ax_bottom.axhline(0, color="#6c757d", linewidth=1.0, linestyle="--")
    ax_bottom.scatter(
        frame["Z"],
        frame["residual_smooth"],
        color="#2a9d8f",
        s=18,
        alpha=0.9,
        label="Residual: smooth",
    )
    ax_bottom.scatter(
        frame["Z"],
        frame["residual_constraint"],
        color="#b23a48",
        s=18,
        alpha=0.9,
        label="Residual: constraint",
    )
    ax_bottom.set_xlabel("Atomic number (Z)")
    ax_bottom.set_ylabel(f"Residual ({property_name})")
    ax_bottom.set_title("Residual structure (constraint residuals flatten the period signal)")
    ax_bottom.legend(loc="upper left")
    ax_bottom.grid(alpha=0.2)

    for axis in axes:
        for idx, span in spans.iterrows():
            start = span["min"] - 0.5
            end = span["max"] + 0.5
            axis.axvspan(start, end, color="#e9ecef", alpha=0.08 if idx % 2 else 0.12, zorder=-1)
        for boundary in spans["max"][:-1]:
            axis.axvline(boundary + 0.5, color="#adb5bd", linestyle=":", linewidth=0.9, zorder=-1)

    ax_top.set_xlim(frame["Z"].min() - 0.5, frame["Z"].max() + 0.5)

    fig.tight_layout()

    if save_path is not None:
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved diagnostic figure to {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def _compute_period_spans(df: pd.DataFrame) -> pd.DataFrame:
    if "period" not in df.columns:
        return pd.DataFrame(
            columns=["min", "max"],
        )
    spans = (
        df.dropna(subset=["period"])
        .groupby("period")["Z"]
        .agg(["min", "max"])
        .sort_index()
    )
    return spans


def main() -> None:
    args = parse_args()

    df = load_elements()
    properties = tuple(args.properties) if args.properties else DEFAULT_PROPERTIES

    results = evaluate_properties(
        df,
        properties,
        n_knots=args.n_knots,
        spline_degree=args.spline_degree,
        n_splits=args.cv_splits,
    )

    if results.empty:
        print("No properties produced valid evaluations.")
        return

    available = results["property"].tolist()
    ordered_properties = [prop for prop in properties if prop in available]
    if ordered_properties:
        results["property"] = pd.Categorical(
            results["property"],
            categories=ordered_properties,
            ordered=True,
        )
        results = results.sort_values("property").reset_index(drop=True)
        first_property = ordered_properties[0]
    else:
        results = results.sort_values("property").reset_index(drop=True)
        first_property = results.iloc[0]["property"]

    with pd.option_context("display.max_columns", None):
        print(results)

    if not args.no_save:
        table_path = args.table.resolve()
        table_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(table_path, index=False)
        print(f"Saved metrics table to {table_path}")

    # Build diagnostic figure for the first evaluated property
    prediction_frame = build_prediction_frame(
        df,
        first_property,
        n_knots=args.n_knots,
        spline_degree=args.spline_degree,
    )

    figure_path = None if args.no_save else args.figure
    create_diagnostic_figure(
        prediction_frame,
        first_property,
        save_path=figure_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
