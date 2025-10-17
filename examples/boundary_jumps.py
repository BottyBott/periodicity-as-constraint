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

from pac import boundary_jump_result, evaluate_boundary_jumps, load_elements

DEFAULT_PROPERTIES = (
    "first_ionization_energy_ev",
    "pauling_en",
    "electron_affinity_ev",
    "melting_point_k",
    "boiling_point_k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify boundary jump magnitudes vs. interior steps across element properties."
    )
    parser.add_argument(
        "--property",
        dest="properties",
        action="append",
        help="Property column to evaluate (can be repeated). Defaults to canonical property set.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("reports/boundary_jump_stats.csv"),
        help="Path to write the summary CSV table.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/boundary_jump_boxplot.png"),
        help="Path to save the boundary vs. interior comparison plot.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist CSV/figure outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after generating it.",
    )
    return parser.parse_args()


def create_boxplot(result, *, save_path: Path | None, show: bool) -> None:
    if result is None:
        return

    data = [
        result.boundary_values,
        result.interior_values,
    ]
    labels = ["Boundary jumps", "Interior steps"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#2a9d8f", "alpha": 0.5},
        medianprops={"color": "#264653", "linewidth": 2},
        whiskerprops={"color": "#264653"},
        capprops={"color": "#264653"},
    )

    ax.set_ylabel(f"|Δ property| ({result.property_name})")
    ax.set_title(
        f"Boundary jumps vs. interior steps for {result.property_name}\n"
        f"M-W p={result.p_value:.3e}, median ratio={result.median_ratio:.2f}"
    )
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()

    if save_path is not None:
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved boxplot to {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def create_effect_size_plot(
    table: pd.DataFrame,
    *,
    metric: str = "cliffs_delta",
    save_path: Path | None,
    show: bool,
) -> None:
    if table.empty or metric not in table.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(table))
    ax.bar(
        x,
        table[metric],
        color="#264653",
        alpha=0.8,
    )
    ax.axhline(0, color="#6c757d", linestyle="--", linewidth=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(table["property"], rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Effect sizes for boundary vs. interior differences")
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()

    if save_path is not None:
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved effect size plot to {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_elements()

    properties = tuple(args.properties) if args.properties else DEFAULT_PROPERTIES
    table, results = evaluate_boundary_jumps(df, properties)

    if table.empty:
        print("No boundary jump statistics could be computed. Check property availability.")
        return

    # order table by requested property order if provided
    available = table["property"].tolist()
    ordered_properties = [prop for prop in properties if prop in available]
    if ordered_properties:
        table["property"] = pd.Categorical(table["property"], categories=ordered_properties, ordered=True)
        table = table.sort_values("property").reset_index(drop=True)

    with pd.option_context("display.max_columns", None):
        print(table)

    if not args.no_save:
        csv_path = args.table.resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        print(f"Saved boundary jump metrics to {csv_path}")

    # Use the first requested property that produced results for the diagnostic plot
    first_property = ordered_properties[0] if ordered_properties else table.iloc[0]["property"]
    selected = next((res for res in results if res.property_name == first_property), None)

    figure_path = None if args.no_save else args.figure
    create_boxplot(selected, save_path=figure_path, show=args.show)

    effect_path = None
    if not args.no_save:
        effect_path = args.figure.with_name("boundary_effect_sizes.png")
    create_effect_size_plot(table, save_path=effect_path, show=False)


if __name__ == "__main__":
    main()
