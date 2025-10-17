from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pac import load_elements


def _compute_period_spans(df):
    spans = (
        df.dropna(subset=["period"])
        .groupby("period")["Z"]
        .agg(["min", "max"])
        .sort_index()
    )
    return spans


def create_staircase_axes(df):
    subset = (
        df[["Z", "first_ionization_energy_ev", "period", "symbol"]]
        .dropna(subset=["first_ionization_energy_ev"])
        .sort_values("Z")
    )

    spans = _compute_period_spans(df)
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.step(
        subset["Z"],
        subset["first_ionization_energy_ev"],
        where="mid",
        color="#264653",
        linewidth=1.6,
        label="First ionization energy",
    )
    axis.scatter(
        subset["Z"],
        subset["first_ionization_energy_ev"],
        s=24,
        color="#2a9d8f",
        edgecolor="white",
        linewidth=0.4,
    )

    y_max = subset["first_ionization_energy_ev"].max()
    for idx, span in spans.iterrows():
        start = span["min"] - 0.5
        end = span["max"] + 0.5
        shade_alpha = 0.08 if idx % 2 else 0.12
        axis.axvspan(start, end, color="#e9ecef", alpha=shade_alpha)
        axis.text(
            (start + end) / 2,
            y_max + 0.3,
            f"Period {idx}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#495057",
        )

    for boundary in spans["max"][:-1]:
        axis.axvline(boundary + 0.5, color="#b23a48", linestyle="--", linewidth=1.0)

    axis.set_xlabel("Atomic number (Z)")
    axis.set_ylabel("First ionization energy (eV)")
    axis.set_title(
        "First ionization energy shows period resets\n"
        "(vertical lines mark transitions between periods)"
    )
    axis.set_xlim(
        subset["Z"].min() - 0.5,
        subset["Z"].max() + 0.5,
    )
    axis.set_ylim(0, y_max + 1.2)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")

    return figure, axis


def main(output: Path | None, show: bool) -> None:
    df = load_elements()
    figure, axis = create_staircase_axes(df)

    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved staircase plot to {output}")

    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot first ionization energy versus atomic number with period boundaries."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/staircase_ionization.png"),
        help="Where to write the PNG figure (default: %(default)s).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the PNG file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window in addition to saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    output_path = None if options.no_save else options.output
    main(output_path, show=options.show)
