# periodicity-as-constraint

Elemental properties follow constraint-driven resets rather than a single smooth march with atomic number. This repository operationalizes that claim with a minimal dataset and a handful of decisive analyses that contrast smooth (f(Z)) fits against group/period/block structure.

## Thesis
Electron-shell constraints induce step-like regularities—period boundaries, group families, and block assignments—that outperform any globally smooth mapping from atomic number to properties. The project demonstrates that ordering through constraints, not scalar gradualism, is the predictive grammar of the periodic table.

## Repository layout
The outline below tracks the build-out path. Files checked off are present; unchecked items are planned next.

- [x] `README.md`
- [x] `data/elements.csv`
- [x] `src/pac/__init__.py`
- [x] `src/pac/load.py`
- [x] `src/pac/models.py`
- [ ] `src/pac/figures.py`
- [ ] `src/pac/tests_gradualism.py`
- [x] `examples/staircase_ionization.py`
- [x] `examples/smooth_vs_steps.py`
- [x] `examples/boundary_jumps.py`
- [x] `tests/test_boundary_jumps.py`
- [ ] `tests/test_group_period_gain.py`
- [ ] `tests/test_nosignal_smooth.py`

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib scikit-learn
python3 examples/staircase_ionization.py         # staircase plot with period boundaries (writes figures/staircase_ionization.png)
python3 examples/smooth_vs_steps.py              # smooth vs constraint comparison (metrics + diagnostic figure)
python3 examples/boundary_jumps.py               # boundary jump stats (CSV + boxplot + effect sizes)
```

The first three examples are ready today; the remaining modules will appear as the modelling and test harness firm up. Meanwhile, `data/elements.csv` is ready for exploration in notebooks or ad-hoc scripts.

## Data: `data/elements.csv`
One tidy table (118 rows) provides the only required input for the analyses. Columns and units:

| column | description | units/source |
| --- | --- | --- |
| `Z` | atomic number | integer (1–118) |
| `symbol` | element symbol | string |
| `name` | element name | string |
| `group` | IUPAC group; `""` for missing f-block entries | integer or blank |
| `period` | IUPAC period index | integer |
| `block` | block label (`s`, `p`, `d`, `f`) | string |
| `first_ionization_energy_ev` | first ionization energy | electron-volts (converted from kJ/mol) |
| `pauling_en` | Pauling electronegativity | unitless |
| `electron_affinity_ev` | electron affinity | electron-volts (converted from kJ/mol) |
| `melting_point_k` | melting point | Kelvin |
| `boiling_point_k` | boiling point | Kelvin |

Conversion note: values in kJ/mol from the source feed were divided by 96.485 to express them in electron-volts.

Additional adjustments and conventions:
- Lawrencium (`Z=103`) first ionization energy is set to `4.963 eV`, matching the 2015 laser spectroscopy result reported by Johannes Gutenberg University Mainz.
- Copernicium (`Z=112`) adopts the Los Alamos National Laboratory adsorption-derived boiling point of `357 K` with an uncertainty of `357−108+112 K`, replacing the upstream `3570 K` placeholder.
- Helium’s listed melting point (`0.95 K`) refers to measurements under ~2–3 MPa; at 1 atm helium remains liquid. Retained to stay consistent with high-pressure compilations.
- Noble-gas electron affinities are recorded as small negative values to denote metastable anions; some tables omit these entries entirely.
- Thermodynamic entries for transactinides (e.g., Fl, Cn, Og) and Hassium’s melting point are predicted values; treat them as high-uncertainty estimates when modeling.

Running `python examples/staircase_ionization.py` saves `figures/staircase_ionization.png`, highlighting the repeated resets in first ionization energy across period boundaries.

Running `python examples/smooth_vs_steps.py` prints cross-validated MAE/BIC metrics contrasting spline(Z) and constraint models, writes `reports/smooth_vs_steps_metrics.csv`, and saves `figures/smooth_vs_steps.png`.

Running `python examples/boundary_jumps.py` reports Mann–Whitney U statistics comparing period-boundary jumps vs. interior steps, writes `reports/boundary_jump_stats.csv`, and saves both `figures/boundary_jump_boxplot.png` and `figures/boundary_effect_sizes.png` (the latter summarises effect sizes per property).

Loader metadata exposes per-value uncertainty flags via `df.attrs["value_flags"]`, and downstream summaries automatically surface the affected symbols (e.g., predicted transactinide thermals).

### Source
The CSV is derived from the [Bowserinator/Periodic-Table-JSON](https://github.com/Bowserinator/Periodic-Table-JSON) dataset (retrieved 2025-10-17). The upstream project lists the data as free to use; retain attribution if you redistribute this derivative.

## Planned analyses
- **Staircase ionization** — implemented in `examples/staircase_ionization.py`, highlighting discontinuities with period boundary markers.
- **Smooth vs. constraint models** — implemented in `examples/smooth_vs_steps.py`, contrasting spline(Z) fits with group/period/block regression.
- **Boundary jump tests** — implemented in `examples/boundary_jumps.py`, Mann–Whitney contrasts of boundary vs. interior step magnitudes.
- **Context-driven variance gain** — quantify predictive lift from `group`, `period`, and `block` over atomic number alone.
- **Changepoint sanity checks** — simple segmentation confirming that detected changepoints align with known period resets.


## Tests
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib scikit-learn
python3 -m unittest discover -s tests
```
Regression tests assert that constraint models match or beat spline(Z) on first ionization energy and that prediction frames expose the expected diagnostics.
