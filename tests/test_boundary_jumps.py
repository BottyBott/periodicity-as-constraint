from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pac import boundary_jump_result, evaluate_boundary_jumps, load_elements


class BoundaryJumpTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elements = load_elements()

    def test_first_ionization_shows_boundary_signal(self):
        result = boundary_jump_result(self.elements, "first_ionization_energy_ev")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(result.boundary_median, result.interior_median)
        self.assertLess(result.p_value, 0.05)

    def test_evaluate_boundary_jumps_returns_table(self):
        table, results = evaluate_boundary_jumps(
            self.elements,
            properties=["first_ionization_energy_ev", "pauling_en"],
        )
        self.assertFalse(table.empty)
        self.assertEqual(len(results), len(table))
        required_columns = {
            "property",
            "boundary_median",
            "interior_median",
            "median_difference",
            "median_ratio",
            "count_boundary",
            "count_interior",
            "u_statistic",
            "p_value",
            "cliffs_delta",
            "flagged_symbols",
        }
        self.assertTrue(required_columns.issubset(table.columns))

        first_row = table.iloc[0]
        self.assertGreater(first_row["cliffs_delta"], 0)


if __name__ == "__main__":
    unittest.main()
