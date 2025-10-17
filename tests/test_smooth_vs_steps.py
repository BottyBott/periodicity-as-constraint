from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pac import build_prediction_frame, evaluate_property, load_elements


class SmoothVsStepsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elements = load_elements()

    def test_constraint_model_beats_spline_for_ionization(self):
        result = evaluate_property(
            self.elements,
            "first_ionization_energy_ev",
            n_knots=10,
            spline_degree=3,
            n_splits=5,
            random_state=0,
        )
        self.assertIsNotNone(result, "Evaluation result should not be None")
        assert result is not None

        self.assertLess(
            result.constraint_bic,
            result.smooth_bic,
            "Constraint model should reduce BIC relative to spline(Z).",
        )

        self.assertLessEqual(
            result.constraint_mae_cv,
            result.smooth_mae_cv * 1.05,
            "Constraint model MAE should be comparable or better within 5%.",
        )

    def test_prediction_frame_contains_residuals(self):
        frame = build_prediction_frame(
            self.elements,
            "first_ionization_energy_ev",
            n_knots=10,
            spline_degree=3,
        )
        self.assertIsNotNone(frame, "Prediction frame should not be None")
        assert frame is not None

        for column in [
            "smooth_prediction",
            "constraint_prediction",
            "residual_smooth",
            "residual_constraint",
        ]:
            self.assertIn(column, frame.columns)
            self.assertFalse(frame[column].isna().any(), f"{column} should not contain NaNs")


if __name__ == "__main__":
    unittest.main()
