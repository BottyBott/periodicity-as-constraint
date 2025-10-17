from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pac import load_elements


class LoaderValidationTestCase(unittest.TestCase):
    def test_load_elements_schema(self):
        df = load_elements()
        self.assertEqual(df["Z"].iloc[0], 1)
        self.assertGreaterEqual(len(df), 118)
        self.assertTrue(pd.api.types.is_integer_dtype(df["Z"]))
        self.assertTrue(pd.api.types.is_string_dtype(df["symbol"]))
        self.assertIn(("Cn", "boiling_point_k"), df.attrs["uncertainty_notes"])
        self.assertEqual(float(df.loc[df["symbol"] == "Cn", "boiling_point_k"].iloc[0]), 357.0)

    def test_loader_rejects_missing_columns(self):
        df = load_elements()
        tmp = Path(tempfile.mkdtemp()) / "bad.csv"
        df.drop(columns=["block"]).to_csv(tmp, index=False)

        with self.assertRaises(ValueError) as ctx:
            load_elements(tmp)
        self.assertIn("Missing expected columns", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
