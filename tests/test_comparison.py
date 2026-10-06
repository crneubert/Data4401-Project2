"""Read-only comparison tests; no real training, downloads, or dataset loading."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import comparison_utils as comparison
from test_preparation import cells, execute_cell

ROOT = Path(__file__).resolve().parents[1]


def record(model="nn", feature="application_only", timestamp="2026-10-05T01:00:00+00:00", auc=.75):
    return {
        "model": model, "feature_set": feature, "configuration": "starter_v1",
        "split_strategy": "stratified", "split_fingerprint": "a" * 64,
        "feature_schema_fingerprint": ("b" if feature == "application_only" else "c") * 64,
        "created_at_utc": timestamp, "source_file": model + feature + timestamp,
        "evaluation_partition": "validation", "holdout_evaluated": False,
        "rows": {"train": 20, "validation": 4}, "positive_rates": {"validation": .5},
        "metrics": {"roc_auc": auc, "average_precision": .6, "log_loss": .5,
                    "accuracy": .5, "f1": .5, "precision": .5, "recall": .5,
                    "threshold": .5, "confusion_matrix": {"tn": 1, "fp": 1, "fn": 1, "tp": 1}},
        "training_seconds": 60, "preparation_seconds": 2, "selected_iteration": 2,
        "training_history": [{"epoch": 1, "validation_roc_auc": .7}, {"epoch": 2, "validation_roc_auc": auc}],
    }


class ComparisonTests(unittest.TestCase):
    def test_latest_not_best_and_baseline_is_unpaired(self):
        older = record(auc=.9)
        newer = record(timestamp="2026-10-05T02:00:00+00:00", auc=.7)
        baseline = record(model="logistic_regression", auc=.74)
        baseline["training_seconds"] = None
        baseline["split_fingerprint"] = None
        baseline["feature_schema_fingerprint"] = None
        baseline["metrics"]["average_precision"] = None
        selected, split_hash = comparison.select_runs([older, newer, baseline])
        self.assertEqual(split_hash, "a" * 64)
        self.assertEqual(selected[0]["metrics"]["roc_auc"], .7)
        frame = comparison.results_frame(selected)
        self.assertEqual(frame.loc[frame.model == "logistic_regression", "comparison_group"].iloc[0],
                         "R baseline: unpaired reference")
        self.assertTrue(frame.loc[frame.model == "logistic_regression", "training_minutes"].isna().all())

    def test_cohort_schema_and_threshold_guards(self):
        first, other = record(), record(model="catboost")
        other["split_fingerprint"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "Multiple split"):
            comparison.select_runs([first, other])
        selected, _ = comparison.select_runs([first, other], split_hash="a" * 64)
        self.assertEqual(len(selected), 1)
        with self.assertRaisesRegex(ValueError, "no matching"):
            comparison.select_runs([first], split_hash="e" * 64)
        other["split_fingerprint"] = first["split_fingerprint"]
        other["feature_schema_fingerprint"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "schemas differ"):
            comparison.select_runs([first, other])
        other["feature_schema_fingerprint"] = first["feature_schema_fingerprint"]
        other["metrics"]["threshold"] = .3
        with self.assertRaisesRegex(ValueError, "Threshold"):
            comparison.select_runs([first, other])

    def test_history_gains_and_curve_formats(self):
        app, history = record(), record(feature="application_history", auc=.8)
        baseline = record(model="logistic_regression")
        gains = comparison.history_gains(comparison.results_frame([app, history, baseline]))
        self.assertEqual(len(gains), 1)
        self.assertAlmostEqual(gains.iloc[0].auc_gain, .05)
        self.assertEqual(comparison.validation_auc_history(app), ([1, 2], [.7, .75]))
        for model in ("lightgbm", "catboost"):
            tree = record(model=model)
            tree["training_history"] = {"validation": {"AUC": [.6, .7]}}
            self.assertEqual(comparison.validation_auc_history(tree), ([1, 2], [.6, .7]))
        self.assertEqual(comparison.validation_auc_history(baseline), ([], []))

    def test_missing_and_corrupt_records(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(comparison.load_run_records(directory), [])
            selected, split_hash = comparison.select_runs([])
            self.assertEqual(selected, [])
            self.assertIsNone(split_hash)
            self.assertTrue(comparison.coverage_frame([]).status.eq("missing export").all())
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(record()), encoding="utf-8")
            self.assertEqual(len(comparison.load_run_records(directory)), 1)
            for modification in (
                {"holdout_evaluated": True}, {"evaluation_partition": "holdout"},
                {"split_fingerprint": None}, {"created_at_utc": "bad"},
                {"rows": {"train": 20, "validation": 99}},
            ):
                broken = {**record(), **modification}
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.assertRaises(ValueError):
                    comparison.load_run_records(directory)

    def test_notebook_with_current_exports_and_no_results(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for run_records in (comparison.load_run_records(ROOT / "results" / "metrics"), []):
            namespace = {"__name__": "__main__"}
            with patch.object(comparison, "load_run_records", return_value=run_records):
                with patch.object(plt, "show"), contextlib.redirect_stdout(io.StringIO()):
                    for cell in cells("model_comparison.ipynb"):
                        execute_cell(namespace, cell)
                        if cell.get("id") == "dependencies":
                            namespace["display"] = lambda value: None
            selected, _ = comparison.select_runs(run_records)
            self.assertEqual(len(namespace["paired"]), len([r for r in selected if r["model"] in comparison.MODELS]))


if __name__ == "__main__":
    unittest.main()
