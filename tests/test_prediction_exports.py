"""Synthetic applicant-level export checks; no data access or real training."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import experiment_utils as experiments


class PredictionExportTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([104, 101, 103, 102])  # Deliberately not sorted.
        self.labels = pd.Series([0, 0, 1, 1], index=self.ids)
        self.probabilities = np.array([.1, .6, .4, .9])
        self.audit = pd.DataFrame({
            "gender": ["F", None, "M", "UNKNOWN"],
            "age_years": [25, np.nan, 35, 65],
            "age_group": pd.Categorical(["25-34", None, "35-44", "65+"]),
            "DO_NOT_EXPORT": ["raw feature"] * 4,
        }, index=self.ids)
        self.record = {
            "model": "catboost", "feature_set": "application_history",
            "split_strategy": "stratified", "configuration": "starter_v1",
            "evaluation_partition": "validation", "holdout_evaluated": False,
            "rows": {"train": 20, "validation": 4},
            "metrics": experiments.validation_metrics(self.labels, self.probabilities),
            "split_fingerprint": "a" * 64, "parameters": {"depth": 6},
        }

    def export(self, result_path, directory, **kwargs):
        args = {"applicant_ids": self.ids, "labels": self.labels,
                "probabilities": self.probabilities, "audit": self.audit,
                "result_path": result_path, "directory": directory}
        return experiments.export_validation_predictions(**{**args, **kwargs})

    def test_roundtrip_alignment_audit_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            metric_path = experiments.save_run_record(self.record, Path(directory) / "metrics")
            csv_path, metadata_path = self.export(metric_path, Path(directory) / "predictions")
            rows = pd.read_csv(csv_path)
            np.testing.assert_array_equal(rows.SK_ID_CURR, self.ids)
            np.testing.assert_array_equal(rows.TARGET, self.labels)
            np.testing.assert_allclose(rows.predicted_probability, self.probabilities)
            self.assertEqual(rows.error_type.tolist(), ["TN", "FP", "FN", "TP"])
            self.assertEqual(rows.predicted_label.tolist(), [0, 1, 0, 1])
            self.assertEqual(rows.gender.iloc[1], "UNKNOWN")
            self.assertEqual(rows.age_group.iloc[1], "UNKNOWN")
            self.assertTrue(pd.isna(rows.age_years.iloc[1]))
            self.assertNotIn("DO_NOT_EXPORT", rows)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            saved_run = json.loads(metric_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["run_record"], saved_run)
            self.assertEqual(rows.run_id.unique().tolist(), [saved_run["run_id"]])
            self.assertEqual(metadata["row_count"], 4)
            self.assertEqual(metadata["prediction_sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest())
            self.assertFalse(metadata["holdout_exported"])
            self.assertFalse(metadata["raw_features_exported"])
            self.assertEqual(set(metadata["columns"]), set(rows.columns))
            self.assertEqual(experiments.validation_metrics(rows.TARGET, rows.predicted_probability), self.record["metrics"])

    def test_repeated_export_reuses_identical_files_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            metric_path = experiments.save_run_record(self.record, directory)
            first = self.export(metric_path, Path(directory) / "predictions")
            contents = [path.read_bytes() for path in first]
            self.assertEqual(first, self.export(metric_path, Path(directory) / "predictions"))
            self.assertEqual(contents, [path.read_bytes() for path in first])
            # Deliberate corruption should not be silently replaced.
            first[0].write_text("changed\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.export(metric_path, Path(directory) / "predictions")
            self.assertEqual(first[0].read_text(), "changed\n")

    def test_rejects_misaligned_invalid_and_other_run_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            metric_path = experiments.save_run_record(self.record, directory)
            bad_inputs = [
                {"labels": self.labels.iloc[::-1]},
                {"audit": self.audit.iloc[::-1]},
                {"applicant_ids": np.array([104, 104, 103, 102])},
                {"applicant_ids": self.ids.astype(float)},
                {"probabilities": self.probabilities[::-1]},
                {"probabilities": pd.Series(self.probabilities, index=self.ids[::-1])},
                {"probabilities": np.array([np.nan, .6, .4, .9])},
                {"audit": self.audit.drop(columns="gender")},
                {"audit": self.audit.assign(age_years=-1)},
            ]
            for inputs in bad_inputs:
                with self.subTest(inputs=list(inputs)):
                    with self.assertRaises(ValueError):
                        self.export(metric_path, Path(directory) / "predictions", **inputs)
            self.assertFalse((Path(directory) / "predictions").exists())

    def test_holdout_is_rejected_and_default_location_is_repo_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            metric_path = experiments.save_run_record(self.record, Path(directory) / "metrics")
            with patch.object(experiments, "PREDICTIONS_DIR", Path(directory) / "repo" / "results" / "predictions"):
                csv_path, _ = self.export(metric_path, None)
            self.assertEqual(csv_path.parent, Path(directory) / "repo" / "results" / "predictions")
            bad = {**self.record, "holdout_evaluated": True}
            bad_path = experiments.save_run_record(bad, Path(directory) / "metrics")
            with self.assertRaisesRegex(ValueError, "validation-only"):
                self.export(bad_path, Path(directory) / "bad")

    def test_repo_size_guard_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            metric_path = experiments.save_run_record(self.record, directory)
            default_path = Path(directory) / "predictions"
            with patch.object(experiments, "PREDICTIONS_DIR", default_path), \
                    patch.object(experiments, "MAX_GIT_PREDICTION_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "50 MiB"):
                    self.export(metric_path, None)
            self.assertFalse(default_path.exists())


if __name__ == "__main__":
    unittest.main()
