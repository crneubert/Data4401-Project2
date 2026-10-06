"""Offline metric/export and starter wiring checks; never fit a real classifier."""

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import experiment_utils
from scripts import project_data
import test_preparation
from test_preparation import cells, execute_cell


class ExperimentUtilsTests(unittest.TestCase):
    def test_metrics_and_fixed_threshold(self):
        result = experiment_utils.validation_metrics([0, 0, 1, 1], [.1, .6, .4, .9])
        self.assertEqual(result["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertEqual(result[name], .5)
        self.assertEqual(result["roc_auc"], .75)
        self.assertEqual(result["threshold"], .5)

    def test_metrics_reject_bad_inputs(self):
        for labels, probabilities in [
            ([0, 1], [.2]), ([0, 0], [.1, .2]), ([0, 1], [np.nan, .8]),
            ([0, 1], [-.1, .8]), ([0, 1], [.1, 1.2]),
            ([0, 1], [[.1], [.8]]),
        ]:
            with self.subTest(labels=labels, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    experiment_utils.validation_metrics(labels, probabilities)

    def test_fingerprint_tracks_membership_order_and_labels(self):
        split = {"train": [1, 2], "validation": [3, 4], "holdout": [5, 6]}
        labels = pd.Series([0, 1, 0, 1, 0, 1], index=[1, 2, 3, 4, 5, 6])
        first = experiment_utils.split_fingerprint(split, labels)
        self.assertEqual(first, experiment_utils.split_fingerprint(split, labels.copy()))
        changed_order = {**split, "train": [2, 1]}
        self.assertNotEqual(first, experiment_utils.split_fingerprint(changed_order, labels))
        changed_label = labels.copy()
        changed_label.loc[1] = 1
        self.assertNotEqual(first, experiment_utils.split_fingerprint(split, changed_label))

    def test_unique_json_records_and_safe_names(self):
        record = {
            "model": "nn", "feature_set": "application_only", "split_strategy": "stratified",
            "configuration": "starter_v1", "metrics": {"roc_auc": .75},
        }
        with tempfile.TemporaryDirectory() as directory:
            first = experiment_utils.save_run_record(record, directory)
            second = experiment_utils.save_run_record(record, directory)
            self.assertNotEqual(first, second)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)
            saved = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(saved["metrics"], record["metrics"])
            self.assertIn("created_at_utc", saved)
            with self.assertRaises(ValueError):
                experiment_utils.save_run_record({**record, "model": "../bad"}, directory)
            with self.assertRaises(ValueError):
                experiment_utils.save_run_record({**record, "metrics": {"roc_auc": np.nan}}, directory)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)


class MockClassifier:
    """API spy only: fit records arguments, but performs no learning."""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.best_iteration_ = 3
        self.n_estimators_ = 4
        self.evals_result_ = {"validation": {"auc": [.6, .7, .75]}}

    def fit(self, inputs, labels, **kwargs):
        self.fit_ids = inputs.index.to_numpy()
        self.fit_label_ids = labels.index.to_numpy()
        self.fit_kwargs = kwargs
        return self

    def predict_proba(self, inputs, **kwargs):
        self.prediction_ids = inputs.index.to_numpy()
        self.predict_kwargs = kwargs
        positive = np.linspace(.1, .9, len(inputs))
        return np.column_stack([1 - positive, positive])

    def get_best_iteration(self):
        return 2

    def get_evals_result(self):
        return self.evals_result_


class StarterWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the same small offline history fixture used by prep tests.
        test_preparation.PreparationTests.setUpClass()
        cls.bundle = test_preparation.PreparationTests.ns["prepared_data"]

    def prepare_namespace(self, notebook, feature_set):
        namespace = {"__name__": "__main__"}
        with patch.object(project_data, "load_prepared_data", return_value=self.bundle):
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in cells(notebook):
                    if cell.get("id") == "training-config":
                        break
                    execute_cell(namespace, cell)
                    if cell.get("id") == "dependencies":
                        namespace["FEATURE_SET"] = feature_set
        namespace["display"] = lambda value: None
        return namespace

    def check_prediction_export(self, notebook, namespace, directory):
        # Execute the real new cell, redirecting ONLY its output directory for tests.
        cell = next(c for c in cells(notebook) if c.get("id") == "prediction-export")
        source = "".join(cell["source"]).replace(
            "PREDICTION_EXPORT_DIR = None",
            "PREDICTION_EXPORT_DIR = " + repr(str(Path(directory) / "predictions")),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "prediction-export", "exec"), namespace)
        frame = pd.read_csv(namespace["prediction_csv_path"])
        np.testing.assert_array_equal(frame.SK_ID_CURR, namespace["applicant_ids"]["validation"])
        np.testing.assert_array_equal(frame.TARGET, namespace["labels"]["validation"])
        np.testing.assert_allclose(frame.predicted_probability, namespace["validation_probability"], rtol=1e-7)
        self.assertEqual(len(frame), len(namespace["labels"]["validation"]))
        metadata = json.loads(namespace["prediction_metadata_path"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["metric_file"], namespace["result_path"].name)
        self.assertEqual(metadata["run_record"]["feature_set"], namespace["FEATURE_SET"])
        self.assertFalse(metadata["holdout_exported"])

    def test_tree_training_wiring_with_mocks_only(self):
        fake_lgb = types.SimpleNamespace(
            LGBMClassifier=MockClassifier,
            early_stopping=lambda rounds, **kwargs: ("early_stopping", rounds, kwargs),
            log_evaluation=lambda rounds: ("log_evaluation", rounds),
        )
        fake_cat = types.SimpleNamespace(CatBoostClassifier=MockClassifier)
        for notebook, module_name, fake in [
            ("lightgbm.ipynb", "lightgbm", fake_lgb),
            ("catboost.ipynb", "catboost", fake_cat),
        ]:
            for feature_set in project_data.FEATURE_SETS:
                with self.subTest(notebook=notebook, feature_set=feature_set):
                    namespace = self.prepare_namespace(notebook, feature_set)
                    with tempfile.TemporaryDirectory() as directory:
                        namespace["save_run_record"] = lambda record: experiment_utils.save_run_record(record, directory)
                        with patch.dict("sys.modules", {module_name: fake}):
                            with contextlib.redirect_stdout(io.StringIO()):
                                for cell in cells(notebook):
                                    if cell.get("id") in {"training-config", "train", "validation-export"}:
                                        execute_cell(namespace, cell)
                        model = namespace["model"]
                        self.check_prediction_export(notebook, namespace, directory)
                        np.testing.assert_array_equal(model.fit_ids, namespace["applicant_ids"]["train"])
                        np.testing.assert_array_equal(model.fit_label_ids, model.fit_ids)
                        np.testing.assert_array_equal(model.prediction_ids, namespace["applicant_ids"]["validation"])
                        eval_set = model.fit_kwargs["eval_set"]
                        if notebook == "lightgbm.ipynb":
                            self.assertEqual(len(eval_set), 1)
                            eval_inputs, eval_labels = eval_set[0]
                            self.assertEqual(model.predict_kwargs["num_iteration"], 3)
                            self.assertEqual(model.parameters["n_jobs"], 4)
                            self.assertEqual(model.parameters["num_leaves"], 31)
                        else:
                            eval_inputs, eval_labels = eval_set
                            self.assertTrue(model.fit_kwargs["use_best_model"])
                            self.assertFalse(model.parameters["allow_writing_files"])
                            self.assertEqual(model.parameters["depth"], 6)
                        np.testing.assert_array_equal(eval_inputs.index, namespace["applicant_ids"]["validation"])
                        self.assertTrue(eval_labels.index.equals(eval_inputs.index))
                        saved = json.loads(namespace["result_path"].read_text(encoding="utf-8"))
                        self.assertFalse(saved["holdout_evaluated"])
                        self.assertFalse(saved["threshold_tuned"])
                        self.assertEqual(saved["evaluation_partition"], "validation")
                        self.assertEqual(saved["metrics"]["threshold"], .5)
                        self.assertEqual(saved["selected_iteration"], 3)

    def test_nn_forward_and_export_without_training(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed; syntax and shared prep checks still apply.")
        for feature_set in project_data.FEATURE_SETS:
            namespace = self.prepare_namespace("neural_network.ipynb", feature_set)
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in cells("neural_network.ipynb"):
                    if cell.get("id") in {"training-config", "nn-training-functions"}:
                        execute_cell(namespace, cell)
            config = namespace["NN_CONFIG"]
            self.assertEqual(config["hidden_units"], [128, 64])
            self.assertEqual(config["max_epochs"], 30)
            network = namespace["StarterNetwork"](
                namespace["nn_inputs"]["train"].shape[1], config["hidden_units"], config["dropout"],
            ).eval()
            before = {key: value.detach().clone() for key, value in network.state_dict().items()}
            probabilities = namespace["predict_nn"](network, namespace["nn_inputs"]["validation"], 2)
            self.assertEqual(probabilities.shape, namespace["nn_labels"]["validation"].shape)
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
            for key, value in network.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))  # no weight updates

            # Exercise the notebook's train/export orchestration with a stub, not the trainer.
            namespace["fit_starter_nn"] = lambda *args: (network, [], 1)
            with tempfile.TemporaryDirectory() as directory:
                namespace["save_run_record"] = lambda record: experiment_utils.save_run_record(record, directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    for cell in cells("neural_network.ipynb"):
                        if cell.get("id") in {"train", "validation-export"}:
                            execute_cell(namespace, cell)
                saved = json.loads(namespace["result_path"].read_text(encoding="utf-8"))
                self.check_prediction_export("neural_network.ipynb", namespace, directory)
                self.assertEqual(saved["model"], "nn")
                self.assertEqual(saved["selected_iteration"], 1)
                self.assertFalse(saved["holdout_evaluated"])
                for key, value in network.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__":
    unittest.main()
