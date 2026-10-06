"""Path checks for notebooks launched from the root or modeling directory."""

import contextlib
import unittest
from pathlib import Path

from scripts import experiment_utils
from test_preparation import cells, execute_cell

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_notebook_imports_from_both_working_directories(self):
        notebooks = (
            "neural_network.ipynb", "lightgbm.ipynb", "catboost.ipynb",
            "model_comparison.ipynb",
        )
        for working_directory in (ROOT, ROOT / "modeling"):
            for notebook in notebooks:
                with self.subTest(directory=working_directory, notebook=notebook):
                    namespace = {"__name__": "__main__"}
                    dependency_cell = next(c for c in cells(notebook) if c.get("id") == "dependencies")
                    with contextlib.chdir(working_directory):
                        execute_cell(namespace, dependency_cell)
                    self.assertEqual(namespace["REPO_ROOT"], ROOT)
                    if notebook == "model_comparison.ipynb":
                        self.assertEqual(namespace["RESULTS_DIR"], ROOT / "results" / "metrics")

    def test_export_defaults_remain_at_repository_root(self):
        self.assertEqual(experiment_utils.REPO_ROOT, ROOT)
        self.assertEqual(experiment_utils.RESULTS_DIR, ROOT / "results" / "metrics")
        self.assertEqual(experiment_utils.PREDICTIONS_DIR, ROOT / "results" / "predictions")


if __name__ == "__main__":
    unittest.main()
