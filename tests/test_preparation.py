"""Small offline checks for notebook preparation; no model training or downloads."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import project_data

ROOT = Path(__file__).resolve().parents[1]


def cells(filename):
    directory = ROOT if filename == "data_preparation.ipynb" else ROOT / "modeling"
    return json.loads((directory / filename).read_text(encoding="utf-8"))["cells"]


def execute_cell(namespace, cell):
    if cell["cell_type"] == "code":
        # User-run notebooks may contain active dependency-install magics.
        # Offline tests must never install anything, and exec() cannot parse magics.
        source = "".join(line for line in cell["source"]
                         if not line.lstrip().startswith("%pip install "))
        exec(compile(source, cell.get("id", "notebook"), "exec"), namespace)


def fixture_tables():
    train = pd.DataFrame({
        "SK_ID_CURR": np.arange(100, 160), "TARGET": [0, 1] * 30,
        "DAYS_EMPLOYED": [-100, 365243, np.nan] * 20,
        "CODE_GENDER": ["M", "F", "XNA"] * 20,
        "DAYS_BIRTH": [-10000] * 60,
        "AMT_INCOME_TOTAL": [1000, 2000, np.nan] * 20,
        "ALL_MISSING_NUMERIC": [np.nan] * 60,
    })
    competition = train.drop(columns="TARGET").iloc[:6].copy()
    competition["SK_ID_CURR"] = np.arange(200, 206)
    competition["CODE_GENDER"] = "UNSEEN"
    bureau = pd.DataFrame({
        "SK_ID_CURR": [100, 100, 200, 101], "SK_ID_BUREAU": [1, 2, 3, 4],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active"],
        "DAYS_CREDIT": [-30, -500, -10, 10], "CREDIT_DAY_OVERDUE": [3, 0, 0, 99],
        "AMT_CREDIT_SUM": [100, 200, 300, 999],
        "AMT_CREDIT_SUM_DEBT": [50, np.nan, 0, 999],
        "AMT_CREDIT_SUM_OVERDUE": [2, np.nan, 0, 999],
        "AMT_CREDIT_MAX_OVERDUE": [2, np.nan, 0, 999],
        "CNT_CREDIT_PROLONG": [0, 1, 0, 9],
    })
    bb = pd.DataFrame({
        "SK_ID_BUREAU": [1, 1, 1, 2, 3, 4, 999],
        "MONTHS_BALANCE": [-1, -2, 1, -20, 0, 0, 0],
        "STATUS": ["1", "X", "5", "C", "0", "5", "5"],
    })
    previous = pd.DataFrame({
        "SK_ID_CURR": [100, 100, 200, 101], "SK_ID_PREV": [10, 11, 12, 13],
        "AMT_APPLICATION": [100, 0, 200, 999], "AMT_CREDIT": [80, 0, 200, 999],
        "AMT_ANNUITY": [10, 0, 20, 999], "CNT_PAYMENT": [8, 0, 10, 999],
        "DAYS_DECISION": [-10, -500, -50, 10],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Approved"],
    })
    installments = pd.DataFrame({
        "SK_ID_CURR": [100, 100, 100, 100, 200, 101],
        "SK_ID_PREV": [10, 10, 10, 10, 12, 13],
        "NUM_INSTALMENT_VERSION": [1] * 6,
        "NUM_INSTALMENT_NUMBER": [1, 1, 1, 2, 1, 1],
        "DAYS_INSTALMENT": [-10, -10, -10, 10, -20, -15],
        "DAYS_ENTRY_PAYMENT": [-12, -8, 5, -5, -20, 10],
        "AMT_INSTALMENT": [100] * 6, "AMT_PAYMENT": [40, 60, 500, 200, 100, 999],
    })
    pos = pd.DataFrame({
        "SK_ID_CURR": [100, 100, 100, 200], "MONTHS_BALANCE": [-1, -20, 1, 0],
        "CNT_INSTALMENT_FUTURE": [2, 4, 99, 0], "SK_DPD": [0, 3, 99, 0],
        "SK_DPD_DEF": [0, 1, 99, 0], "NAME_CONTRACT_STATUS": ["Active", "Active", "Active", "Completed"],
    })
    cc = pd.DataFrame({
        "SK_ID_CURR": [100, 100, 100, 200], "MONTHS_BALANCE": [-1, -20, 1, 0],
        "AMT_BALANCE": [50, 100, 999, 0], "AMT_CREDIT_LIMIT_ACTUAL": [100, 0, 999, 100],
        "AMT_DRAWINGS_CURRENT": [10, 0, 999, 0], "AMT_PAYMENT_CURRENT": [5, 0, 999, 0],
        "AMT_INST_MIN_REGULARITY": [5, 0, 999, 0], "SK_DPD": [0, 3, 999, 0],
        "SK_DPD_DEF": [0, 1, 999, 0],
    })
    return {
        "application_train": train, "application_test": competition, "bureau": bureau,
        "bureau_balance": bb, "previous_application": previous,
        "installments_payments": installments, "POS_CASH_balance": pos,
        "credit_card_balance": cc,
    }


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = fixture_tables()

        def fixture_load(name, **kwargs):
            frame = cls.tables[name].copy()
            if "usecols" in kwargs:
                frame = frame[kwargs["usecols"]]
            if kwargs.get("dtype") is not None:
                frame = frame.astype(kwargs["dtype"])
            if "chunksize" in kwargs:
                return (frame.iloc[i:i + 2].copy() for i in range(0, len(frame), 2))
            return frame

        cls.ns = {
            "__name__": "__main__", "np": np, "pd": pd, "gc": __import__("gc"),
            "display": lambda value: None, "load_table": fixture_load,
            "application_train": cls.tables["application_train"],
            "application_test": cls.tables["application_test"],
            "SCHEMA_VERSION": project_data.SCHEMA_VERSION,
            "get_partitions": project_data.get_partitions,
            "validate_prepared_data": project_data.validate_prepared_data,
        }
        prep = cells("data_preparation.ipynb")
        with contextlib.redirect_stdout(io.StringIO()):
            for i in [13, 15, 17] + list(range(19, len(prep) - 1)):
                execute_cell(cls.ns, prep[i])

    def test_chunk_reductions_are_weighted_and_preserve_missing(self):
        frame = pd.DataFrame({"SK_ID_CURR": [1, 1, 1, 2], "VALUE": [2, 4, 12, np.nan]})
        spec = {"VALUE": ["mean", "sum", "min", "max"]}
        stats = self.ns["chunk_statistics"]
        combine = self.ns["combine_statistics"]
        one = combine([stats(frame, spec)], spec, "QA_")
        many = combine([stats(frame.iloc[:1], spec), stats(frame.iloc[1:], spec)], spec, "QA_")
        pd.testing.assert_frame_equal(one.sort_index(), many.sort_index())
        self.assertAlmostEqual(float(many.loc[1, "QA_VALUE_MEAN"]), 6)
        self.assertTrue(pd.isna(many.loc[2, "QA_VALUE_SUM"]))
        self.assertEqual(many.loc[2, "QA_RECORD_COUNT"], 1)

    def test_bureau_bridge_unknown_status_and_future_exclusion(self):
        bb = self.ns["history_blocks"]["bureau_balance"]
        self.assertEqual(bb.loc[100, "BB_ALL_RECORD_COUNT"], 3)
        self.assertAlmostEqual(float(bb.loc[100, "BB_ALL_DELINQUENT_MEAN"]), .5)
        self.assertEqual(bb.loc[100, "BB_RECENT_12M_DELINQUENT_MEAN"], 1)
        self.assertNotIn(101, bb.index)  # Future bureau loan never enters the bridge.
        self.assertEqual(self.ns["history_blocks"]["bureau"].loc[100, "BUR_ALL_RECORD_COUNT"], 2)

    def test_partial_installments_and_future_payments(self):
        block = self.ns["history_blocks"]["installments_payments"]
        self.assertEqual(block.loc[100, "INST_ALL_RECORD_COUNT"], 1)
        self.assertEqual(block.loc[100, "INST_ALL_AMT_INSTALMENT_SUM"], 100)
        self.assertEqual(block.loc[100, "INST_ALL_AMT_PAYMENT_SUM"], 100)
        self.assertEqual(block.loc[100, "INST_ALL_UNDERPAID_MEAN"], 0)
        self.assertEqual(block.loc[100, "INST_ALL_LAST_PAYMENT_DELAY_MAX"], 2)
        self.assertTrue(pd.isna(block.loc[101, "INST_ALL_AMT_PAYMENT_SUM"]))
        self.assertTrue(pd.isna(block.loc[101, "INST_ALL_LATE_LAST_PAYMENT_MEAN"]))

    def test_no_history_rows_and_identical_partitions(self):
        augmented = self.ns["prepared_data"]["feature_sets"]["application_history"]["labeled"]
        count_cols = [c for c in augmented if c.endswith("RECORD_COUNT")]
        flag_cols = [c for c in augmented if c.endswith("HAS_HISTORY")]
        self.assertTrue(augmented.loc[159, count_cols + flag_cols].eq(0).all())
        self.assertTrue(pd.isna(augmented.loc[159, "BUR_ALL_AMT_CREDIT_SUM_SUM"]))
        for strategy in ("random", "stratified"):
            for part in project_data.PARTITIONS:
                np.testing.assert_array_equal(
                    self.ns["application_splits"][strategy]["applicant_ids"][part],
                    self.ns["history_splits"][strategy]["applicant_ids"][part],
                )

    def test_monthly_recent_windows_and_zero_denominators(self):
        cc = self.ns["history_blocks"]["credit_card_balance"]
        self.assertEqual(cc.loc[100, "CC_ALL_RECORD_COUNT"], 2)
        self.assertEqual(cc.loc[100, "CC_RECENT_12M_RECORD_COUNT"], 1)
        self.assertAlmostEqual(float(cc.loc[100, "CC_ALL_UTILIZATION_MEAN"]), .5)

    def test_cache_roundtrip_and_all_model_notebooks(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            project_data.save_prepared_data(self.ns["prepared_data"], directory)
            with patch.object(project_data, "PREPARED_DIR", Path(directory)):
                for notebook in ("neural_network.ipynb", "lightgbm.ipynb", "catboost.ipynb"):
                    for feature_set in project_data.FEATURE_SETS:
                        for strategy in ("random", "stratified"):
                            namespace = {"__name__": "__main__"}
                            with contextlib.redirect_stdout(io.StringIO()):
                                for cell in cells(notebook):
                                    # Never import/execute model training when checking preparation.
                                    if cell.get("id") == "training-config":
                                        break
                                    execute_cell(namespace, cell)
                                    if cell.get("id") == "dependencies":
                                        namespace["FEATURE_SET"] = feature_set
                                        namespace["SPLIT_STRATEGY"] = strategy
                            data = project_data.get_partitions(
                                self.ns["prepared_data"], feature_set, strategy,
                                partitions=("train", "validation"),
                            )
                            self.assertEqual(set(namespace["labels"]), {"train", "validation"})
                            self.assertNotIn("raw_inputs", namespace)  # released before training
                            probe = data["raw_inputs"]["validation"].iloc[:2].copy()
                            probe["CODE_GENDER"] = "NEW_CATEGORY"
                            if notebook == "neural_network.ipynb":
                                values = namespace["nn_preprocessor"].transform(probe)
                                self.assertTrue(np.isfinite(values).all())
                            elif notebook == "lightgbm.ipynb":
                                converted = namespace["prepare_lightgbm"](
                                    probe, namespace["lgbm_category_levels"],
                                )
                                self.assertTrue(converted["CODE_GENDER"].isna().all())
                            else:
                                converted = namespace["prepare_catboost"](probe, data["categorical_features"])
                                self.assertTrue(converted["CODE_GENDER"].eq("NEW_CATEGORY").all())

    def test_selected_partitions_do_not_materialize_holdout(self):
        subset = project_data.get_partitions(
            self.ns["prepared_data"], "application_history", partitions=("train", "validation"),
        )
        for key in ("raw_inputs", "labels", "applicant_ids", "audit_by_partition"):
            self.assertEqual(set(subset[key]), {"train", "validation"})
        with self.assertRaises(ValueError):
            project_data.get_partitions(self.ns["prepared_data"], "application_only", partitions=("holdout",))

    def test_comparison_is_read_only_and_retired_notebook_is_gone(self):
        source = "\n".join("".join(c["source"]) for c in cells("model_comparison.ipynb") if c["cell_type"] == "code")
        self.assertIn("load_run_records", source)
        self.assertNotIn("load_prepared_data", source)
        self.assertNotIn(".fit(", source)
        self.assertFalse((ROOT / "classification_experiments.ipynb").exists())


if __name__ == "__main__":
    (ROOT / ".cache").mkdir(exist_ok=True)
    unittest.main()
