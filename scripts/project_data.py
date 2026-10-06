"""Shared cache/partition handoff; feature engineering lives in the prep notebook."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DRIVE_FILE_IDS = {
    "application_test": "1aSCg7GlC4hluYPybtM00p9FzDWJHHtql",
    "application_train": "1leSsunoq4XyViBVgrZ0dCIEWQePWi_AV",
    "bureau": "1XoTBbU-SMbjz2G8h12TxiZpjJIa-cZh_",
    "bureau_balance": "1vDQPrD5LEc3E4FFBrpLqWaPRq3ZRg4UP",
    "credit_card_balance": "10gP4aVPjxhXt_uWanDNHeUV2HCA_j0Vp",
    "HomeCredit_columns_description": "1z95kRXkg5bPPAenOFPnsE8pZwQcznCok",
    "installments_payments": "19LEAwLvsJCOMuQ-GMmKNeQr8YNA4glVd",
    "POS_CASH_balance": "1_e2CfFcyrxki6j9KI5azFoQIP0EqlmYw",
    "previous_application": "1UltfcT1hOhF3S8e0VKYFyeKR7DGzisOd",
    "sample_submission": "1YNKdy7FL30LaeRLV7r1uyzQY6gOXJBFj",
}

_cache_root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache")))
DATA_DIR = Path(os.environ.get(
    "HOME_CREDIT_DATA_DIR",
    str(_cache_root / "data4401-project2" / "home-credit-default-risk"),
))
PREPARED_DIR = DATA_DIR.parent / "prepared"
SCHEMA_VERSION = 2
FEATURE_SETS = ("application_only", "application_history")
PARTITIONS = ("train", "validation", "holdout", "competition")


def load_table(name, **read_csv_kwargs):
    """Read a CSV (or chunk iterator), downloading once into the external cache."""
    if name not in DRIVE_FILE_IDS:
        raise ValueError(f"Unknown table {name!r}. Choose from {sorted(DRIVE_FILE_IDS)}")
    csv_path = DATA_DIR / f"{name}.csv"
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        import gdown

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Download to a separate path so an interrupted CSV is not mistaken for complete.
        download_path = csv_path.with_suffix(".csv.download")
        result = gdown.download(
            id=DRIVE_FILE_IDS[name], output=str(download_path),
            quiet=False, use_cookies=False, resume=True,
        )
        if result is None or not download_path.is_file() or download_path.stat().st_size == 0:
            raise FileNotFoundError(f"Download failed for {name}; rerun to retry.")
        download_path.replace(csv_path)
    return pd.read_csv(csv_path, **read_csv_kwargs)


def validate_prepared_data(bundle):
    """Reject inconsistent features, labels, or applicant membership before handoff."""
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Prepared cache version changed. Rerun data_preparation.ipynb.")
    y = bundle["labels"]
    if not y.index.is_unique or y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Expected unique labeled applicants with binary TARGET.")
    if not bundle["audit_metadata"].index.equals(y.index):
        raise ValueError("Audit metadata and labels are misaligned.")
    competition_index = None
    for name in FEATURE_SETS:
        frames = bundle["feature_sets"][name]
        train, competition = frames["labeled"], frames["competition"]
        if not train.index.equals(y.index) or not train.columns.equals(competition.columns):
            raise ValueError(f"Feature schema/alignment mismatch in {name}.")
        if not train.columns.is_unique or not competition.index.is_unique:
            raise ValueError(f"Duplicate features/applicants in {name}.")
        if {"TARGET", "SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"} & set(train.columns):
            raise ValueError("Targets and join IDs must not be predictors.")
        if not y.index.intersection(competition.index).empty:
            raise ValueError("Labeled and competition applicants overlap.")
        if competition_index is not None and not competition_index.equals(competition.index):
            raise ValueError("Competition row order differs across feature sets.")
        competition_index = competition.index
        for frame in (train, competition):
            numeric = frame.select_dtypes(include=np.number)
            if np.isinf(numeric.to_numpy()).any():
                raise ValueError("Numeric infinities must be replaced with missing values.")
    for strategy in ("random", "stratified"):
        splits = bundle["split_ids"][strategy]
        groups = [set(splits[name]) for name in PARTITIONS[:3]]
        if any(len(groups[i]) != len(splits[name]) for i, name in enumerate(PARTITIONS[:3])):
            raise ValueError("Repeated applicant in a partition.")
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Training, validation, and holdout must be disjoint.")
        if set.union(*groups) != set(y.index):
            raise ValueError("Partitions must cover every labeled applicant exactly once.")


def save_prepared_data(bundle, directory=None):
    """Atomically replace only our generated cache; no data is written to the repo."""
    validate_prepared_data(bundle)
    directory = Path(directory) if directory is not None else PREPARED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "classification_inputs_v2.pkl"
    handle, temporary = tempfile.mkstemp(prefix="classification_inputs_", suffix=".tmp", dir=directory)
    os.close(handle)
    try:
        pd.to_pickle(bundle, temporary)
        os.replace(temporary, output)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return output


def load_prepared_data(directory=None):
    """Load ONLY our trusted local pickle, never a downloaded/untrusted pickle."""
    directory = Path(directory) if directory is not None else PREPARED_DIR
    path = directory / "classification_inputs_v2.pkl"
    if not path.is_file():
        raise FileNotFoundError(
            f"No prepared cache at {path}. Run data_preparation.ipynb through its export cell first."
        )
    bundle = pd.read_pickle(path)
    validate_prepared_data(bundle)
    return bundle


def get_partitions(bundle, feature_set, strategy="stratified", partitions=None):
    """Select identical ID-ordered partitions for either raw feature set."""
    if feature_set not in FEATURE_SETS or strategy not in {"random", "stratified"}:
        raise ValueError("Choose a documented feature set and random/stratified strategy.")
    requested = tuple(partitions) if partitions is not None else PARTITIONS
    if "train" not in requested or len(set(requested)) != len(requested) or set(requested) - set(PARTITIONS):
        raise ValueError("Select unique documented partitions including train.")
    frames = bundle["feature_sets"][feature_set]
    applicant_ids = {
        part: bundle["split_ids"][strategy][part]
        for part in requested if part != "competition"
    }
    if "competition" in requested:
        applicant_ids["competition"] = frames["competition"].index.to_numpy()
    raw_inputs = {
        part: frames["labeled"].loc[applicant_ids[part]]
        for part in requested if part != "competition"
    }
    if "competition" in requested:
        raw_inputs["competition"] = frames["competition"]
    labels = {
        part: bundle["labels"].loc[applicant_ids[part]]
        for part in requested if part != "competition"
    }
    audit = {
        part: bundle["audit_metadata"].loc[applicant_ids[part]]
        for part in requested if part != "competition"
    }
    for part, frame in raw_inputs.items():
        if not np.array_equal(frame.index.to_numpy(), applicant_ids[part]):
            raise ValueError("Partition applicant ordering changed.")
        if part in labels and not frame.index.equals(labels[part].index):
            raise ValueError("Partition labels are misaligned.")
    numeric = raw_inputs["train"].select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in raw_inputs["train"] if col not in numeric]
    return {
        "raw_inputs": raw_inputs, "labels": labels, "applicant_ids": applicant_ids,
        "audit_by_partition": audit, "numeric_features": numeric,
        "categorical_features": categorical,
    }
