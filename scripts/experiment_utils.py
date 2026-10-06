"""Validation metrics and small shareable run records; no training side effects."""

import hashlib
import json
import platform
import re
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "metrics"
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
MAX_GIT_PREDICTION_BYTES = 50 * 1024 * 1024


def validation_metrics(y_true, probabilities, threshold=0.5):
    """One fixed-threshold evaluation; AP and ROC-AUC use probabilities."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("Labels and probabilities must be nonempty, aligned 1D arrays.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation needs both binary classes for ROC-AUC.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")
    if not 0 < threshold < 1:
        raise ValueError("Choose a probability threshold strictly between 0 and 1.")
    predicted = (p >= threshold).astype("uint8")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def split_fingerprint(split_ids, labels):
    """Hash ordered IDs AND labels, without exporting any applicant-level values."""
    digest = hashlib.sha256()
    for part in ("train", "validation", "holdout"):
        ids = np.asarray(split_ids[part], dtype="<i8")
        y = labels.loc[ids].to_numpy(dtype="uint8")
        digest.update(part.encode("utf-8"))
        digest.update(len(ids).to_bytes(8, "little"))
        digest.update(ids.tobytes())
        digest.update(y.tobytes())
    return digest.hexdigest()


def feature_schema_fingerprint(columns):
    return hashlib.sha256(json.dumps(list(columns), ensure_ascii=True).encode("utf-8")).hexdigest()


def package_versions(model_package):
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", model_package):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def save_run_record(record, directory=None):
    """Write a unique JSON file; never append/overwrite another teammate's run."""
    directory = Path(directory) if directory is not None else RESULTS_DIR
    slug_fields = ("model", "feature_set", "split_strategy", "configuration")
    for key in slug_fields:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(record.get(key, ""))):
            raise ValueError(f"Invalid run filename component: {key}")
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    output_record = {**record, "run_id": run_id, "created_at_utc": timestamp.isoformat()}
    # Validate serialization before creating a file. No NaN/Infinity JSON values.
    text = json.dumps(output_record, indent=2, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    filename = "_".join(str(record[key]) for key in slug_fields) + "_" + run_id + ".json"
    path = directory / filename
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def export_validation_predictions(applicant_ids, labels, probabilities, audit, result_path, directory=None):
    """Save applicant-level validation output in the repo, tied to a saved run.

    This function consumes existing predictions. It never fits a model, loads
    raw/prepared data, predicts again, tunes thresholds, or accesses holdout.
    """
    result_path = Path(result_path)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if record.get("evaluation_partition") != "validation" or record.get("holdout_evaluated") is not False:
        raise ValueError("Only validation-only run records can be exported here.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(record.get("run_id", ""))):
        raise ValueError("The saved metric file needs a valid run_id.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result_path.stem):
        raise ValueError("Invalid saved run filename.")
    ids = np.asarray(applicant_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Applicant IDs must be a one-dimensional integer array.")
    index = pd.Index(ids)
    if not index.is_unique:
        raise ValueError("Duplicate validation applicant IDs.")
    if not isinstance(labels, pd.Series) or not labels.index.equals(index):
        raise ValueError("Validation labels must be indexed in exactly the applicant-ID order.")
    if not isinstance(audit, pd.DataFrame) or not audit.index.equals(index):
        raise ValueError("Audit metadata must be indexed in exactly the applicant-ID order.")
    if not {"gender", "age_years", "age_group"}.issubset(audit.columns):
        raise ValueError("Audit metadata needs gender, age_years, and age_group.")
    if isinstance(probabilities, pd.Series) and not probabilities.index.equals(index):
        raise ValueError("Prediction Series is not in applicant-ID order.")
    p = np.asarray(probabilities)
    threshold = record["metrics"]["threshold"]
    measured = validation_metrics(labels.to_numpy(), p, threshold=threshold)
    if record["rows"]["validation"] != len(ids):
        raise ValueError("Validation row count differs from the saved run.")
    for key, value in measured.items():
        saved = record["metrics"].get(key)
        if key == "confusion_matrix":
            matches = saved == value
        else:
            matches = saved is not None and np.isclose(saved, value, rtol=1e-8, atol=1e-8)
        if not matches:
            raise ValueError(f"Predictions do not reproduce the saved run's {key}; export metrics for this fit first.")
    age = pd.to_numeric(audit["age_years"], errors="raise")
    if np.isinf(age.to_numpy(dtype=float)).any() or age.dropna().lt(0).any():
        raise ValueError("Age must be nonnegative/finite or missing, not an invented zero.")
    y = labels.to_numpy(dtype="uint8")
    predicted = (p >= threshold).astype("uint8")
    frame = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": y,
        "predicted_probability": p, "predicted_label": predicted,
        "error_type": np.select(
            [(y == 0) & (predicted == 0), (y == 0) & (predicted == 1),
             (y == 1) & (predicted == 0), (y == 1) & (predicted == 1)],
            ["TN", "FP", "FN", "TP"], default="UNKNOWN",
        ),
        "gender": audit["gender"].astype("string").fillna("UNKNOWN").to_numpy(),
        "age_years": age.to_numpy(),
        "age_group": audit["age_group"].astype("string").fillna("UNKNOWN").to_numpy(),
        "run_id": record["run_id"],
    })
    csv_bytes = frame.to_csv(index=False, na_rep="", lineterminator="\n").encode("utf-8")
    # Keep normal Git exports comfortably below its large-file warning threshold.
    repo_export = directory is None or Path(directory).resolve().is_relative_to(REPO_ROOT)
    if repo_export and len(csv_bytes) >= MAX_GIT_PREDICTION_BYTES:
        raise ValueError("Prediction CSV is at least 50 MiB. Use shared data storage instead of regular Git.")
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    if directory is None:
        directory = PREDICTIONS_DIR
    directory = Path(directory)
    csv_path = directory / (result_path.stem + ".csv")
    manifest_path = directory / (result_path.stem + ".metadata.json")
    metadata = {
        "export_schema_version": 1, "evaluation_partition": "validation",
        "row_count": len(frame), "prediction_file": csv_path.name,
        "prediction_sha256": csv_hash, "metric_file": result_path.name,
        "run_record": record,
        "columns": {
            "SK_ID_CURR": "Current application identifier; not a guaranteed unique person identifier.",
            "TARGET": "Observed binary target: 1 = payment difficulties, 0 = no such target event.",
            "predicted_probability": "Model probability for TARGET=1; full precision, not rounded for display.",
            "predicted_label": f"1 when predicted_probability >= {threshold}; otherwise 0.",
            "error_type": "TN=correct negative; FP=false positive; FN=false negative; TP=correct positive.",
            "gender": "Original audit category; XNA/missing represented by UNKNOWN.",
            "age_years": "Age at application from -DAYS_BIRTH/365.25; blank means missing.",
            "age_group": "Prepared age bands: [0,25), [25,35), [35,45), [45,55), [55,65), [65,infinity); missing=UNKNOWN.",
            "run_id": "Matches run_record.run_id and the aggregate metric JSON.",
        },
        "sharing_note": "Share CSV and metadata together in the team repository or data storage. Review data-sharing rules before making applicant-level records public.",
        "holdout_exported": False, "raw_features_exported": False,
    }
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    # Repeating the export cell for the same fit is safe; never overwrite changes.
    if csv_path.exists() or manifest_path.exists():
        if csv_path.is_file() and manifest_path.is_file():
            existing_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if hashlib.sha256(csv_path.read_bytes()).hexdigest() == csv_hash and existing_metadata == metadata:
                return csv_path, manifest_path
        raise FileExistsError("A different or incomplete prediction export already exists; refusing to overwrite it.")
    directory.mkdir(parents=True, exist_ok=True)
    csv_created = False
    try:
        with csv_path.open("xb") as stream:
            csv_created = True
            stream.write(csv_bytes)
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(metadata_text)
    except Exception:
        # Remove only our newly created CSV if the paired metadata could not be saved.
        if csv_created:
            csv_path.unlink(missing_ok=True)
        raise
    return csv_path, manifest_path
