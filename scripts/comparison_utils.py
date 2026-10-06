"""Read aggregate run records; never load data, fit models, or score holdout rows."""

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_NAMES = {
    "nn": "Neural network", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "logistic_regression": "Logistic regression (R baseline)",
}
METRICS = ("roc_auc", "average_precision", "accuracy", "f1", "precision", "recall", "log_loss")
FEATURE_SETS = ("application_only", "application_history")
MODELS = ("nn", "lightgbm", "catboost")


def _validate_record(record, source):
    for field in ("model", "feature_set", "split_strategy", "configuration", "created_at_utc"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{source}: missing/invalid {field}")
    if record["model"] not in MODEL_NAMES or record["feature_set"] not in FEATURE_SETS:
        raise ValueError(f"{source}: unknown model/feature set")
    timestamp = datetime.fromisoformat(record["created_at_utc"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"{source}: timestamp must include a timezone")
    if record.get("evaluation_partition") != "validation" or record.get("holdout_evaluated") is not False:
        raise ValueError(f"{source}: only validation-only records belong in this comparison")
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict) or metrics.get("roc_auc") is None:
        raise ValueError(f"{source}: ROC-AUC is required")
    for field in METRICS + ("threshold",):
        value = metrics.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (field != "log_loss" and value > 1)
        ):
            raise ValueError(f"{source}: invalid metric {field}")
    if metrics.get("threshold") is None or not 0 < metrics["threshold"] < 1:
        raise ValueError(f"{source}: classification threshold is required")
    rows = record.get("rows", {})
    for part in ("train", "validation"):
        if not isinstance(rows.get(part), int) or isinstance(rows[part], bool) or rows[part] <= 0:
            raise ValueError(f"{source}: invalid {part} row count")
    confusion = metrics.get("confusion_matrix")
    if confusion is not None:
        if not isinstance(confusion, dict) or set(confusion) != {"tn", "fp", "fn", "tp"}:
            raise ValueError(f"{source}: invalid confusion matrix keys")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in confusion.values()):
            raise ValueError(f"{source}: confusion counts must be nonnegative integers")
        if sum(confusion.values()) != rows["validation"]:
            raise ValueError(f"{source}: confusion counts do not match validation rows")
    for field in ("preparation_seconds", "training_seconds", "prediction_seconds"):
        value = record.get(field)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"{source}: invalid {field}")
    # The historical R baseline has different sampling AND raw feature cleaning.
    # Never promote it to a paired comparison merely because its seed is also 42.
    if record["model"] != "logistic_regression":
        for field in ("split_fingerprint", "feature_schema_fingerprint"):
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{source}: a SHA-256 {field} is required")


def load_run_records(directory):
    """Fail clearly on corrupt results rather than silently omitting a model."""
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError("expected one JSON object")
            _validate_record(record, path.name)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Cannot compare {path.name}: {exc}") from exc
        records.append({**record, "source_file": path.name})
    return records


def select_runs(records, strategy="stratified", configuration="starter_v1", split_hash=None):
    """Use latest run, NOT maximum AUC; require one explicit Python split cohort."""
    eligible = [r for r in records if r["split_strategy"] == strategy]
    candidates = [r for r in eligible if r["model"] in MODELS and r["configuration"] == configuration]
    hashes = {r["split_fingerprint"] for r in candidates}
    if split_hash is None and len(hashes) > 1:
        raise ValueError("Multiple split fingerprints found. Set SPLIT_HASH to select one cohort explicitly.")
    if split_hash is not None and split_hash not in hashes:
        raise ValueError("Selected SPLIT_HASH has no matching Python runs.")
    if split_hash is None and hashes:
        split_hash = next(iter(hashes))
    candidates = [r for r in candidates if r["split_fingerprint"] == split_hash]
    latest = {}
    for record in candidates:
        key = (record["model"], record["feature_set"])
        order = (pd.Timestamp(record["created_at_utc"]), record["source_file"])
        if key not in latest or order > latest[key][0]:
            latest[key] = (order, record)
    selected = [item[1] for item in latest.values()]
    for feature_set in FEATURE_SETS:
        group = [r for r in selected if r["feature_set"] == feature_set]
        if len({r["feature_schema_fingerprint"] for r in group}) > 1:
            raise ValueError(f"{feature_set}: raw feature schemas differ; rerun matching preparation/configurations.")
    if len({(r["rows"]["train"], r["rows"]["validation"]) for r in selected}) > 1:
        raise ValueError("Row counts differ within the selected split cohort.")
    if len({r["metrics"]["threshold"] for r in selected}) > 1:
        raise ValueError("Threshold metrics cannot be compared: selected runs use different thresholds.")
    baselines = [r for r in eligible if r["model"] == "logistic_regression"]
    if baselines:
        selected.append(max(baselines, key=lambda r: (pd.Timestamp(r["created_at_utc"]), r["source_file"])))
    return selected, split_hash


def results_frame(records):
    rows = []
    for record in records:
        metrics = record["metrics"]
        rows.append({
            "model": record["model"], "model_name": MODEL_NAMES[record["model"]],
            "feature_set": record["feature_set"], "configuration": record["configuration"],
            "comparison_group": "R baseline: unpaired reference" if record["model"] == "logistic_regression" else "paired Python runs",
            **{key: metrics.get(key) for key in METRICS},
            "threshold": metrics["threshold"], "train_rows": record["rows"]["train"],
            "validation_rows": record["rows"]["validation"],
            "validation_positive_rate": record.get("positive_rates", {}).get("validation"),
            "training_minutes": (record["training_seconds"] / 60
                                 if record.get("training_seconds") is not None else float("nan")),
            "preparation_seconds": record.get("preparation_seconds"),
            "selected_iteration": record.get("selected_iteration"),
            "source_file": record["source_file"], "created_at_utc": record["created_at_utc"],
        })
    return pd.DataFrame(rows).reindex(columns=[
        "model", "model_name", "feature_set", "configuration", "comparison_group", *METRICS,
        "threshold", "train_rows", "validation_rows", "validation_positive_rate", "training_minutes",
        "preparation_seconds", "selected_iteration", "source_file", "created_at_utc",
    ])


def coverage_frame(records):
    present = {(r["model"], r["feature_set"]) for r in records}
    rows = [{"model": MODEL_NAMES[m], "feature_set": f, "status": "available" if (m, f) in present else "missing export"}
            for m in MODELS for f in FEATURE_SETS]
    rows.append({"model": MODEL_NAMES["logistic_regression"], "feature_set": "application_only",
                 "status": "available (unpaired)" if ("logistic_regression", "application_only") in present else "missing export"})
    return pd.DataFrame(rows)


def history_gains(frame):
    """Within-model paired differences only; no differences from the R baseline."""
    paired = frame[frame["model"].isin(MODELS)]
    rows = []
    for model, group in paired.groupby("model", sort=True):
        by_feature = group.set_index("feature_set")
        if not set(FEATURE_SETS).issubset(by_feature.index):
            continue
        app, history = (by_feature.loc[name] for name in FEATURE_SETS)
        rows.append({"model": MODEL_NAMES[model], "application_only_auc": app.roc_auc,
                     "application_history_auc": history.roc_auc,
                     "auc_gain": history.roc_auc - app.roc_auc,
                     "average_precision_gain": history.average_precision - app.average_precision,
                     "log_loss_change": history.log_loss - app.log_loss,
                     "extra_training_minutes": history.training_minutes - app.training_minutes})
    return pd.DataFrame(rows).reindex(columns=["model", "application_only_auc", "application_history_auc",
                                             "auc_gain", "average_precision_gain", "log_loss_change", "extra_training_minutes"])


def validation_auc_history(record):
    """Normalize the three saved learning-curve formats without requiring ML libraries."""
    history = record.get("training_history")
    if record["model"] == "nn" and isinstance(history, list):
        return [row["epoch"] for row in history], [row["validation_roc_auc"] for row in history]
    if isinstance(history, dict):
        for name in ("validation", "valid_0", "validation_0"):
            metrics = history.get(name, {})
            for metric, values in metrics.items():
                if metric.lower() == "auc":
                    return list(range(1, len(values) + 1)), values
    return [], []
