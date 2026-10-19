from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.project_data import load_table


PROJECT_DIR = Path(__file__).resolve().parent
PREDICTION_DIR = PROJECT_DIR / "results" / "predictions"

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "error_analysis" / "review"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

THRESHOLD = 0.5
ERRORS_PER_MODEL = 100
CONTROLS_PER_MODEL = 20

run_ids = {
    "nn": "20261006T080721021120Z_70241f3b",
    "catboost": "20261006T080453081602Z_1732aa07",
    "lightgbm": "20261006T080557325548Z_da5c5738",
}


predictions = {}

for model, run_id in run_ids.items():
    filename = (
        f"{model}_application_history_stratified_"
        f"starter_v1_{run_id}.csv"
    )

    data = pd.read_csv(PREDICTION_DIR / filename)

    required = {
        "SK_ID_CURR",
        "TARGET",
        "predicted_probability",
        "run_id",
    }

    if not required.issubset(data.columns):
        raise ValueError(f"{model}: required columns are missing.")

    if data["SK_ID_CURR"].isna().any():
        raise ValueError(f"{model}: missing applicant IDs.")

    if not data["SK_ID_CURR"].is_unique:
        raise ValueError(f"{model}: duplicate applicant IDs.")

    if not data["TARGET"].isin([0, 1]).all():
        raise ValueError(f"{model}: invalid TARGET values.")

    if not data["predicted_probability"].between(0, 1).all():
        raise ValueError(f"{model}: invalid probabilities.")

    if not data["run_id"].eq(run_id).all():
        raise ValueError(f"{model}: unexpected run ID.")

    predictions[model] = (
        data.set_index("SK_ID_CURR").sort_index()
    )

reference = predictions["nn"]

for model, data in predictions.items():
    if not data.index.equals(reference.index):
        raise ValueError(f"{model}: applicant IDs differ.")

    if not np.array_equal(
        data["TARGET"].to_numpy(),
        reference["TARGET"].to_numpy(),
    ):
        raise ValueError(f"{model}: actual outcomes differ.")



# Sorting IDs above and using the same seed makes this reproducible.
eyeball_ids, blackbox_ids = train_test_split(
    reference.index.to_numpy(),
    test_size=0.80,
    random_state=42,
    stratify=reference["TARGET"].to_numpy(),
)

partition = pd.Series(
    "blackbox",
    index=reference.index,
    name="review_partition",
)
partition.loc[eyeball_ids] = "eyeball"

partition.reset_index().to_csv(
    OUTPUT_DIR / "validation_review_split.csv",
    index=False,
)



comparison = reference[["TARGET"]].copy()
comparison["review_partition"] = partition

for model, data in predictions.items():
    probability = data["predicted_probability"]
    label = (probability >= THRESHOLD).astype(int)

    error_type = np.where(
        data["TARGET"] == 1,
        np.where(label == 1, "TP", "FN"),
        np.where(label == 1, "FP", "TN"),
    )

    comparison[f"{model}_probability"] = probability
    comparison[f"{model}_label"] = label
    comparison[f"{model}_error_type"] = error_type
    comparison[f"{model}_run_id"] = run_ids[model]

comparison["threshold"] = THRESHOLD



applicants = load_table("application_train")

if applicants["SK_ID_CURR"].isna().any():
    raise ValueError("Application data contains missing IDs.")

if not applicants["SK_ID_CURR"].is_unique:
    raise ValueError("Application data contains duplicate IDs.")

original_labels = applicants.set_index("SK_ID_CURR")["TARGET"]
matched_labels = original_labels.reindex(reference.index)

if matched_labels.isna().any():
    raise ValueError("Some validation IDs are missing from source data.")

if not np.array_equal(
    matched_labels.to_numpy(),
    reference["TARGET"].to_numpy(),
):
    raise ValueError("Source TARGET differs from prediction TARGET.")

applicant_features = applicants.drop(columns=["TARGET"])

eyeball = comparison[
    comparison["review_partition"].eq("eyeball")
].reset_index()

eyeball = eyeball.merge(
    applicant_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not eyeball["_merge"].eq("both").all():
    raise ValueError("Some eyeball applicants did not match source data.")

eyeball = eyeball.drop(columns="_merge")



# Ratios are unknown if income is missing or nonpositive.
income = eyeball["AMT_INCOME_TOTAL"].where(
    eyeball["AMT_INCOME_TOTAL"] > 0
)

eyeball["credit_to_income"] = eyeball["AMT_CREDIT"] / income
eyeball["annuity_to_income"] = eyeball["AMT_ANNUITY"] / income


def flag_above(series, cutoff):
    """Keep missing values unknown rather than marking them False."""
    flag = (series > cutoff).astype("boolean")
    return flag.mask(series.isna())


# Exploratory rules for review, not established lending cutoffs.
pattern_flags = [
    "flag_missing_external_score",
    "flag_credit_over_5x_income",
    "flag_annuity_over_30pct_income",
    "flag_employment_sentinel",
]

eyeball["flag_missing_external_score"] = eyeball[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
].isna().any(axis=1).astype("boolean")

eyeball["flag_credit_over_5x_income"] = flag_above(
    eyeball["credit_to_income"], 5
)

eyeball["flag_annuity_over_30pct_income"] = flag_above(
    eyeball["annuity_to_income"], 0.30
)

eyeball["flag_employment_sentinel"] = (
    eyeball["DAYS_EMPLOYED"].eq(365243).astype("boolean")
).mask(eyeball["DAYS_EMPLOYED"].isna())

# These flags describe prediction outcomes.
# They are not used in flagged/unflagged weakness comparisons.
for model in predictions:
    eyeball[f"flag_{model}_near_threshold_fn"] = (
        eyeball[f"{model}_error_type"].eq("FN")
        & eyeball[f"{model}_probability"].ge(
            max(0, THRESHOLD - 0.10)
        )
        & eyeball[f"{model}_probability"].lt(THRESHOLD)
    )

all_flag_columns = [
    column for column in eyeball.columns
    if column.startswith("flag_")
]


def describe_flags(row):
    descriptions = []

    for column in all_flag_columns:
        value = row[column]

        if pd.isna(value):
            continue

        if bool(value):
            descriptions.append(column.removeprefix("flag_"))

    return ", ".join(descriptions) or "No rule triggered"


eyeball["automatic_flags"] = eyeball.apply(
    describe_flags,
    axis=1,
)



selection_records = []

for model in predictions:
    errors = eyeball[
        eyeball[f"{model}_error_type"].isin(["FP", "FN"])
    ]

    selected_errors = errors.sample(
        n=min(ERRORS_PER_MODEL, len(errors)),
        random_state=42,
    )

    correct = eyeball[
        eyeball[f"{model}_error_type"].isin(["TP", "TN"])
    ]

    selected_controls = correct.sample(
        n=min(CONTROLS_PER_MODEL, len(correct)),
        random_state=42,
    )

    for applicant_id in selected_errors["SK_ID_CURR"]:
        selection_records.append({
            "SK_ID_CURR": applicant_id,
            "model": model,
            "sample_type": "error",
        })

    for applicant_id in selected_controls["SK_ID_CURR"]:
        selection_records.append({
            "SK_ID_CURR": applicant_id,
            "model": model,
            "sample_type": "correct_control",
        })

selection = pd.DataFrame(selection_records)

# Keep the exact sample membership for each model.
selection.to_csv(
    OUTPUT_DIR / "manual_review_sample_membership.csv",
    index=False,
)

selection["reason"] = (
    selection["model"] + ": " + selection["sample_type"]
)

selected_for = selection.groupby("SK_ID_CURR")["reason"].agg(
    lambda values: "; ".join(sorted(set(values)))
)

review = (
    eyeball.set_index("SK_ID_CURR")
    .loc[selected_for.index]
    .copy()
)

review["selected_for"] = selected_for

for column in [
    "observations",
    "error_categories",
    "models_observation_applies_to",
    "hypothesized_cause",
    "proposed_change",
    "review_date",
]:
    review[column] = ""

# Put the main review fields first.
front_columns = [
    "TARGET",
    "selected_for",
    "automatic_flags",
    "observations",
    "error_categories",
    "models_observation_applies_to",
    "hypothesized_cause",
    "proposed_change",
]

remaining_columns = [
    column for column in review.columns
    if column not in front_columns
]

review = review[front_columns + remaining_columns].reset_index()

review.to_csv(
    OUTPUT_DIR / "manual_review_all_models.csv",
    index=False,
)



def divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan


results = []

for model in predictions:
    for flag in pattern_flags:
        groups = pd.Series(
            "Unknown",
            index=eyeball.index,
            dtype="object",
        )

        groups.loc[eyeball[flag].eq(True).fillna(False)] = "Flagged"
        groups.loc[eyeball[flag].eq(False).fillna(False)] = "Unflagged"

        for group in ["Flagged", "Unflagged", "Unknown"]:
            subset = eyeball.loc[groups.eq(group)]

            actual = subset["TARGET"]
            predicted = subset[f"{model}_label"]

            tp = int(((actual == 1) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            tn = int(((actual == 0) & (predicted == 0)).sum())

            results.append({
                "model": model,
                "flag": flag.removeprefix("flag_"),
                "group": group,
                "threshold": THRESHOLD,
                "applicants": len(subset),
                "actual_positives": tp + fn,
                "actual_negatives": tn + fp,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "positive_rate": divide(tp + fn, len(subset)),
                "error_rate": divide(fn + fp, len(subset)),
                "false_negative_rate": divide(fn, tp + fn),
                "false_positive_rate": divide(fp, tn + fp),
                "recall": divide(tp, tp + fn),
            })

flag_results = pd.DataFrame(results)

flag_results.to_csv(
    OUTPUT_DIR / "flag_group_performance.csv",
    index=False,
)

differences = []

for model in predictions:
    for flag in pattern_flags:
        flag_name = flag.removeprefix("flag_")

        table = flag_results[
            flag_results["model"].eq(model)
            & flag_results["flag"].eq(flag_name)
        ].set_index("group")

        flagged = table.loc["Flagged"]
        unflagged = table.loc["Unflagged"]

        differences.append({
            "model": model,
            "flag": flag_name,
            "threshold": THRESHOLD,
            "flagged_applicants": flagged["applicants"],
            "unflagged_applicants": unflagged["applicants"],
            "unknown_applicants": table.loc["Unknown", "applicants"],
            "flagged_actual_positives": flagged["actual_positives"],
            "unflagged_actual_positives": unflagged["actual_positives"],
            "flagged_actual_negatives": flagged["actual_negatives"],
            "unflagged_actual_negatives": unflagged["actual_negatives"],
            "FNR_gap_percentage_points": 100 * (
                flagged["false_negative_rate"]
                - unflagged["false_negative_rate"]
            ),
            "FPR_gap_percentage_points": 100 * (
                flagged["false_positive_rate"]
                - unflagged["false_positive_rate"]
            ),
        })

gap_results = pd.DataFrame(differences)

gap_results.to_csv(
    OUTPUT_DIR / "flag_performance_gaps.csv",
    index=False,
)



experiment_rows = []

for model, run_id in run_ids.items():
    experiment_rows.append({
        "model": model,
        "baseline_run_id": run_id,
        "observed_error_pattern": "",
        "supporting_applicant_ids": "",
        "change_tested": "",
        "new_run_id": "",
        "evaluation_partition": "",
        "before_roc_auc": "",
        "after_roc_auc": "",
        "before_recall": "",
        "after_recall": "",
        "before_f1": "",
        "after_f1": "",
        "before_threshold": THRESHOLD,
        "after_threshold": "",
        "fairness_findings": "",
        "improved": "",
        "decision": "",
    })

pd.DataFrame(experiment_rows).to_csv(
    OUTPUT_DIR / "experiment_log.csv",
    index=False,
)


print("Eyeball applicants:", len(eyeball_ids))
print("Blackbox applicants:", len(blackbox_ids))
print("Unique manual-review applicants:", len(review))

print("\nSamples per model:")
print(
    selection.groupby(["model", "sample_type"])
    .size()
    .to_string()
)

print("\nFlag comparisons:")
print(gap_results.round(3).to_string(index=False))

print("\nFiles saved to:")
print(OUTPUT_DIR)