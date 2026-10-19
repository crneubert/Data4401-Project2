# Income-ratio experiment

catboost_income_ratios.py trains two new CatBoost models: a control with the existing application/history features and a variant adding application credit-to-income and annuity-to-income. It does not edit the original notebooks or replace historical results.


The first command checks the saved split and feature-schema fingerprints, trains on the original training applicants, uses only the existing eyeball partition for early stopping and maximum-F1 threshold selection, and evaluates on blackbox. The original holdout is unused. Blackbox belongs to validation previously used by historical models, so it is not an untouched final holdout. The second command computes a paired stratified bootstrap interval for the AUC difference, adds group summaries to the report, and updates only CatBoost's row in the error-analysis experiment log.

The experiment writes to results/experiments/catboost_income_ratios containing separate models, parameters and environment versions, predictions, fixed-threshold and selected-threshold comparisons, group metrics, and a report. Predictions and model files should remain local/team-only. Threshold selection maximizes eyeball F1 as an exploratory objective; it does not encode lending costs. Compare the two fixed-0.5 rows for feature effects. Compare the two selected-threshold policies separately. All results remain single-run development evidence.

## Run the comparison

With the prepared cache and dependencies (including CatBoost and tabulate) available:

```sh
python modeling/experiments/catboost_income_ratios.py
python modeling/experiments/summarize_income_ratios.py results/experiments/catboost_income_ratios
```

Generated prediction CSVs and model weights stay local. The annotated review, review split and membership, small metric summaries, configurations, and reports are versioned. Re-running merge_predictions.py creates a new blank review rather than reproducing completed annotations.

Result folders use readable names: `results/error_analysis/review` and `results/experiments/catboost_income_ratios`. Generation scripts refuse to overwrite existing folders; archive the existing folder before a fresh run to preserve annotations and results.
