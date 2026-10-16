# Preprocessing inspection and proposed hypothesis

Current preparation already maps DAYS_EMPLOYED=365243 to missing and creates DAYS_EMPLOYED_ANOMALY data_preparation.ipynb, code cell 20, zero-based cell index. Adding this handling again is not justified.

The neural network uses training-median numeric imputation, scaling, and missingness indicators for columns missing in training (modeling/neural_network.ipynb, cell 5). CatBoost preserves numeric missing values and replaces missing categories with a string token modeling/catboost.ipynb, cell 5. LightGBM preserves numeric missing values and uses training category vocabularies (modeling/lightgbm.ipynb, cell 5). These observations establish code behavior, not the cause of an individual error.

Application preparation retains original predictors and adds the employment anomaly flag. The history join adds history aggregates, including other financial ratios, but neither application credit-to-income nor application annuity-to-income is constructed (preparation cells 20, 22–32). Those two ratios currently exist in the review export only (merge_predictions.py). There is no explicit combined external-score missing-count feature in application preparation; NN numeric indicators and tree missing-value handling already represent individual missingness.

## Representative comparisons

representative_case_comparisons.csv compares a shared high-ratio miss, a shared employment-sentinel miss, and a model disagreement with all-model correct negative controls. Prefer controls with exactly the same four exploratory flags, then minimize absolute differences in age (10-year scale), income (100,000 scale), and credit (500,000 scale) within the available reviewed controls. If no exact flag match exists, use all reviewed controls and mark same_four_flags=False. These are illustrative comparisons, not causal matching or a population analysis. Controls have a different actual outcome, so matching visible features does not imply they should receive identical scores. History features are absent from the review export, limiting explanations of history-model predictions.

- shared_miss_high_ratio: case 101842 versus control 402651; same four flags: False. Case outcomes: nn FN (0.055), catboost FN (0.070), lightgbm FN (0.058).
- shared_miss_employment_sentinel: case 101842 versus control 402651; same four flags: False. Case outcomes: nn FN (0.055), catboost FN (0.070), lightgbm FN (0.058).
- model_disagreement: case 105485 versus control 339595; same four flags: True. Case outcomes: nn TN (0.430), catboost FP (0.625), lightgbm FP (0.602).

## One proposed experiment

Hypothesis: explicit current-application credit-to-income and annuity-to-income features may make relative financial burden easier to represent. Existing raw amounts do not guarantee the fitted models learned this relationship. Add both as one future feature experiment; use missing for missing or nonpositive income. Hold threshold and other settings fixed to isolate the feature change. Sample patterns and mixed existing group gaps do not establish that these features will improve performance.

The experiment log records this hypothesis and example IDs for each baseline. change_tested, new run IDs, before/after results, evaluation partition, and improvement fields remain blank. No experiment was run. The existing prepared cache and historical execution were not reconstructed, so current source inspection does not independently prove the precise transformed values used by the saved runs. The reserved blackbox partition was not inspected.

## Subsequent experiment

The no-testing statements above describe the initial review phase. An authorized follow-up has now completed a separate paired CatBoost ratio-feature and threshold comparison. See ../../experiments/catboost_income_ratios_20261014T174042113531Z/report.md. The ratio AUC gain is inconclusive; the existing feature set remains the reference. Original notebooks and the final holdout remain unchanged.
