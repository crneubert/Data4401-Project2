# Error review findings

Annotated all 313 unique applicants in manual_review_all_models.csv. These cases come from the eyeball partition. Each model contributed 100 sampled errors and 20 correct controls; overlapping applicants were merged. Counts below describe the combined, deliberately error-enriched review sample, not population prevalence. A row selected as a control for one model may be an error for another.

## Recurring patterns

| Pattern | Applicants |
|---|---:|
| shared_missed_default | 241 |
| missing_external_score | 219 |
| high_credit_to_income | 79 |
| correct_prediction_control | 57 |
| employment_sentinel | 45 |
| high_annuity_to_income | 35 |
| near_threshold_missed_default | 21 |
| model_disagreement | 15 |
| model_specific_missed_default | 8 |
| false_alarm | 7 |

## Outcomes within the combined review sample

| Model | FN | FP | TP | TN |
|---|---:|---:|---:|---:|
| nn | 249 | 0 | 0 | 64 |
| catboost | 245 | 5 | 4 | 59 |
| lightgbm | 243 | 5 | 6 | 59 |

Shared missed defaults dominate the reviewed cases. Near-threshold missed defaults form a smaller subset, so a threshold explanation alone is not established. Missing external scores also occur among correct controls; missingness must not be treated as a proven cause. High financial ratios and employment sentinel values are candidate inspection categories, not established model weaknesses.

## Checks against the existing eyeball-group summaries

These comparisons use the existing 9,225-applicant eyeball summaries, rather than the enriched sample. Positive gaps mean a higher error rate among flagged applicants. They are descriptive and have no uncertainty estimates.

| Model | Pattern | FNR gap (percentage points) | FPR gap (percentage points) |
|---|---|---:|---:|
| nn | missing_external_score | -0.30 | -0.01 |
| nn | credit_over_5x_income | -0.10 | +0.03 |
| nn | annuity_over_30pct_income | -1.70 | -0.03 |
| nn | employment_sentinel | +1.22 | -0.03 |
| catboost | missing_external_score | -3.13 | +0.11 |
| catboost | credit_over_5x_income | +1.33 | +0.04 |
| catboost | annuity_over_30pct_income | -0.00 | +0.21 |
| catboost | employment_sentinel | +4.42 | -0.29 |
| lightgbm | missing_external_score | -2.22 | +0.25 |
| lightgbm | credit_over_5x_income | +0.71 | -0.05 |
| lightgbm | annuity_over_30pct_income | -0.70 | +0.17 |
| lightgbm | employment_sentinel | +3.91 | -0.48 |

Missing external scores have a lower observed false-negative rate than complete scores for all three models at 0.5, which does not support claiming missingness as a demonstrated missed-default weakness. Employment-sentinel cases have higher observed false-negative rates, but only 89 actual defaults occur in that flagged group; differences may reflect case mix and limited counts. Credit and annuity ratio findings are mixed across models.

## Next decisions, without testing

1. Inspect existing missing-value, employment-sentinel, and ratio preprocessing before proposing any feature change.
2. Examine representative shared misses, disagreements, and correct controls together; distinguish associations from causes.
3. Select an explicit precision/recall or cost objective before any later threshold experiment.

No model was retrained, no threshold was changed or evaluated, and the blackbox partition was not inspected. The experiment log remains unchanged. Reviewer was removed; review_date remains blank.

## Completed preprocessing inspection

See `preprocessing_review.md` and `representative_case_comparisons.csv`. Sentinel cleaning and missing-value handling already exist in current source. Application income ratios are absent from preparation and remain an untested feature hypothesis. Case annotations and the experiment log now reflect these findings; no experiments were run.

## Subsequent experiment

The no-testing statements above describe the initial review phase. An authorized follow-up has now completed a separate paired CatBoost ratio-feature and threshold comparison. See `../../experiments/catboost_income_ratios_20261014T174042113531Z/report.md`. The ratio AUC gain is inconclusive; the existing feature set remains the reference. Original notebooks and the final holdout remain unchanged.
