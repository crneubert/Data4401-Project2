# CatBoost income-ratio experiment

Historical models used full validation for early stopping. This paired rerun uses eyeball only; blackbox is reserved within previously used validation and is not an untouched final holdout.

Both variants use the same training applicants, parameters, seed, and eyeball-only early stopping. Only the two income-ratio features differ. F1 thresholds were selected on eyeball rows, separately per variant, before blackbox evaluation. F1 is an exploratory selection objective; it does not encode lending costs. Compare fixed-0.5 rows to isolate feature effects and threshold-policy rows to assess the combined decision rule.

| variant        | policy     | partition   |   rows |   training_seconds |   trees |   roc_auc |   average_precision |   accuracy |        f1 |   precision |    recall |   log_loss |   threshold |    tn |   fp |   fn |   tp |
|:---------------|:-----------|:------------|-------:|-------------------:|--------:|----------:|--------------------:|-----------:|----------:|------------:|----------:|-----------:|------------:|------:|-----:|-----:|-----:|
| baseline_rerun | fixed_0.5  | blackbox    |  36902 |            434.876 |     979 |  0.783802 |            0.285494 |   0.920411 | 0.0898667 |    0.584677 | 0.0486741 |   0.237517 |    0.5      | 33820 |  103 | 2834 |  145 |
| baseline_rerun | eyeball_f1 | blackbox    |  36902 |            434.876 |     979 |  0.783802 |            0.285494 |   0.856485 | 0.336673  |    0.268531 | 0.451158  |   0.237517 |    0.154263 | 30262 | 3661 | 1635 | 1344 |
| income_ratios  | fixed_0.5  | blackbox    |  36902 |            410.956 |     994 |  0.784274 |            0.285549 |   0.920492 | 0.0927644 |    0.588235 | 0.0503525 |   0.237434 |    0.5      | 33818 |  105 | 2829 |  150 |
| income_ratios  | eyeball_f1 | blackbox    |  36902 |            410.956 |     994 |  0.784274 |            0.285549 |   0.841743 | 0.331349  |    0.251434 | 0.485733  |   0.237434 |    0.141003 | 29615 | 4308 | 1532 | 1447 |

Group counts, recall, and false-positive rates are in group_comparison.csv; small groups can be unstable. No final holdout evaluation was performed. Historical results are references, not the paired control. Model files are saved separately in this experiment directory.

## Decision

No clear AUC improvement established; retain the original model as the reference and do not promote the ratio variant based on this run. AUC change: +0.00047; paired stratified bootstrap 95% interval [-0.00058, +0.00172] (200 resamples, seed 42). This interval describes applicant sampling conditional on these fitted models and excludes training variability.

## Readable blackbox comparison

| variant        | policy     |   threshold |   roc_auc |   average_precision |   precision |    recall |        f1 |   fp |   fn |
|:---------------|:-----------|------------:|----------:|--------------------:|------------:|----------:|----------:|-----:|-----:|
| baseline_rerun | fixed_0.5  |    0.5      |  0.783802 |            0.285494 |    0.584677 | 0.0486741 | 0.0898667 |  103 | 2834 |
| baseline_rerun | eyeball_f1 |    0.154263 |  0.783802 |            0.285494 |    0.268531 | 0.451158  | 0.336673  | 3661 | 1635 |
| income_ratios  | fixed_0.5  |    0.5      |  0.784274 |            0.285549 |    0.588235 | 0.0503525 | 0.0927644 |  105 | 2829 |
| income_ratios  | eyeball_f1 |    0.141003 |  0.784274 |            0.285549 |    0.251434 | 0.485733  | 0.331349  | 4308 | 1532 |

## Group review

The following recall and false-positive-rate ranges cover groups with at least 30 actual positives and 30 actual negatives. Smaller groups remain in the detailed CSV but are excluded from these ranges. Differences describe these data and do not establish equal performance or causation.

| variant        | policy     | attribute   |   eligible_groups |   excluded_small_groups |   recall_min |   recall_max |    fpr_min |    fpr_max |
|:---------------|:-----------|:------------|------------------:|------------------------:|-------------:|-------------:|-----------:|-----------:|
| baseline_rerun | eyeball_f1 | age_group   |                 6 |                       0 |    0.0857143 |    0.594595  | 0.0079096  | 0.231349   |
| baseline_rerun | eyeball_f1 | gender      |                 2 |                       1 |    0.392032  |    0.53328   | 0.0853318  | 0.152614   |
| baseline_rerun | fixed_0.5  | age_group   |                 6 |                       0 |    0         |    0.0645856 | 0          | 0.00678222 |
| baseline_rerun | fixed_0.5  | gender      |                 2 |                       1 |    0.0438799 |    0.0553328 | 0.00230507 | 0.00448865 |
| income_ratios  | eyeball_f1 | age_group   |                 6 |                       0 |    0.114286  |    0.632432  | 0.0146893  | 0.264506   |
| income_ratios  | eyeball_f1 | gender      |                 2 |                       1 |    0.42552   |    0.569366  | 0.100536   | 0.17937    |
| income_ratios  | fixed_0.5  | age_group   |                 6 |                       0 |    0         |    0.0702703 | 0          | 0.00602864 |
| income_ratios  | fixed_0.5  | gender      |                 2 |                       1 |    0.0409931 |    0.063352  | 0.00234939 | 0.00457666 |

## Practical interpretation

The ratio variant adds only 0.00047 AUC; the interval includes zero. Keep the existing feature set as the reference. At their independently selected eyeball-F1 thresholds, the baseline rerun has higher blackbox F1 (0.3367 versus 0.3313), so the ratio variant is not the preferred candidate under that exploratory objective either.

For the baseline rerun, lowering the threshold from 0.5 to the eyeball-selected 0.1543 raises recall from 4.87% to 45.12% and F1 from 0.0899 to 0.3367, but lowers precision from 58.47% to 26.85%. It catches 1,199 additional defaults while adding 3,558 false positives on 36,902 blackbox applicants. This is a candidate policy for further evaluation, not an adopted deployment threshold.

At that baseline threshold, female/male recall is 39.2%/53.3%, and false-positive rates are 8.5%/15.3%. Age-group recall ranges from 8.6% (65+, only 35 actual defaults) to 59.5% (<25). These differences remain unresolved; no fairness improvement is claimed. UNKNOWN gender has only two applicants and no actual defaults.

Completed validation: all 28 repository tests passed, including ratio boundary and threshold-selection tests. Both saved model files load successfully, their tree counts match the comparison, and the prediction export contains one row per applicant per variant. The original notebooks remain unchanged.

Next decision: choose an acceptable false-positive burden or cost objective before selecting a final threshold. Lock the resulting policy before any final holdout evaluation. No final holdout was accessed in this experiment.
