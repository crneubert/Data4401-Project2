# Data4401-Project2

Home Credit default-risk classification: shared data preparation, model experiments, and validation comparisons.

## Repository layout

```text
data_preparation.ipynb           Application exploration, history features, shared splits
modeling/
  logistic_regression_baseline.qmd  Original R baseline
  neural_network.ipynb           Neural-network preprocessing, training, evaluation
  lightgbm.ipynb                 LightGBM preprocessing, training, evaluation
  catboost.ipynb                 CatBoost preprocessing, training, evaluation
  model_comparison.ipynb         Saved metrics, history gains, learning curves
scripts/
  project_data.py                Drive cache and prepared-data handoff
  experiment_utils.py           Metrics, run records, validation predictions
  comparison_utils.py           Record validation and paired comparisons
results/
  metrics/                      Aggregate JSON run records
  predictions/                  Validation CSVs and matching metadata
tests/                          Offline preparation and experiment checks
```

## Data and experiment flow

1. `data_preparation.ipynb` creates the prepared cache. Part 1 explores application data and defines random/stratified 70/15/15 splits. Part 2 produces cleaned `application_only` and `application_history` feature sets with the same applicant assignments.
2. The model notebooks load one feature set, apply model-specific preprocessing, fit one configuration, and export validation results. `FEATURE_SET` selects the variant; the default split is stratified.
3. `modeling/model_comparison.ipynb` reads saved metric records, checks comparability, and reports scores, history-feature gains, training costs, and learning curves. It does not fit models.

History tables are aggregated to one row per applicant before joining applications. Applicants without history remain in the dataset. Numeric imputation, scaling, and category vocabularies are model-specific and learned from training rows only.

The initial configurations use seed 42 and four CPU threads. Validation ROC-AUC controls early stopping, while classification metrics use a fixed 0.5 threshold. These experiments do not include class balancing, threshold tuning, cross-validation, a non-random deployment split, or holdout/competition evaluation.

## Environment and cache

Preparation requires `gdown`, `pandas`, `numpy`, and `scikit-learn>=1.3`. Modeling additionally uses `torch`, `lightgbm`, or `catboost`; comparison uses `matplotlib`.

```text
python -m pip install gdown pandas numpy "scikit-learn>=1.3" torch lightgbm catboost matplotlib ipykernel
```

Modeling notebooks resolve imports and result paths from either the repository root or `modeling/`. Data preparation runs from the repository root. The R baseline uses `tidyverse` and `googledrive`.

Raw CSVs are downloaded from the shared Drive into a user-local cache, not the repository. On Windows this is `LOCALAPPDATA/data4401-project2/home-credit-default-risk`. `HOME_CREDIT_DATA_DIR` overrides the raw-data location. Prepared inputs are stored in the sibling `prepared/classification_inputs_v2.pkl`; each machine generates its own cache. Only trusted, locally generated pickle files should be loaded.

## Result artifacts

Aggregate JSON records contain metrics, timings, parameters, learning history, library versions, and split/feature fingerprints. Comparisons select the latest run per model/feature set, not the highest-scoring rerun. Multiple split cohorts require explicit selection.

Validation CSVs contain `SK_ID_CURR`, `TARGET`, probability, predicted label, error type, gender, age, age group, and run ID. Matching metadata identifies the run, threshold, schemas, and checksum. These artifacts support downstream error/fairness analysis without training; additional local features can be joined on applicant ID. No training, holdout, competition rows, or model weights are exported. Applicant-level artifacts should remain team-only, subject to dataset sharing rules.

The R baseline is preserved unchanged in `modeling/` and has no automatic Python-metrics export. Its applicant split and preprocessing differ from the Python experiments; any separately supplied baseline record is treated as an unpaired reference. Missing records are shown explicitly rather than estimated.

## Checks

```text
python -m unittest discover -s tests -v
Rscript tests/test_baseline.R
```

Tests use synthetic data and mock training orchestration; they do not download data or fit the full models. The R check parses baseline code without executing it.
