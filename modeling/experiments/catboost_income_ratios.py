
from pathlib import Path
import sys
import json
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from scripts.project_data import load_prepared_data, get_partitions
from scripts.experiment_utils import validation_metrics, split_fingerprint, feature_schema_fingerprint, package_versions


def add_ratios(frame):
    result = frame.copy()
    income = result['AMT_INCOME_TOTAL'].where(result['AMT_INCOME_TOTAL'].gt(0))
    for source, name in [('AMT_CREDIT', 'APP_CREDIT_TO_INCOME'), ('AMT_ANNUITY', 'APP_ANNUITY_TO_INCOME')]:
        result[name] = result[source].div(income).replace([np.inf, -np.inf], np.nan)
    return result


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    valid = (thresholds > 0) & (thresholds < 1)
    # In an exact tie choose the higher threshold, favoring fewer alerts.
    best = np.flatnonzero(valid & np.isclose(f1, f1[valid].max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def main():
    from catboost import CatBoostClassifier
    review = ROOT / 'results/error_analysis/review'
    output = ROOT / 'results/experiments' / 'catboost_income_ratios'
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_prepared_data()
    data = get_partitions(bundle, 'application_history', 'stratified', partitions=('train', 'validation'))
    baseline_file = next((ROOT / 'results/metrics').glob('catboost*080453*.json'))
    historical = json.loads(baseline_file.read_text())
    assert split_fingerprint(bundle['split_ids']['stratified'], bundle['labels']) == historical['split_fingerprint']
    assert feature_schema_fingerprint(data['raw_inputs']['train'].columns) == historical['feature_schema_fingerprint']
    partition = pd.read_csv(review / 'validation_review_split.csv').set_index('SK_ID_CURR')['review_partition']
    ids = data['applicant_ids']['validation']
    assert partition.index.is_unique and set(partition.index) == set(ids)
    partition = partition.loc[ids]
    masks = {key: partition.eq(key).to_numpy() for key in ['eyeball', 'blackbox']}
    assert (masks['eyeball'] ^ masks['blackbox']).all()
    parameters = dict(historical['parameters'])
    configuration = {'hypothesis': 'Explicit application income ratios improve relative-burden representation.', 'threshold_objective': 'maximize eyeball F1; exploratory proxy, not a business cost objective', 'early_stopping_partition': 'eyeball', 'holdout_evaluated': False, 'parameters': parameters, 'versions': package_versions('catboost'), 'split_fingerprint': historical['split_fingerprint'], 'historical_baseline': historical['run_id'], 'caveat': 'Historical models used full validation for early stopping. This paired rerun uses eyeball only; blackbox is reserved within previously used validation and is not an untouched final holdout.'}
    (output / 'configuration.json').write_text(json.dumps(configuration, indent=2))
    ytrain = data['labels']['train']; yval = data['labels']['validation'].to_numpy()
    summaries=[]; predictions=[]; fairness=[]
    for variant in ['baseline_rerun', 'income_ratios']:
        print('Training',variant,flush=True)
        frames = {part: (add_ratios(f) if variant == 'income_ratios' else f.copy()) for part,f in data['raw_inputs'].items()}
        for f in frames.values():
            for col in data['categorical_features']:
                f[col] = f[col].astype('object').fillna('__MISSING__').astype(str)
        model = CatBoostClassifier(**parameters)
        started = perf_counter()
        model.fit(frames['train'], ytrain, cat_features=data['categorical_features'], eval_set=(frames['validation'].loc[masks['eyeball']], yval[masks['eyeball']]), verbose=100)
        seconds = perf_counter()-started
        model.save_model(str(output / (variant+'.cbm')))
        p = model.predict_proba(frames['validation'])[:,1]
        selected = choose_threshold(yval[masks['eyeball']],p[masks['eyeball']])
        predictions.append(pd.DataFrame({'SK_ID_CURR':ids,'TARGET':yval,'review_partition':partition.to_numpy(),'variant':variant,'probability':p}))
        for policy, threshold in [('fixed_0.5',.5),('eyeball_f1',selected)]:
            for part,mask in masks.items():
                metrics=validation_metrics(yval[mask],p[mask],threshold)
                counts=metrics.pop('confusion_matrix')
                summaries.append({'variant':variant,'policy':policy,'partition':part,'rows':int(mask.sum()),'training_seconds':seconds,'trees':model.tree_count_,**metrics,**counts})
            audit=data['audit_by_partition']['validation']
            for attribute in ['gender','age_group']:
                values=audit[attribute].astype('string').fillna('UNKNOWN').to_numpy()
                for group in sorted(set(values)):
                    mask=masks['blackbox'] & (values==group)
                    if not mask.any():continue
                    yy=yval[mask]; pred=p[mask]>=threshold
                    tp=int(((yy==1)&pred).sum());fn=int(((yy==1)&~pred).sum());fp=int(((yy==0)&pred).sum());tn=int(((yy==0)&~pred).sum())
                    fairness.append({'variant':variant,'policy':policy,'attribute':attribute,'group':group,'rows':int(mask.sum()),'actual_positives':tp+fn,'actual_negatives':tn+fp,'recall':tp/(tp+fn) if tp+fn else np.nan,'false_positive_rate':fp/(fp+tn) if fp+tn else np.nan,'tp':tp,'fn':fn,'fp':fp,'tn':tn})
        pd.DataFrame(summaries).to_csv(output/'comparison.csv',index=False)
        pd.concat(predictions).to_csv(output/'predictions.csv',index=False)
        pd.DataFrame(fairness).to_csv(output/'group_comparison.csv',index=False)
        print(variant,'complete;',seconds,'seconds; threshold',selected,flush=True)
    scores=pd.DataFrame(summaries)
    black=scores[scores.partition.eq('blackbox')]
    text='# CatBoost income-ratio experiment\n\n'+configuration['caveat']+'\n\nBoth variants use the same training applicants, parameters, seed, and eyeball-only early stopping. Only the two income-ratio features differ. F1 thresholds were selected on eyeball rows, separately per variant, before blackbox evaluation. F1 is an exploratory selection objective; it does not encode lending costs. Compare fixed-0.5 rows to isolate feature effects and threshold-policy rows to assess the combined decision rule.\n\n'+black.to_markdown(index=False)+'\n\nGroup counts, recall, and false-positive rates are in group_comparison.csv; small groups can be unstable. No final holdout evaluation was performed. Historical results are references, not the paired control. Model files are saved separately in this experiment directory.\n'
    (output/'report.md').write_text(text)
    print('OUTPUT',output,flush=True)

if __name__=='__main__':main()
