"""Summarize a completed paired run; never trains or changes models."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main(directory):
    scores = pd.read_csv(directory / 'comparison.csv')
    pred = pd.read_csv(directory / 'predictions.csv')
    groups = pd.read_csv(directory / 'group_comparison.csv')
    black = pred[pred.review_partition.eq('blackbox')]
    wide = black.pivot(index='SK_ID_CURR', columns='variant', values='probability')
    y = black.drop_duplicates('SK_ID_CURR').set_index('SK_ID_CURR').loc[wide.index, 'TARGET'].to_numpy()
    rng = np.random.default_rng(42)
    positives=np.flatnonzero(y==1);negatives=np.flatnonzero(y==0)
    differences=[]
    for _ in range(200):
        idx=np.r_[rng.choice(positives,len(positives),replace=True),rng.choice(negatives,len(negatives),replace=True)]
        differences.append(roc_auc_score(y[idx],wide['income_ratios'].to_numpy()[idx])-roc_auc_score(y[idx],wide['baseline_rerun'].to_numpy()[idx]))
    low, high = np.quantile(differences,[.025,.975])
    fixed=scores[scores.partition.eq('blackbox') & scores.policy.eq('fixed_0.5')].set_index('variant')
    before=fixed.loc['baseline_rerun'];after=fixed.loc['income_ratios']
    delta=float(after.roc_auc-before.roc_auc)
    decision=('Ratios show a positive AUC difference with a bootstrap interval above zero; retain as a candidate pending independent holdout evaluation.' if low>0 else 'No clear AUC improvement established; retain the original model as the reference and do not promote the ratio variant based on this run.')
    evidence={'auc_difference':delta,'bootstrap_95_percent_interval':[float(low),float(high)],'bootstrap_replicates':200,'seed':42,'method':'paired applicant bootstrap stratified by outcome; conditional on fitted models, does not capture training variability','decision':decision}
    (directory/'uncertainty.json').write_text(json.dumps(evidence,indent=2)+'\n')
    report=(directory/'report.md').read_text()
    report+='\n## Decision\n\n'+decision+f' AUC change: {delta:+.5f}; paired stratified bootstrap 95% interval [{low:+.5f}, {high:+.5f}] (200 resamples, seed 42). This interval describes applicant sampling conditional on these fitted models and excludes training variability.\n'
    report+='\n## Readable blackbox comparison\n\n'+scores[scores.partition.eq('blackbox')][['variant','policy','threshold','roc_auc','average_precision','precision','recall','f1','fp','fn']].to_markdown(index=False)+'\n'
    report+='\n## Group review\n\nThe following recall and false-positive-rate ranges cover groups with at least 30 actual positives and 30 actual negatives. Smaller groups remain in the detailed CSV but are excluded from these ranges. Differences describe these data and do not establish equal performance or causation.\n\n'
    summaries=[]
    for (variant,policy,attribute),g in groups.groupby(['variant','policy','attribute']):
        eligible=g[(g.actual_positives>=30)&(g.actual_negatives>=30)]
        summaries.append({'variant':variant,'policy':policy,'attribute':attribute,'eligible_groups':len(eligible),'excluded_small_groups':len(g)-len(eligible),'recall_min':eligible.recall.min(),'recall_max':eligible.recall.max(),'fpr_min':eligible.false_positive_rate.min(),'fpr_max':eligible.false_positive_rate.max()})
    summary=pd.DataFrame(summaries)
    summary.to_csv(directory/'group_summary.csv',index=False)
    report+=summary.to_markdown(index=False)+'\n'
    (directory/'report.md').write_text(report)
    root=Path(__file__).resolve().parents[2]
    logpath=root/'results/error_analysis/review/experiment_log.csv'
    log=pd.read_csv(logpath,dtype=str).fillna('')
    mask=log.model.eq('catboost')
    for key,value in {'baseline_run_id':directory.name+'/baseline_rerun','change_tested':'Added APP_CREDIT_TO_INCOME and APP_ANNUITY_TO_INCOME; paired reruns with identical parameters and eyeball-only early stopping','new_run_id':directory.name+'/income_ratios','evaluation_partition':'blackbox (previously used validation; not final holdout)','before_roc_auc':before.roc_auc,'after_roc_auc':after.roc_auc,'before_recall':before.recall,'after_recall':after.recall,'before_f1':before.f1,'after_f1':after.f1,'before_threshold':.5,'after_threshold':.5,'fairness_findings':'See '+str(directory.relative_to(root))+'/group_summary.csv and group_comparison.csv; descriptive age/gender slices, small groups flagged','improved':'inconclusive' if low<=0<=high else ('yes_auc' if low>0 else 'no_auc'),'decision':decision+' Separate eyeball-F1 threshold policies are compared in report.md; no original notebook or holdout changed.'}.items():log.loc[mask,key]=str(value)
    log.to_csv(logpath,index=False)
    print(scores[scores.partition.eq('blackbox')][['variant','policy','threshold','roc_auc','precision','recall','f1','fp','fn']].to_string(index=False))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
