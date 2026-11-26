"""Benchmark uncached position fits without changing notebooks or saved model results."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'AI_Football_Project'
sys.path.insert(0,str(PROJECT))
from advanced_position_comparison import BASE_FEATURES,ADVANCED_FEATURES,CLASSES,model_pipeline,score,splits,targets


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_fit(kind,train,test,features,cap):
    model=model_pipeline(kind)
    if kind=='neural':model.set_params(mlpclassifier__max_iter=cap)
    if kind=='linear':
        x=train[features];_,y=targets(train);kwargs={}
    else:
        expanded=train.explode('pos_set')
        x=expanded[features];y=expanded.pos_set.map({name:i for i,name in enumerate(CLASSES)})
        kwargs={'logisticregression__sample_weight':1/train.pos_set.str.len().loc[expanded.index].to_numpy()} if kind=='logistic' else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        start=perf_counter();model.fit(x,y,**kwargs);fit_seconds=perf_counter()-start
        start=perf_counter()
        if kind=='linear':
            p=np.clip(model.predict(test[features]),0,None)
            total=p.sum(axis=1,keepdims=True)
            p=np.divide(p,total,out=np.full_like(p,1/3),where=total>0)
        else:p=model.predict_proba(test[features])
        predict_seconds=perf_counter()-start
    warning_count=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
    iterations=getattr(model[-1],'n_iter_',None)
    return p,fit_seconds,predict_seconds,warning_count,float(np.max(iterations)) if iterations is not None else np.nan


def main():
    result_root=PROJECT/'advanced_position_results'
    dataset=result_root/'comparison_dataset.csv'
    protected=[dataset,*PROJECT.glob('position_*.ipynb'),*result_root.glob('*/results.json')]
    before={str(p):digest(p) for p in protected}
    data=pd.read_csv(dataset)
    data['pos_set']=data.pos_set.map(ast.literal_eval)
    folds,holdout=splits(data)
    configurations=[(kind,arm,cap) for kind in ['linear','logistic','neural']
                    for cap in ([500,1500] if kind=='neural' else [None])
                    for arm in ['Baseline','Baseline + advanced']]
    rows=[];details=[]
    with threadpool_limits(limits=1):
        # Rotate execution order to reduce first-run/order effects; every fit is uncached.
        for repeat in range(3):
            order=configurations[repeat:]+configurations[:repeat]
            for kind,arm,cap in order:
                features=BASE_FEATURES+(ADVANCED_FEATURES if arm!='Baseline' else [])
                oof=np.full((len(data),3),np.nan);fit_total=0;predict_total=0;warn_total=0;iters=[]
                started=perf_counter()
                for fold,(train,test) in enumerate([*folds,holdout],1):
                    p,fit_time,predict_time,warn,n_iter=one_fit(kind,data.iloc[train],data.iloc[test],features,cap)
                    fit_total+=fit_time;predict_total+=predict_time;warn_total+=warn
                    if not np.isnan(n_iter):iters.append(n_iter)
                    if fold<=5:oof[test]=p
                    else:hold_metrics=score(data.iloc[test],p)
                    details.append({'repeat':repeat+1,'model':kind,'arm':arm,'max_iter':cap,'fold':fold,
                                    'fit_seconds':fit_time,'prediction_seconds':predict_time,
                                    'iterations':n_iter,'convergence_warnings':warn})
                elapsed=perf_counter()-started
                metrics=score(data,oof)
                if kind!='neural' or cap==1500:
                    saved=json.loads((result_root/kind/'results.json').read_text())['arms'][arm]
                    for metric in metrics:np.testing.assert_allclose(metrics[metric],saved['cv'][metric],atol=1e-12)
                    for metric in hold_metrics:np.testing.assert_allclose(hold_metrics[metric],saved['holdout'][metric],atol=1e-12)
                rows.append({'repeat':repeat+1,'model':kind,'arm':arm,'max_iter':cap,
                             'features':len(features),'fit_seconds_six_fits':fit_total,
                             'prediction_seconds_six_fits':predict_total,'evaluation_seconds':elapsed,
                             'cv_accuracy':metrics['any_position_accuracy'],
                             'first_listed_cv_accuracy':metrics['primary_accuracy'],
                             'single_position_cv_accuracy':metrics['single_position_accuracy'],
                             'holdout_accuracy':hold_metrics['any_position_accuracy'],
                             'convergence_warnings_six_fits':warn_total,
                             'mean_actual_iterations':float(np.mean(iters)) if iters else np.nan})
                print(f'Repeat {repeat+1}: {kind}, {arm}, cap={cap}; CV={metrics["any_position_accuracy"]:.3%}; fit={fit_total:.3f}s; total={elapsed:.3f}s',flush=True)
    frame=pd.DataFrame(rows)
    frame['max_iter']=frame.max_iter.fillna(0).astype(int)
    summary=frame.groupby(['model','arm','max_iter'],sort=False).agg(
        features=('features','first'),cv_accuracy=('cv_accuracy','first'),holdout_accuracy=('holdout_accuracy','first'),
        first_listed_cv_accuracy=('first_listed_cv_accuracy','first'),single_position_cv_accuracy=('single_position_cv_accuracy','first'),
        median_fit_seconds=('fit_seconds_six_fits','median'),median_predict_seconds=('prediction_seconds_six_fits','median'),
        median_evaluation_seconds=('evaluation_seconds','median'),
        convergence_warnings_per_repeat=('convergence_warnings_six_fits','max'),
        mean_actual_iterations=('mean_actual_iterations','mean')).reset_index()
    after={str(p):digest(p) for p in protected}
    assert before==after,'A protected notebook/data/result file changed.'
    folder=result_root/('evolution_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir()
    frame.to_csv(folder/'timing_repeats.csv',index=False)
    pd.DataFrame(details).to_csv(folder/'fit_details.csv',index=False)
    summary.to_csv(folder/'summary.csv',index=False)
    try:cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):cpu=platform.processor()
    (folder/'manifest.json').write_text(json.dumps({
        'measured_at_utc':datetime.now(timezone.utc).isoformat(),'cpu':cpu,'platform':platform.platform(),
        'python':platform.python_version(),'logical_cpus':os.cpu_count(),'numerical_library_thread_limit':1,
        'repeats':3,'records':len(data),'unique_players':int(data.source_player_id.nunique()),
        'timing_scope':'Uncached five cross-validation fits plus one 20% group-holdout fit; no downloading, matching, plotting, imports or cached-result loading.',
        'evaluation_time_includes':'Model creation, training-label expansion, six pipeline fits, prediction, normalization and scoring.',
        'protected_file_sha256':before,'protected_files_unchanged':True,
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']},
        'benchmark_script_sha256':digest(Path(__file__)),
    },indent=2))
    print('\n'+summary.to_string(index=False),flush=True)
    print('BENCHMARK_OUTPUT',folder,flush=True)


if __name__=='__main__':main()
