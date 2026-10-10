"""Fixed supplementary pool comparison and temporal failure diagnosis."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys, json, math, time, warnings, hashlib
from pathlib import Path
from dataclasses import replace
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'original'
sys.path.insert(0,str(SOURCE/'code'))
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import budget_rl_experiments as budget
import rl_core_matrix as core
import rl_depth_experiments as depth
import rl_router_experiments as common

OUT=ROOT/'revision_experiments/results'
OUT.mkdir(parents=True,exist_ok=True)
BASE=('daily','weekly','ridge','hgb')
ORIGINAL_POOL=budget.model_pool

def neural():
    return TransformedTargetRegressor(regressor=make_pipeline(
        SimpleImputer(strategy='median'),StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32,16),activation='relu',solver='adam',
        alpha=.001,learning_rate_init=.001,batch_size=128,max_iter=150,
        shuffle=False,early_stopping=False,random_state=common.SEED)),
        transformer=StandardScaler())

def activate(expanded,credit):
    experts=BASE+('mlp',) if expanded else BASE
    budget.EXPERTS=core.EXPERTS=depth.EXPERTS=experts
    budget.FROZEN_RELATIVE_COST['mlp']=credit
    return experts

def record_windows(p, train, calpred, testpred, folder):
    cal=p['calibration']; test=p['test']
    windows={'fit':(cal.iloc[:p['fit_end']],0,p['fit_end']),
      'tune':(cal.iloc[p['tune_start']:p['tune_end']],p['tune_start'],p['tune_end']),
      'gate_unused':(cal.iloc[p['tune_end']+24:],p['tune_end']+24,len(cal)),
      'test':(test,0,len(test))}
    summary=[]; fit=windows['fit'][0]
    for label,(frame,a,b) in windows.items():
        predictions=testpred if label=='test' else calpred
        trace=pd.DataFrame({'timestamp':frame.index,'target':frame.target.to_numpy()})
        for name,values in predictions.items():
            trace[name]=values[a:b]
            trace[name+'_ae']=np.abs(trace[name]-trace.target)
        trace.to_csv(folder/f'experts_{label}.csv',index=False)
        for col in frame.columns:
            if col in ('daily','weekly'): continue
            values=frame[col]
            summary.append({'window':label,'feature':col,'n':len(values),
              'mean':values.mean(),'std':values.std(),'min':values.min(),'max':values.max(),
              'outside_fit_range':((values<fit[col].min())|(values>fit[col].max())).mean()})
    pd.DataFrame(summary).to_csv(folder/'window_statistics.csv',index=False)

def run_policy(p,expanded,credit,folder):
    experts=activate(expanded,credit)
    costs=np.array([budget.FROZEN_RELATIVE_COST[n] for n in experts])
    fit=p['fit_end']; y=p['y_test']; test=p['test']
    losses=np.column_stack([np.abs(p['calibration'].target.to_numpy()-p['cal_pred'][n])/p['scale'] for n in experts])
    static,static_id=budget.static_budget_baseline(losses[p['tune_start']:p['tune_end']],p['test_pred'],costs,108,test.index)
    static_mae=np.abs(y-static).mean()
    oracle,_=budget.oracle_budget_mae(y,p['test_pred'],costs,108,test.index)
    rows=[]
    for algorithm,gamma,backups in [('contextual',0.,1),('fqi',1.,6)]:
        start=time.perf_counter()
        model=depth.train_stationary_fqi(p['cal_states'][:fit],losses[:fit],p['calibration'].index[:fit],costs,108,gamma,backups,common.SEED)
        pred,actions,remaining,stage=depth.deploy(model,p['test_states'],p['test_pred'],costs,108,test.index)
        trace=pd.DataFrame({'timestamp':test.index,'target':y,'prediction':pred,'static_prediction':static,
          'action':[experts[a] for a in actions],'cost':costs[actions],'remaining':remaining,
          'absolute_error':np.abs(y-pred),'static_error':np.abs(y-static)})
        trace['excess_error']=trace.absolute_error-trace.static_error
        daily=trace.groupby(pd.to_datetime(trace.timestamp).dt.date).agg(
          mae=('absolute_error','mean'),static_mae=('static_error','mean'),
          excess_error=('excess_error','mean'),spend=('cost','sum'))
        assert daily.spend.max()<=108 and np.min(remaining)>=0
        mae=np.abs(y-pred).mean()
        assert oracle<=min(mae,static_mae)+1e-7
        row={'dataset':p['dataset'],'fold':p['fold'],'pool':len(experts),'algorithm':algorithm,
          'mae':mae,'static_mae':static_mae,'static_action':experts[static_id],
          'gain_pct':100*(static_mae-mae)/static_mae,'oracle_mae':oracle,
          'max_daily_spend':int(daily.spend.max()),'mean_cost':costs[actions].mean(),
          'seconds':time.perf_counter()-start,**{f'share_{n}':np.mean(actions==i) for i,n in enumerate(experts)}}
        trace.to_csv(folder/f'pool{len(experts)}_{algorithm}_trace.csv',index=False)
        daily.to_csv(folder/f'pool{len(experts)}_{algorithm}_daily.csv')
        rows.append(row)
        print(json.dumps(row),flush=True)
    return rows

def main():
    os.chdir(SOURCE)
    for dataset in ('steel','uci_ecl','tetouan','opsd'):
        spec=replace(common.SPECS[dataset],calibration_days=84,folds=5)
        frame=common.feature_frame(common.load_hourly(spec),spec.target,12)
        for fold,train,cal,test in common.chronological_folds(frame,spec,12):
            folder=OUT/f'{dataset}_fold{fold}'; folder.mkdir(exist_ok=True)
            if (folder/'metrics.csv').exists(): continue
            activate(False,1)
            p=core.prepare_fold(spec,12,fold,train,cal,test)
            cols=[c for c in train if c not in ('target','daily','weekly')]
            mlp=neural(); start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); mlp.fit(train[cols],train.target)
            (folder/'mlp_fit.json').write_text(json.dumps({'seconds':time.perf_counter()-start,
              'epochs':mlp.regressor_[-1].n_iter_,'warnings':[str(w.message) for w in caught]},indent=2))
            costfile=OUT/'frozen_mlp_cost.json'
            if not costfile.exists():
                assert dataset=='steel' and fold==1
                hgb=ORIGINAL_POOL()['hgb'].fit(train[cols],train.target)
                sample=train[cols].iloc[-336:]
                profiles={n:budget.latency_profile(lambda m=m:m.predict(sample),len(sample)) for n,m in [('mlp',mlp),('hgb',hgb)]}
                credit=max(1,math.ceil(18*profiles['mlp']['median_ms_per_origin']/profiles['hgb']['median_ms_per_origin']))
                costfile.write_text(json.dumps({'credit':credit,'profiles':profiles,'python':sys.version,
                  'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
                  'single_row':{n:budget.latency_profile(lambda m=m:m.predict(sample.iloc[:1]),1) for n,m in [('mlp',mlp),('hgb',hgb)]}},indent=2))
            credit=json.loads(costfile.read_text())['credit']
            p['cal_pred']['mlp']=mlp.predict(p['calibration'][cols]); p['test_pred']['mlp']=mlp.predict(test.loc[p['test'].index,cols])
            record_windows(p,train,p['cal_pred'],p['test_pred'],folder)
            rows=run_policy(p,False,credit,folder)
            archived=pd.read_csv(SOURCE/f'results/rl_core_matrix_v2_{dataset}/policy_metrics.csv')
            diffs=[]
            for row in rows:
                a=archived[(archived.horizon==12)&(archived.budget==108)&(archived.fold==fold)&(archived.algorithm==row['algorithm'])].iloc[0]
                diffs.append({'algorithm':row['algorithm'],'mae_delta':row['mae']-a.mae,'static_delta':row['static_mae']-a.static_mae})
            (folder/'replication_check.json').write_text(json.dumps(diffs,indent=2))
            if any(abs(d['mae_delta'])>1e-6 or abs(d['static_delta'])>1e-6 for d in diffs):
                raise RuntimeError(f'Original result mismatch: {folder}; inspect replication_check.json')
            rows+=run_policy(p,True,credit,folder)
            pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    metrics=pd.concat([pd.read_csv(f) for f in OUT.glob('*/metrics.csv')],ignore_index=True)
    assert len(metrics)==80
    metrics.to_csv(OUT/'all_metrics.csv',index=False)
    metrics.groupby(['dataset','pool','algorithm']).agg(mean_gain_pct=('gain_pct','mean'),mean_mae=('mae','mean')).to_csv(OUT/'summary.csv')
    print('COMPLETE: 80 policy evaluations, 20 paired units',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
