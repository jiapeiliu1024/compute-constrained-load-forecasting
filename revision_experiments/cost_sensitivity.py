import run_revision as r
import pandas as pd
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits

def main():
    out=r.OUT/'cost9';out.mkdir(exist_ok=True)
    for dataset in ('steel','uci_ecl','tetouan','opsd'):
        spec=replace(r.common.SPECS[dataset],calibration_days=84,folds=5)
        frame=r.common.feature_frame(r.common.load_hourly(spec),spec.target,12)
        for fold,train,cal,test in r.common.chronological_folds(frame,spec,12):
            folder=out/f'{dataset}_fold{fold}';folder.mkdir(exist_ok=True)
            if (folder/'metrics.csv').exists():continue
            r.activate(False,9)
            p=r.core.prepare_fold(spec,12,fold,train,cal,test)
            saved=r.OUT/f'{dataset}_fold{fold}'
            values=np.full(len(p['calibration']),np.nan)
            for window,a,b in [('fit',0,p['fit_end']),('tune',p['tune_start'],p['tune_end'])]:
                df=pd.read_csv(saved/f'experts_{window}.csv')
                assert len(df)==b-a
                values[a:b]=df.mlp.to_numpy()
            p['cal_pred']['mlp']=values
            p['test_pred']['mlp']=pd.read_csv(saved/'experts_test.csv').mlp.to_numpy()
            pd.DataFrame(r.run_policy(p,True,9,folder)).to_csv(folder/'metrics.csv',index=False)
    metrics=pd.concat([pd.read_csv(p) for p in out.glob('*/metrics.csv')])
    assert len(metrics)==40
    metrics.to_csv(out/'all_metrics.csv',index=False)
    print(metrics.groupby(['dataset','algorithm']).gain_pct.mean().to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
