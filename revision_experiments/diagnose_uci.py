import run_revision as r
import numpy as np
import pandas as pd
from dataclasses import replace
from threadpoolctl import threadpool_limits

def main():
    r.activate(False,2)
    spec=replace(r.common.SPECS['uci_ecl'],calibration_days=84,folds=5)
    frame=r.common.feature_frame(r.common.load_hourly(spec),spec.target,12)
    output=r.OUT/'uci_diagnosis';output.mkdir(exist_ok=True)
    metrics=[]
    for fold,train,cal,test in r.common.chronological_folds(frame,spec,12):
        p=r.core.prepare_fold(spec,12,fold,train,cal,test)
        n=p['fit_end'];costs=p['costs']
        model=r.depth.train_stationary_fqi(p['cal_states'][:n],p['cal_losses'][:n],p['calibration'].index[:n],costs,108,0.,1,r.common.SEED)
        for label,a,b in [('fit',0,n),('tune',p['tune_start'],p['tune_end']),('test',0,len(p['test']))]:
            is_test=label=='test'
            states=(p['test_states'] if is_test else p['cal_states'])[a:b]
            df=(p['test'] if is_test else p['calibration']).iloc[a:b]
            preds={k:v[a:b] for k,v in (p['test_pred'] if is_test else p['cal_pred']).items()}
            pred,actions,remaining,stage=r.depth.deploy(model,states,preds,costs,108,df.index)
            trace=pd.DataFrame({'timestamp':df.index,'action':[r.BASE[i] for i in actions]})
            for i,name in enumerate(r.BASE):
                est=np.array([-model.predict(r.depth.action_rows(s,(rem+costs[act])/108))[i]*p['scale'] for s,rem,act in zip(states,remaining,actions)])
                ae=np.abs(df.target.to_numpy()-preds[name])
                trace[name+'_predicted_loss']=est; trace[name+'_realized_loss']=ae
                metrics.append({'fold':fold,'window':label,'expert':name,'selection_share':np.mean(actions==i),
                  'mean_predicted_loss':est.mean(),'mean_actual_loss':ae.mean(),
                  'loss_prediction_mae':np.abs(est-ae).mean()})
            trace.to_csv(output/f'fold{fold}_{label}_loss_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(output/'loss_calibration.csv',index=False)
    print(pd.DataFrame(metrics).query('fold == 3').to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
