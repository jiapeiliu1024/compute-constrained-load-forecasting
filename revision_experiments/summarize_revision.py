"""Summarize completed paired experiments without selecting configurations."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
metrics=pd.concat([pd.read_csv(p) for p in OUT.glob('*/metrics.csv')],ignore_index=True)
metrics.to_csv(OUT/'all_metrics.csv',index=False)
pairs=[]
for (dataset,fold,algorithm),g in metrics.groupby(['dataset','fold','algorithm']):
    a=g.set_index('pool'); old=a.loc[4]; new=a.loc[5]
    pairs.append({'dataset':dataset,'fold':fold,'algorithm':algorithm,
      'four_pool_gain_pct':old.gain_pct,'five_pool_gain_pct':new.gain_pct,
      'mae_change_pct':100*(new.mae-old.mae)/old.mae,
      'five_pool_vs_four_static_pct':100*(old.static_mae-new.mae)/old.static_mae,
      'mlp_share':new.share_mlp,'old_static':old.static_action,'new_static':new.static_action})
pairs=pd.DataFrame(pairs); pairs.to_csv(OUT/'paired_pool_changes.csv',index=False)
summary=pairs.groupby(['dataset','algorithm']).agg(
    units=('fold','size'),four_pool_gain=('four_pool_gain_pct','mean'),
    five_pool_gain=('five_pool_gain_pct','mean'),mae_change_pct=('mae_change_pct','mean'),
    mlp_share=('mlp_share','mean'))
summary.to_csv(OUT/'paired_summary.csv')
expert=[]; daily=[]
for folder in OUT.glob('*_fold*'):
    for file in folder.glob('experts_*.csv'):
        df=pd.read_csv(file)
        for col in df:
            if col.endswith('_ae'):
                expert.append({'unit':folder.name,'window':file.stem.removeprefix('experts_'),
                  'expert':col[:-3],'mae':df[col].mean()})
    if folder.name.startswith('uci_ecl'):
        for file in folder.glob('pool4_*_daily.csv'):
            df=pd.read_csv(file); df['unit']=folder.name; df['algorithm']=file.stem.split('_')[1];daily.append(df)
expert=pd.DataFrame(expert); expert.to_csv(OUT/'expert_window_mae.csv',index=False)
if daily: pd.concat(daily).to_csv(OUT/'uci_daily_excess_loss.csv',index=False)
lines=['# Supplementary experiment results','',
 'Post-review, fixed h=12 and B=108 analysis. Positive MAE change means deterioration. No test-dependent configuration selection.','',
 '```',summary.to_string(), '```','', '## UCI-ECL four-action expert errors','']
uci=expert[expert.unit.str.startswith('uci_ecl') & expert.expert.ne('mlp')]
lines+=['```',uci.pivot(index=['unit','window'],columns='expert',values='mae').round(4).to_string(),'```']
lines+=['','## Audit','',f'Policy rows: {len(metrics)}; paired rows: {len(pairs)}; maximum daily spend: {metrics.max_daily_spend.max()}.',
 'Each completed fold has replication_check.json comparing the four-action rerun against the frozen submission outputs.',
 'Timing-derived MLP cost and environment: results/frozen_mlp_cost.json. See per-fold mlp_fit.json for convergence warnings.',
 'Interpret these as conditional pool-extension and descriptive temporal diagnostics, not causal identification or prospective generalization evidence.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(summary.to_string())
