from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'
checked=0
for root,expected,credit in [(OUT,80,2),(OUT/'cost9',40,9)]:
    allrows=[]
    for file in root.glob('*/metrics.csv'):
        rows=pd.read_csv(file); allrows.append(rows)
        for row in rows.itertuples():
            trace=pd.read_csv(file.parent/f'pool{row.pool}_{row.algorithm}_trace.csv')
            assert np.allclose(np.abs(trace.target-trace.prediction),trace.absolute_error)
            assert np.isclose(trace.absolute_error.mean(),row.mae)
            assert np.isclose(trace.static_error.mean(),row.static_mae)
            assert np.allclose(np.abs(trace.target-trace.static_prediction),trace.static_error)
            dates=pd.to_datetime(trace.timestamp).dt.date
            assert trace.groupby(dates).size().eq(24).all()
            assert trace.groupby(dates).cost.sum().le(108).all()
            expected_cost=trace.action.map({'daily':1,'weekly':1,'ridge':3,'hgb':18,'mlp':credit})
            assert np.array_equal(trace.cost,expected_cost)
            assert np.array_equal(trace.remaining,108-trace.groupby(dates).cost.cumsum())
            assert row.oracle_mae<=min(row.mae,row.static_mae)+1e-7
            shares=[getattr(row,f'share_{n}') for n in ('daily','weekly','ridge','hgb')]+([row.share_mlp] if row.pool==5 else [])
            assert np.isclose(sum(shares),1)
            checked+=1
    df=pd.concat(allrows); assert len(df)==expected
    assert not df.duplicated(['dataset','fold','pool','algorithm']).any()
rep=[v for p in OUT.glob('*/replication_check.json') for v in json.loads(p.read_text())]
assert len(rep)==40
maxdiff=max(abs(v['mae_delta']) for v in rep); assert maxdiff<1e-6
result={'policy_evaluations':checked,'primary_evaluations':80,'cost_sensitivity_evaluations':40,
 'original_replication_comparisons':40,'max_replication_mae_difference':maxdiff,
 'budget_violations':0,'checks':'per-origin errors, static errors, action costs, cumulative reserve, 24-hour days, shares, oracle ordering, matrix completeness',
 'mlp_convergence_warnings':sum(bool(json.loads(p.read_text())['warnings']) for p in OUT.glob('*/mlp_fit.json'))}
(OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2))
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='MANIFEST_SHA256.txt']
(ROOT/'MANIFEST_SHA256.txt').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)) for p in sorted(files)),encoding='utf-8')
print(json.dumps(result,indent=2))
