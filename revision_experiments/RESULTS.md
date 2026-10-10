# Supplementary experiment results

Post-review, fixed h=12 and B=108 analysis. Positive MAE change means deterioration. No test-dependent configuration selection.

```
                    units  four_pool_gain  five_pool_gain  mae_change_pct  mlp_share
dataset algorithm                                                                   
opsd    contextual      5       19.264906       18.649570       -8.503137   0.470988
        fqi             5       16.740154       14.668681       -6.547664   0.466667
steel   contextual      5       18.152801       18.799963       -4.161922   0.208974
        fqi             5       13.740263       10.262697        1.320780   0.108974
tetouan contextual      5       18.444441       16.613090        1.711560   0.190385
        fqi             5       17.233209       18.190359       -1.282480   0.162179
uci_ecl contextual      5      -18.889325      -27.151406        8.903190   0.168827
        fqi             5      -16.769709      -32.905877       15.699281   0.252778
```

## UCI-ECL four-action expert errors

```
expert                          daily        hgb       ridge      weekly
unit          window                                                    
uci_ecl_fold1 fit           7833.8399  7254.2953   7914.0786  22283.5796
              gate_unused   8906.5587  7717.3844   8595.4227  12745.2915
              test          8438.7615  8612.8404   8251.2568  13276.8089
              tune         10696.5201  8719.0545  10495.8861  20109.1963
uci_ecl_fold2 fit           9193.4018  7292.0585   8884.9330  18618.3022
              gate_unused   8302.7511  7667.9871   8092.8830  12907.3123
              test          7626.2430  7571.4104   7231.8012  12721.4667
              tune          8954.3507  9037.6348   8759.3308  13098.6490
uci_ecl_fold3 fit           9430.6398  8158.7205   9037.0101  13102.1386
              gate_unused   7826.1107  6776.4603   7473.1690  12981.8275
              test          9648.9996  9026.9087   9359.0437  19257.6040
              tune          7925.9105  8127.5564   7341.8663  11903.8248
uci_ecl_fold4 fit           7836.1136  7871.0699   7812.0761  14310.4109
              gate_unused   9786.4910  9129.6370   9719.2510  18494.4307
              test          6368.7868  6077.3995   5976.2625  10357.0592
              tune          9078.1424  7452.1735   8108.4078  11776.9259
uci_ecl_fold5 fit           8038.9591  7811.6365   7680.1056  14730.9518
              gate_unused   5493.3201  5162.0546   5383.5432   6931.9987
              test         11150.2015  8203.6276  11107.0508   9040.0057
              tune         10111.1544  8772.6225   9533.9610  20070.3352
```

## Audit

Policy rows: 80; paired rows: 40; maximum daily spend: 108.
Each completed fold has replication_check.json comparing the four-action rerun against the frozen submission outputs.
Timing-derived MLP cost and environment: results/frozen_mlp_cost.json. See per-fold mlp_fit.json for convergence warnings.
Interpret these as conditional pool-extension and descriptive temporal diagnostics, not causal identification or prospective generalization evidence.