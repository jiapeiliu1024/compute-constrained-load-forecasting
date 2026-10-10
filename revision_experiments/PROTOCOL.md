# Revision supplementary analyses (2026-09-22)

These are post-review analyses: the original test results have already been inspected. No claim of prospective preregistration is made. Original submission sources and frozen results remain untouched.

## Fixed design

The four base-expert charges used by the formal study are experiment-defined credits, not a rounding or linear normalization of measured wall-clock time: persistence is assigned 1, Ridge 3, and HGB 18. The archived prototype timing profile is descriptive and implementation-specific. A separate sensitivity refit changes the HGB charge from 18 to 9 while leaving all other charges fixed; this is one alternative schedule, not a general platform-robustness test.

- Four datasets: Steel, Tetouan, UCI-ECL, OPSD; horizon 12 h; budget 108; five original chronological folds each.
- Compare four-action and five-action pools using contextual (gamma=0, one backup) and stationary FQI (gamma=1, six backups), with the original seeds, trees, state, embargoes and complete-day rules.
- Neural forecasting action: median imputation, input standardization, target standardization, MLP hidden layers (32,16), ReLU, Adam, alpha=0.001, learning rate=0.001, batch=128, 150 epochs, shuffle=False, early_stopping=False, seed=20260821. All training uses the expanding expert-training block. No test tuning. Record convergence warnings rather than select a better run.
- Freeze the new credit before policy evaluation using ONLY the first Steel fold's expert-training data. Time 21 warm predictions on the last 336 training rows. New credit = max(1, ceil(18 * median MLP time / median HGB time)). Retain existing (1,1,3,18) credits. This is a relative experimental extension anchored to HGB, not a physical latency or energy guarantee. Archive the measurement, single-row timings, environment and rule.
- Refit both policies and reselect feasible static comparators separately for both pools; recompute feasible oracle. Report action shares, per-origin and per-day losses, mean costs, hard-budget compliance, and paired changes in native MAE. Also compare five-pool predictions against the original four-pool static baseline.
- First verify the four-pool rerun against archived h=12,B=108 results. Any material discrepancy must be reported and investigated before interpreting the extension.

## UCI-ECL diagnosis

For all five h=12 folds, retain each expert's predictions and errors over policy-fit, static-tune, gate-unused and test windows. Record input/target summaries, feature-range exceedances relative to policy-fit, daily errors, action identity and dynamic-versus-static excess loss. This is descriptive diagnosis; do not attribute causality to weather omission or distribution shift from aggregate comparisons alone.

## Conditional follow-up

An exogenous-variable ablation and expansion of the switching-state audit are optional follow-ups, not necessary components of this first supplementary run. Decide based on whether the above diagnostics resolve the reviewer's question; document any later design separately.
