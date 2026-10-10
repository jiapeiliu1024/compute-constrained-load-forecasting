"""Refit the frozen four-policy matrix for one HGB-cost sensitivity profile.

Primary profile: (1, 1, 3, 18). Sensitivity: (1, 1, 3, 9), halving HGB
credits while keeping all other costs, horizons, test folds, and budgets fixed.
Runs the exact rl_core_matrix training/evaluation implementation at h=12 and
B=108 on all four formal datasets and five chronological folds (20 paired
units). This is a focused sensitivity audit, not a replacement for the primary
240-unit matrix.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CODE = PROJECT / "supplement" / "original" / "code"
sys.path.insert(0, str(CODE))

import rl_core_matrix as core  # noqa: E402
import rl_router_experiments as common  # noqa: E402

HORIZON = 12
BUDGET = 108
SENSITIVITY_COSTS = (1, 1, 3, 9)
DATASETS = ("steel", "tetouan", "uci_ecl", "opsd")


def main() -> None:
    rows = []
    for dataset in DATASETS:
        base_spec = common.SPECS[dataset]
        spec = replace(base_spec, calibration_days=84, folds=5)
        hourly = common.load_hourly(spec)
        frame = common.feature_frame(hourly, spec.target, HORIZON)
        for fold, train, calibration, test in common.chronological_folds(frame, spec, HORIZON):
            prepared = core.prepare_fold(spec, HORIZON, fold, train, calibration, test)
            prepared["costs"] = pd.array(SENSITIVITY_COSTS, dtype="int64").to_numpy()
            unit_rows, _ = core.evaluate(prepared, BUDGET)
            for row in unit_rows:
                row.update({
                    "cost_profile": "1-1-3-9",
                    "cost_daily": 1,
                    "cost_weekly": 1,
                    "cost_ridge": 3,
                    "cost_hgb": 9,
                })
                if row["max_daily_spend"] > BUDGET:
                    raise RuntimeError(f"Budget violation in {dataset} fold {fold}: {row}")
                rows.append(row)
            print(json.dumps({"dataset": dataset, "fold": fold,
                              "algorithms": len(unit_rows)}, ensure_ascii=False), flush=True)

    sensitivity = pd.DataFrame(rows)
    result_dir = HERE / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_path = result_dir / "cost_vector_sensitivity_fold_metrics.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    baseline_path = (PROJECT / "supplement" / "original" / "results" /
                     "expanded_rl_summary" / "all_main_policy_metrics.csv")
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[(baseline["horizon"] == HORIZON) & (baseline["budget"] == BUDGET)]
    paired = sensitivity.merge(
        baseline[["dataset", "horizon", "budget", "fold", "algorithm",
                  "gain_vs_static_pct", "share_daily", "share_weekly", "share_ridge", "share_hgb"]],
        on=["dataset", "horizon", "budget", "fold", "algorithm"],
        how="inner", suffixes=("_alt", "_base"), validate="one_to_one",
    )
    if len(paired) != 80:
        raise RuntimeError(f"Expected 80 paired policy rows, found {len(paired)}")
    paired["gain_change_pp"] = paired["gain_vs_static_pct_alt"] - paired["gain_vs_static_pct_base"]
    summary = (paired.groupby(["dataset", "algorithm"], as_index=False)
               .agg(folds=("fold", "nunique"),
                    base_gain_pct=("gain_vs_static_pct_base", "mean"),
                    alt_gain_pct=("gain_vs_static_pct_alt", "mean"),
                    gain_change_pp=("gain_change_pp", "mean"),
                    base_hgb_share=("share_hgb_base", "mean"),
                    alt_hgb_share=("share_hgb_alt", "mean")))
    summary_path = result_dir / "cost_vector_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {len(sensitivity)} sensitivity rows and {len(paired)} paired rows.", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
