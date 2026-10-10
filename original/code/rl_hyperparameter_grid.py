"""Pre-registered gamma-by-backup grid for representative 12-hour tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

import budget_rl_experiments as budget
import rl_core_matrix as core
import rl_depth_experiments as depth
import rl_router_experiments as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["steel", "tetouan"])
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--daily-budget", type=int, default=108)
    parser.add_argument(
        "--output", type=Path, default=Path("results/rl_hyper_grid_v1")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    gamma_values = (0.0, 0.3, 0.6, 0.9, 1.0)
    depths = (1, 3, 6, 12)
    for dataset in args.datasets:
        spec = replace(common.SPECS[dataset], calibration_days=84, folds=args.folds)
        hourly = common.load_hourly(spec)
        frame = common.feature_frame(hourly, spec.target, args.horizon)
        for fold, train, calibration, test in common.chronological_folds(
            frame, spec, args.horizon
        ):
            prepared = core.prepare_fold(
                spec, args.horizon, fold, train, calibration, test
            )
            fit_end = prepared["fit_end"]
            fit_index = prepared["calibration"].index[:fit_end]
            static_pred, _ = budget.static_budget_baseline(
                prepared["cal_losses"][
                    prepared["tune_start"]:prepared["tune_end"]
                ],
                prepared["test_pred"], prepared["costs"], args.daily_budget,
                prepared["test"].index,
            )
            static_mae = mean_absolute_error(prepared["y_test"], static_pred)
            fitted_cache = {}
            for gamma in gamma_values:
                for backups in depths:
                    effective_backups = 1 if gamma == 0 else backups
                    cache_key = (gamma, effective_backups)
                    if cache_key not in fitted_cache:
                        fitted_cache[cache_key] = depth.train_stationary_fqi(
                            prepared["cal_states"][:fit_end],
                            prepared["cal_losses"][:fit_end], fit_index,
                            prepared["costs"], args.daily_budget,
                            gamma=gamma, backups=effective_backups,
                            seed=depth.BASE_SEED,
                        )
                    model = fitted_cache[cache_key]
                    pred, actions, _, _ = depth.deploy(
                        model, prepared["test_states"], prepared["test_pred"],
                        prepared["costs"], args.daily_budget,
                        prepared["test"].index,
                    )
                    mae = mean_absolute_error(prepared["y_test"], pred)
                    daily_spend = pd.Series(
                        prepared["costs"][actions], index=prepared["test"].index
                    ).groupby(pd.DatetimeIndex(
                        prepared["test"].index
                    ).normalize()).sum()
                    rows.append({
                        "dataset": dataset, "fold": fold,
                        "horizon": args.horizon,
                        "daily_budget": args.daily_budget,
                        "gamma": gamma, "backups": backups,
                        "effective_backups": effective_backups,
                        "mae": mae, "static_mae": static_mae,
                        "gain_vs_static_pct": 100 * (static_mae - mae) / static_mae,
                        "max_daily_spend": int(daily_spend.max()),
                    })
            pd.DataFrame(rows).to_csv(args.output / "grid_metrics.csv", index=False)
            print(json.dumps({"dataset": dataset, "fold": fold}), flush=True)
    metrics = pd.DataFrame(rows)
    metrics.groupby(["gamma", "backups"]).agg(
        units=("fold", "size"),
        mean_gain_pct=("gain_vs_static_pct", "mean"),
        std_gain_pct=("gain_vs_static_pct", "std"),
        max_daily_spend=("max_daily_spend", "max"),
    ).reset_index().to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
