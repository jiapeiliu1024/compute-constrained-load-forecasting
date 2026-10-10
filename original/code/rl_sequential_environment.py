"""Focused finite-horizon and endogenous switching-state RL audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import budget_rl_experiments as budget
import rl_core_matrix as core
import rl_depth_experiments as depth
import rl_router_experiments as common


def evaluate(prepared: dict, daily_budget: int, switching_penalty: float):
    states = prepared["cal_states"][:prepared["fit_end"]]
    losses = prepared["cal_losses"][:prepared["fit_end"]]
    index = prepared["calibration"].index[:prepared["fit_end"]]
    costs = prepared["costs"]
    models = {
        "contextual": (depth.train_stationary_fqi(
            states, losses, index, costs, daily_budget,
            gamma=0.0, backups=1, seed=depth.BASE_SEED,
        ), 0.0),
        "stationary_fqi": (depth.train_stationary_fqi(
            states, losses, index, costs, daily_budget,
            gamma=1.0, backups=6, seed=depth.BASE_SEED,
        ), 0.0),
        "finite_horizon_fqi": (depth.train_stagewise_fqi(
            states, losses, index, costs, daily_budget, depth.BASE_SEED,
        ), 0.0),
        "switch_contextual": (depth.train_stationary_fqi(
            states, losses, index, costs, daily_budget,
            gamma=0.0, backups=1, seed=depth.BASE_SEED,
            switching_penalty=switching_penalty,
        ), switching_penalty),
        "switch_fqi": (depth.train_stationary_fqi(
            states, losses, index, costs, daily_budget,
            gamma=1.0, backups=6, seed=depth.BASE_SEED,
            switching_penalty=switching_penalty,
        ), switching_penalty),
    }
    rows = []
    traces = []
    for algorithm, (model, penalty) in models.items():
        pred, actions, remaining, stage = depth.deploy(
            model, prepared["test_states"], prepared["test_pred"], costs,
            daily_budget, prepared["test"].index,
            switching_penalty=penalty,
        )
        objective = depth.objective(
            prepared["y_test"], pred, actions, prepared["scale"],
            penalty, prepared["test"].index,
        )
        dates = pd.DatetimeIndex(prepared["test"].index).normalize().to_numpy()
        switches = np.zeros(len(actions), dtype=bool)
        switches[1:] = (actions[1:] != actions[:-1]) & (dates[1:] == dates[:-1])
        rows.append({
            "dataset": prepared["dataset"],
            "horizon": prepared["horizon"],
            "fold": prepared["fold"],
            "daily_budget": daily_budget,
            "algorithm": algorithm,
            "switching_penalty": penalty,
            "mae": mean_absolute_error(prepared["y_test"], pred),
            "normalized_objective": objective,
            "switch_rate": float(np.mean(switches)),
            "mean_cost": float(np.mean(costs[actions])),
            "max_daily_spend": int(pd.Series(
                costs[actions], index=prepared["test"].index
            ).groupby(dates).sum().max()),
        })
        traces.append(pd.DataFrame({
            "timestamp": prepared["test"].index,
            "dataset": prepared["dataset"],
            "fold": prepared["fold"],
            "algorithm": algorithm,
            "remaining_budget": remaining,
            "stage": stage,
            "action": [core.EXPERTS[action] for action in actions],
            "switch": switches.astype(int),
        }))
    return rows, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["steel", "tetouan"])
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--daily-budget", type=int, default=108)
    parser.add_argument("--switching-penalty", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path, default=Path("results/rl_sequential_audit_v1")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_traces = []
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
            rows, traces = evaluate(
                prepared, args.daily_budget, args.switching_penalty
            )
            all_rows.extend(rows)
            all_traces.extend(traces)
            pd.DataFrame(all_rows).to_csv(
                args.output / "policy_metrics.csv", index=False
            )
            pd.concat(all_traces, ignore_index=True).to_csv(
                args.output / "policy_traces.csv", index=False
            )
            print(json.dumps({"dataset": dataset, "fold": fold}), flush=True)
    metrics = pd.DataFrame(all_rows)
    metrics.groupby(["algorithm", "switching_penalty"]).agg(
        units=("fold", "size"),
        mean_mae=("mae", "mean"),
        mean_objective=("normalized_objective", "mean"),
        mean_switch_rate=("switch_rate", "mean"),
        max_daily_spend=("max_daily_spend", "max"),
    ).reset_index().to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
