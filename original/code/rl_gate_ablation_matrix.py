"""Disjoint calibration-gate ablation for the expanded contextual policy matrix."""

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


def completed(path: Path):
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return set(map(tuple, frame[["dataset", "horizon", "budget", "fold"]].to_numpy()))


def evaluate(prepared: dict, daily_budget: int):
    calibration = prepared["calibration"]
    fit_end = prepared["fit_end"]
    tune_start, gate_start = budget.calibration_day_boundaries(calibration.index)
    model = depth.train_stationary_fqi(
        prepared["cal_states"][:fit_end], prepared["cal_losses"][:fit_end],
        calibration.index[:fit_end], prepared["costs"], daily_budget,
        gamma=0.0, backups=1, seed=depth.BASE_SEED,
    )
    gate_predictions = {
        name: values[gate_start:] for name, values in prepared["cal_pred"].items()
    }
    dynamic_gate, _, _, _ = depth.deploy(
        model, prepared["cal_states"][gate_start:], gate_predictions,
        prepared["costs"], daily_budget, calibration.index[gate_start:],
    )
    static_gate, static_action = budget.static_budget_baseline(
        prepared["cal_losses"][tune_start:prepared["tune_end"]],
        gate_predictions, prepared["costs"], daily_budget,
        calibration.index[gate_start:],
    )
    y_gate = calibration.target.to_numpy(float)[gate_start:]
    paired_gate_loss = np.abs(y_gate - dynamic_gate) - np.abs(y_gate - static_gate)
    gate_upper = common.moving_block_bootstrap_upper(paired_gate_loss)
    gate_open = bool(gate_upper < 0)

    dynamic_test, actions, _, _ = depth.deploy(
        model, prepared["test_states"], prepared["test_pred"],
        prepared["costs"], daily_budget, prepared["test"].index,
    )
    static_test, static_action_test = budget.static_budget_baseline(
        prepared["cal_losses"][tune_start:prepared["tune_end"]],
        prepared["test_pred"], prepared["costs"], daily_budget,
        prepared["test"].index,
    )
    if static_action != static_action_test:
        raise AssertionError("Static action changed between gate and test")
    gated_test = dynamic_test if gate_open else static_test
    y_test = prepared["y_test"]
    dynamic_mae = mean_absolute_error(y_test, dynamic_test)
    static_mae = mean_absolute_error(y_test, static_test)
    gated_mae = mean_absolute_error(y_test, gated_test)
    daily_spend = pd.Series(
        prepared["costs"][actions], index=prepared["test"].index
    ).groupby(pd.DatetimeIndex(prepared["test"].index).normalize()).sum()
    dynamic_better = bool(dynamic_mae < static_mae)
    return {
        "dataset": prepared["dataset"], "horizon": prepared["horizon"],
        "budget": daily_budget, "fold": prepared["fold"],
        "gate_days": int(len(y_gate) / 24), "gate_upper": gate_upper,
        "gate_open": gate_open, "static_action": core.EXPERTS[static_action],
        "static_mae": static_mae, "dynamic_mae": dynamic_mae,
        "gated_mae": gated_mae,
        "dynamic_gain_pct": 100 * (static_mae - dynamic_mae) / static_mae,
        "gated_gain_pct": 100 * (static_mae - gated_mae) / static_mae,
        "dynamic_better": dynamic_better,
        "harmful_activation": bool(gate_open and not dynamic_better),
        "missed_gain": bool((not gate_open) and dynamic_better),
        "max_dynamic_daily_spend": int(daily_spend.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["steel", "tetouan", "uci_ecl", "opsd"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[6, 12, 24, 48])
    parser.add_argument("--budgets", nargs="+", type=int, default=[60, 108, 180])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/rl_gate_ablation_v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "gate_metrics.csv"
    done = completed(path)
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    for dataset in args.datasets:
        base = common.SPECS[dataset]
        calibration_days = base.calibration_days if dataset == "appliances" else 84
        spec = replace(base, calibration_days=calibration_days, folds=args.folds)
        hourly = common.load_hourly(spec)
        for horizon in args.horizons:
            frame = common.feature_frame(hourly, spec.target, horizon)
            for fold, train, calibration, test in common.chronological_folds(frame, spec, horizon):
                pending = [value for value in args.budgets if (dataset, horizon, value, fold) not in done]
                if not pending:
                    continue
                prepared = core.prepare_fold(spec, horizon, fold, train, calibration, test)
                for value in pending:
                    rows.append(evaluate(prepared, value))
                    pd.DataFrame(rows).to_csv(path, index=False)
                    print(json.dumps({"dataset": dataset, "horizon": horizon, "budget": value, "fold": fold}), flush=True)
    metrics = pd.DataFrame(rows)
    metrics.groupby(["dataset", "horizon", "budget"]).agg(
        folds=("fold", "nunique"),
        mean_dynamic_gain_pct=("dynamic_gain_pct", "mean"),
        mean_gated_gain_pct=("gated_gain_pct", "mean"),
        activation_rate=("gate_open", "mean"),
        harmful_activations=("harmful_activation", "sum"),
        missed_gains=("missed_gain", "sum"),
        max_daily_spend=("max_dynamic_daily_spend", "max"),
    ).reset_index().to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
