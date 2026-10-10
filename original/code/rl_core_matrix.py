"""Resumable four-dataset, four-horizon, three-budget RL comparison matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import budget_rl_experiments as budget
import rl_depth_experiments as depth
import rl_router_experiments as common


EXPERTS = budget.EXPERTS
SEED = common.SEED


class MeanPredictor:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def predict(self, x):
        return 0.5 * (self.first.predict(x) + self.second.predict(x))


def neural_regressor(seed: int):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=256,
            learning_rate_init=2e-3,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=5,
            tol=1e-3,
            random_state=seed,
        ),
    )


def transition_table(states, losses, index, costs, daily_budget):
    x_rows, rewards, transitions = [], [], []
    remaining_after, next_is_new_day = budget.calendar_budget_geometry(index)
    for t in range(len(states) - 1):
        reserve = remaining_after[t] * int(np.min(costs))
        for fraction in (0.33, 0.66, 1.0):
            remaining = int(round(fraction * daily_budget))
            rows = depth.action_rows(states[t], fraction)
            for action in range(len(EXPERTS)):
                if costs[action] + reserve > remaining:
                    continue
                x_rows.append(rows[action])
                rewards.append(-losses[t, action])
                next_remaining = remaining - int(costs[action])
                if next_is_new_day[t]:
                    next_remaining = daily_budget
                transitions.append((t + 1, next_remaining, remaining_after[t + 1]))
    return np.asarray(x_rows), np.asarray(rewards), transitions


def future_values(model, states, transitions, costs, daily_budget, selector=None):
    if not transitions:
        return np.empty(0, dtype=float)
    next_t = np.fromiter((item[0] for item in transitions), dtype=int)
    remaining = np.fromiter((item[1] for item in transitions), dtype=int)
    later = np.fromiter((item[2] for item in transitions), dtype=int)
    action_blocks = np.stack([
        depth.action_rows(states[t], rem / daily_budget)
        for t, rem in zip(next_t, remaining)
    ])
    flat_rows = action_blocks.reshape(-1, action_blocks.shape[-1])
    evaluation = model.predict(flat_rows).reshape(len(transitions), len(EXPERTS))
    selection = (
        selector.predict(flat_rows).reshape(len(transitions), len(EXPERTS))
        if selector is not None else evaluation
    )
    reserve = later[:, None] * int(np.min(costs))
    feasible = costs[None, :] + reserve <= remaining[:, None]
    masked = np.where(feasible, selection, -np.inf)
    chosen = np.argmax(masked, axis=1)
    any_feasible = feasible.any(axis=1)
    future = np.zeros(len(transitions), dtype=float)
    rows = np.arange(len(transitions))[any_feasible]
    future[any_feasible] = evaluation[rows, chosen[any_feasible]]
    return future


def train_neural_fqi(states, losses, index, costs, daily_budget, backups=6):
    x, reward, transitions = transition_table(
        states, losses, index, costs, daily_budget
    )
    target = reward.copy()
    model = None
    for iteration in range(backups):
        model = neural_regressor(SEED + iteration).fit(x, target)
        target = reward + future_values(
            model, states, transitions, costs, daily_budget
        )
    return model


def train_double_fqi(states, losses, index, costs, daily_budget, backups=6):
    x, reward, transitions = transition_table(
        states, losses, index, costs, daily_budget
    )
    target_a = reward.copy()
    target_b = reward.copy()
    model_a = None
    model_b = None
    for iteration in range(backups):
        model_a = ExtraTreesRegressor(
            n_estimators=8, min_samples_leaf=10, max_features=0.75,
            n_jobs=1, random_state=SEED + 2 * iteration,
        ).fit(x, target_a)
        model_b = ExtraTreesRegressor(
            n_estimators=8, min_samples_leaf=10, max_features=0.75,
            n_jobs=1, random_state=SEED + 2 * iteration + 1,
        ).fit(x, target_b)
        target_a = reward + future_values(
            model_b, states, transitions, costs, daily_budget, selector=model_a
        )
        target_b = reward + future_values(
            model_a, states, transitions, costs, daily_budget, selector=model_b
        )
    return MeanPredictor(model_a, model_b)


def prepare_fold(spec, horizon, fold, train, calibration, test):
    calibration = budget.complete_calendar_days(calibration)
    test = budget.complete_calendar_days(test)
    columns, cal_pred, test_pred, _, _, relative = budget.predictions_and_costs(
        train, calibration, test
    )
    y_cal = calibration.target.to_numpy(float)
    y_test = test.target.to_numpy(float)
    scale = max(float(np.mean(np.abs(y_cal))), 1e-6)
    cal_losses = np.column_stack([
        np.abs(y_cal - cal_pred[name]) / scale for name in EXPERTS
    ])
    combined = pd.concat((calibration, test))
    states = depth.policy_state(combined, columns, "full")
    tune_start, gate_start = budget.calibration_day_boundaries(calibration.index)
    embargo_rows = int(np.ceil(horizon / 24)) * 24
    fit_end = tune_start - embargo_rows
    tune_end = gate_start - embargo_rows
    return {
        "dataset": spec.name,
        "horizon": horizon,
        "fold": fold,
        "calibration": calibration,
        "test": test,
        "y_test": y_test,
        "scale": scale,
        "cal_pred": cal_pred,
        "test_pred": test_pred,
        "cal_losses": cal_losses,
        "cal_states": states[:len(calibration)],
        "test_states": states[len(calibration):],
        "fit_end": fit_end,
        "tune_start": tune_start,
        "tune_end": tune_end,
        "costs": np.asarray([relative[name] for name in EXPERTS], dtype=int),
    }


def evaluate(prepared, daily_budget):
    fit_end = prepared["fit_end"]
    states = prepared["cal_states"][:fit_end]
    losses = prepared["cal_losses"][:fit_end]
    index = prepared["calibration"].index[:fit_end]
    costs = prepared["costs"]
    algorithms = {
        "contextual": depth.train_stationary_fqi(
            states, losses, index, costs, daily_budget,
            gamma=0.0, backups=1, seed=SEED,
        ),
        "fqi": depth.train_stationary_fqi(
            states, losses, index, costs, daily_budget,
            gamma=1.0, backups=6, seed=SEED,
        ),
        "double_fqi": train_double_fqi(
            states, losses, index, costs, daily_budget,
        ),
        "neural_fqi": train_neural_fqi(
            states, losses, index, costs, daily_budget,
        ),
    }
    static_pred, _ = budget.static_budget_baseline(
        prepared["cal_losses"][prepared["tune_start"]:prepared["tune_end"]],
        prepared["test_pred"], costs, daily_budget, prepared["test"].index,
    )
    static_mae = mean_absolute_error(prepared["y_test"], static_pred)
    oracle_mae, _ = budget.oracle_budget_mae(
        prepared["y_test"], prepared["test_pred"], costs,
        daily_budget, prepared["test"].index,
    )
    rows = []
    traces = []
    for name, model in algorithms.items():
        pred, actions, remaining, stage = depth.deploy(
            model, prepared["test_states"], prepared["test_pred"], costs,
            daily_budget, prepared["test"].index,
        )
        mae = mean_absolute_error(prepared["y_test"], pred)
        rows.append({
            "dataset": prepared["dataset"],
            "horizon": prepared["horizon"],
            "budget": daily_budget,
            "fold": prepared["fold"],
            "algorithm": name,
            "mae": mae,
            "static_mae": static_mae,
            "oracle_mae": oracle_mae,
            "gain_vs_static_pct": 100 * (static_mae - mae) / static_mae,
            "oracle_gap_pct": 100 * (mae - oracle_mae) / static_mae,
            "mean_cost": float(np.mean(costs[actions])),
            "switch_rate": float(np.mean(actions[1:] != actions[:-1])),
            "max_daily_spend": int(
                pd.Series(costs[actions], index=prepared["test"].index)
                .groupby(pd.DatetimeIndex(prepared["test"].index).normalize())
                .sum().max()
            ),
            **{
                f"share_{expert}": float(np.mean(actions == action))
                for action, expert in enumerate(EXPERTS)
            },
        })
        if prepared["fold"] == 1:
            traces.append(pd.DataFrame({
                "timestamp": prepared["test"].index,
                "dataset": prepared["dataset"],
                "horizon": prepared["horizon"],
                "budget": daily_budget,
                "algorithm": name,
                "stage": stage,
                "remaining_budget": remaining,
                "action": [EXPERTS[action] for action in actions],
                "absolute_error": np.abs(prepared["y_test"] - pred),
            }))
    return rows, traces


def completed_keys(path: Path):
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    counts = frame.groupby(["dataset", "horizon", "budget", "fold"]).size()
    return {tuple(key) for key, count in counts.items() if count == 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=["steel", "tetouan", "uci_ecl", "opsd"],
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[6, 12, 24, 48])
    parser.add_argument("--budgets", nargs="+", type=int, default=[60, 108, 180])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/rl_core_matrix_v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "policy_metrics.csv"
    traces_path = args.output / "policy_traces.csv"
    done = completed_keys(metrics_path)
    all_rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    trace_frames = [pd.read_csv(traces_path)] if traces_path.exists() else []
    for dataset in args.datasets:
        base_spec = common.SPECS[dataset]
        calibration_days = (
            base_spec.calibration_days if dataset == "appliances" else 84
        )
        spec = replace(
            base_spec, calibration_days=calibration_days, folds=args.folds
        )
        hourly = common.load_hourly(spec)
        for horizon in args.horizons:
            frame = common.feature_frame(hourly, spec.target, horizon)
            for fold, train, calibration, test in common.chronological_folds(
                frame, spec, horizon
            ):
                needed = [
                    budget_value for budget_value in args.budgets
                    if (dataset, horizon, budget_value, fold) not in done
                ]
                if not needed:
                    continue
                prepared = prepare_fold(
                    spec, horizon, fold, train, calibration, test
                )
                for budget_value in needed:
                    rows, traces = evaluate(prepared, budget_value)
                    all_rows.extend(rows)
                    trace_frames.extend(traces)
                    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
                    if trace_frames:
                        pd.concat(trace_frames, ignore_index=True).to_csv(
                            traces_path, index=False
                        )
                    print(json.dumps({
                        "dataset": dataset,
                        "horizon": horizon,
                        "budget": budget_value,
                        "fold": fold,
                        "algorithms": len(rows),
                    }), flush=True)
    metrics = pd.DataFrame(all_rows)
    summary = (
        metrics.groupby(["dataset", "horizon", "budget", "algorithm"])
        .agg(
            folds=("fold", "nunique"),
            mean_gain_pct=("gain_vs_static_pct", "mean"),
            mean_oracle_gap_pct=("oracle_gap_pct", "mean"),
            mean_cost=("mean_cost", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
            max_daily_spend=("max_daily_spend", "max"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "metadata.json").write_text(json.dumps({
        "datasets": args.datasets,
        "horizons": args.horizons,
        "budgets": args.budgets,
        "folds": args.folds,
        "algorithms": ["contextual", "fqi", "double_fqi", "neural_fqi"],
        "neural_hidden_layers": [16, 8],
        "neural_max_epochs_per_backup": 40,
        "bellman_backups": 6,
        "selection": "all algorithm variants fixed before test; no test selection",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
