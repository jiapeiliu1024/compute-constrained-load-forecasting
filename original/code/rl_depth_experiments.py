"""Controlled depth, state, seed, and endogenous-transition audits for budget RL.

This script complements the primary disjoint-gate experiment.  It keeps the
forecast models, information set, calendar-day budget, and chronological folds
fixed while varying only the policy learner.  Test outcomes are never used to
select a reported variant.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

import budget_rl_experiments as budget
import rl_router_experiments as common


EXPERTS = budget.EXPERTS
BASE_SEED = common.SEED


def policy_state(frame: pd.DataFrame, columns: list[str], variant: str) -> np.ndarray:
    if variant == "full":
        prefixes = ("lag_", "mean_", "std_", "hour_", "dow_", "x_")
    elif variant == "no_exogenous":
        prefixes = ("lag_", "mean_", "std_", "hour_", "dow_")
    elif variant == "no_history":
        prefixes = ("hour_", "dow_", "x_")
    elif variant == "no_calendar":
        prefixes = ("lag_", "mean_", "std_", "x_")
    elif variant == "calendar":
        prefixes = ("hour_", "dow_")
    else:
        raise ValueError(f"Unknown state variant: {variant}")
    selected = [column for column in columns if column.startswith(prefixes)]
    state = frame[selected].to_numpy(float)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite policy state")
    return state


def action_rows(
    state: np.ndarray,
    budget_fraction: float,
    include_budget: bool = True,
    previous_action: int | None = None,
) -> np.ndarray:
    shared = np.asarray(state, dtype=float)
    if include_budget:
        shared = np.r_[shared, budget_fraction]
    if previous_action is not None:
        shared = np.r_[shared, np.eye(len(EXPERTS))[previous_action]]
    return np.asarray([
        np.r_[shared, np.eye(len(EXPERTS))[action]]
        for action in range(len(EXPERTS))
    ])


def regressor(seed: int, n_estimators: int = 16) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=10,
        max_features=0.75,
        n_jobs=1,
        random_state=seed,
    )


def train_stationary_fqi(
    states: np.ndarray,
    losses: np.ndarray,
    index: pd.Index,
    costs: np.ndarray,
    daily_budget: int,
    gamma: float,
    backups: int,
    seed: int,
    include_budget: bool = True,
    switching_penalty: float = 0.0,
    n_estimators: int = 16,
) -> ExtraTreesRegressor:
    x_rows: list[np.ndarray] = []
    immediate: list[float] = []
    transitions: list[tuple[int, int, int, int | None]] = []
    remaining_after, next_is_new_day = budget.calendar_budget_geometry(index)
    budget_grid = (0.50, 1.0) if switching_penalty > 0 else (0.33, 0.66, 1.0)
    previous_actions: tuple[int | None, ...] = (
        tuple(range(len(EXPERTS))) if switching_penalty > 0 else (None,)
    )
    for t in range(len(states) - 1):
        reserve = remaining_after[t] * int(np.min(costs))
        for budget_fraction in budget_grid:
            remaining = int(round(budget_fraction * daily_budget))
            for previous in previous_actions:
                rows = action_rows(states[t], budget_fraction, include_budget, previous)
                for action in range(len(EXPERTS)):
                    if costs[action] + reserve > remaining:
                        continue
                    x_rows.append(rows[action])
                    switch = float(
                        switching_penalty > 0
                        and previous is not None
                        and action != previous
                    )
                    immediate.append(-losses[t, action] - switching_penalty * switch)
                    next_remaining = remaining - int(costs[action])
                    if next_is_new_day[t]:
                        next_remaining = daily_budget
                    next_previous = action if switching_penalty > 0 else None
                    transitions.append(
                        (t + 1, next_remaining, remaining_after[t + 1], next_previous)
                    )
    x = np.asarray(x_rows)
    reward = np.asarray(immediate)
    target = reward.copy()
    iterations = 1 if gamma == 0 else backups
    model: ExtraTreesRegressor | None = None
    for iteration in range(iterations):
        model = regressor(seed + iteration, n_estimators).fit(x, target)
        if gamma == 0:
            break
        next_t = np.fromiter((item[0] for item in transitions), dtype=int)
        remaining = np.fromiter((item[1] for item in transitions), dtype=int)
        later = np.fromiter((item[2] for item in transitions), dtype=int)
        previous = [item[3] for item in transitions]
        blocks = np.stack([
            action_rows(
                states[t], rem / daily_budget, include_budget, prev
            )
            for t, rem, prev in zip(next_t, remaining, previous)
        ])
        q = model.predict(blocks.reshape(-1, blocks.shape[-1])).reshape(
            len(transitions), len(EXPERTS)
        )
        reserve = later[:, None] * int(np.min(costs))
        feasible = costs[None, :] + reserve <= remaining[:, None]
        masked = np.where(feasible, q, -np.inf)
        future = np.where(feasible.any(axis=1), np.max(masked, axis=1), 0.0)
        target = reward + gamma * future
    if model is None:
        raise RuntimeError("FQI model was not fitted")
    return model


def train_stagewise_fqi(
    states: np.ndarray,
    losses: np.ndarray,
    index: pd.Index,
    costs: np.ndarray,
    daily_budget: int,
    seed: int,
) -> list[ExtraTreesRegressor]:
    """Backward finite-horizon fitted Q with one value model per daily stage."""
    dates = pd.DatetimeIndex(index).normalize()
    positions_by_date = [np.flatnonzero(dates == day) for day in pd.unique(dates)]
    positions_by_date = [positions for positions in positions_by_date if len(positions) == 24]
    if not positions_by_date:
        raise ValueError("Stagewise FQI needs complete days")
    models: list[ExtraTreesRegressor | None] = [None] * 24
    for stage in range(23, -1, -1):
        x_rows: list[np.ndarray] = []
        targets: list[float] = []
        reserve = (23 - stage) * int(np.min(costs))
        for positions in positions_by_date:
            t = int(positions[stage])
            for budget_fraction in (0.33, 0.66, 1.0):
                remaining = int(round(budget_fraction * daily_budget))
                rows = action_rows(states[t], budget_fraction)
                for action in range(len(EXPERTS)):
                    if costs[action] + reserve > remaining:
                        continue
                    value = -losses[t, action]
                    if stage < 23:
                        next_remaining = remaining - int(costs[action])
                        next_t = int(positions[stage + 1])
                        next_rows = action_rows(
                            states[next_t], next_remaining / daily_budget
                        )
                        next_q = models[stage + 1].predict(next_rows)  # type: ignore[union-attr]
                        next_reserve = (22 - stage) * int(np.min(costs))
                        feasible = costs + next_reserve <= next_remaining
                        if feasible.any():
                            value += float(np.max(next_q[feasible]))
                    x_rows.append(rows[action])
                    targets.append(value)
        models[stage] = regressor(seed + stage).fit(
            np.asarray(x_rows), np.asarray(targets)
        )
    return [model for model in models if model is not None]


def deploy(
    model,
    states: np.ndarray,
    predictions: dict[str, np.ndarray],
    costs: np.ndarray,
    daily_budget: int,
    index: pd.Index,
    include_budget: bool = True,
    switching_penalty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected: list[int] = []
    remaining_trace: list[int] = []
    stage_trace: list[int] = []
    remaining = daily_budget
    previous = int(np.argmin(costs))
    remaining_after, next_is_new_day = budget.calendar_budget_geometry(index)
    stage = 0
    for t, state in enumerate(states):
        if t > 0 and next_is_new_day[t - 1]:
            remaining = daily_budget
            previous = int(np.argmin(costs))
            stage = 0
        reserve = remaining_after[t] * int(np.min(costs))
        feasible = costs + reserve <= remaining
        rows = action_rows(
            state,
            remaining / daily_budget,
            include_budget,
            previous if switching_penalty > 0 else None,
        )
        fitted = model[stage] if isinstance(model, list) else model
        q = np.asarray(fitted.predict(rows), dtype=float)
        q[~feasible] = -np.inf
        action = int(np.argmax(q))
        selected.append(action)
        remaining -= int(costs[action])
        remaining_trace.append(remaining)
        stage_trace.append(stage)
        previous = action
        stage += 1
    matrix = np.column_stack([predictions[name] for name in EXPERTS])
    actions = np.asarray(selected, dtype=int)
    return (
        matrix[np.arange(len(actions)), actions],
        actions,
        np.asarray(remaining_trace, dtype=int),
        np.asarray(stage_trace, dtype=int),
    )


def objective(
    y: np.ndarray,
    pred: np.ndarray,
    actions: np.ndarray,
    scale: float,
    switching_penalty: float,
    index: pd.Index,
) -> float:
    dates = pd.DatetimeIndex(index).normalize().to_numpy()
    switches = np.zeros(len(actions), dtype=float)
    if len(actions) > 1:
        switches[1:] = (actions[1:] != actions[:-1]) & (dates[1:] == dates[:-1])
    return float(np.mean(np.abs(y - pred) / scale + switching_penalty * switches))


def evaluate_fold(
    spec, horizon, fold, train, calibration, test, daily_budget,
    include_stagewise=True, include_switching=True,
):
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
    states = {
        variant: policy_state(combined, columns, variant)
        for variant in (
            "full", "no_exogenous", "no_history", "no_calendar", "calendar"
        )
    }
    cal_states = {key: value[:len(calibration)] for key, value in states.items()}
    test_states = {key: value[len(calibration):] for key, value in states.items()}
    costs = np.asarray([relative[name] for name in EXPERTS], dtype=int)
    tune_start, gate_start = budget.calibration_day_boundaries(calibration.index)
    embargo_rows = int(np.ceil(horizon / 24)) * 24
    fit_end = tune_start - embargo_rows
    tune_end = gate_start - embargo_rows
    fit_index = calibration.index[:fit_end]
    static_pred, _ = budget.static_budget_baseline(
        cal_losses[tune_start:tune_end], test_pred, costs, daily_budget, test.index
    )
    static_mae = mean_absolute_error(y_test, static_pred)

    rows: list[dict] = []
    traces: list[pd.DataFrame] = []

    def record(
        family: str,
        variant: str,
        model,
        state_variant: str = "full",
        include_budget: bool = True,
        gamma: float = 0.0,
        backups: int = 1,
        seed: int = BASE_SEED,
        switching_penalty: float = 0.0,
        n_estimators: int = 16,
    ):
        pred, actions, remaining, stage = deploy(
            model,
            test_states[state_variant],
            test_pred,
            costs,
            daily_budget,
            test.index,
            include_budget,
            switching_penalty,
        )
        mae = mean_absolute_error(y_test, pred)
        rows.append({
            "dataset": spec.name,
            "horizon": horizon,
            "fold": fold,
            "family": family,
            "variant": variant,
            "state_variant": state_variant,
            "include_budget": include_budget,
            "gamma": gamma,
            "backups": backups,
            "seed": seed,
            "switching_penalty": switching_penalty,
            "n_estimators": n_estimators,
            "mae": mae,
            "mae_static": static_mae,
            "gain_vs_static_pct": 100 * (static_mae - mae) / static_mae,
            "normalized_objective": objective(
                y_test, pred, actions, scale, switching_penalty, test.index
            ),
            "switch_rate": float(np.mean(actions[1:] != actions[:-1])),
            "mean_cost": float(np.mean(costs[actions])),
            "max_daily_spend": int(
                pd.Series(costs[actions], index=test.index)
                .groupby(pd.DatetimeIndex(test.index).normalize()).sum().max()
            ),
            **{
                f"share_{name}": float(np.mean(actions == action))
                for action, name in enumerate(EXPERTS)
            },
        })
        if fold == 1 and variant in {
            "contextual_full", "stationary_fqi", "stagewise_fqi",
            "switch_contextual", "switch_fqi",
        }:
            traces.append(pd.DataFrame({
                "timestamp": test.index,
                "dataset": spec.name,
                "variant": variant,
                "stage": stage,
                "remaining_budget": remaining,
                "action": [EXPERTS[action] for action in actions],
                "absolute_error": np.abs(y_test - pred),
            }))

    full_fit = cal_states["full"][:fit_end]
    full_loss = cal_losses[:fit_end]

    contextual = train_stationary_fqi(
        full_fit, full_loss, fit_index, costs, daily_budget,
        gamma=0.0, backups=1, seed=BASE_SEED,
    )
    record("algorithm", "contextual_full", contextual)
    print(json.dumps({"fold": fold, "completed": "contextual"}), flush=True)

    stationary = train_stationary_fqi(
        full_fit, full_loss, fit_index, costs, daily_budget,
        gamma=1.0, backups=6, seed=BASE_SEED,
    )
    record(
        "algorithm", "stationary_fqi", stationary,
        gamma=1.0, backups=6,
    )
    print(json.dumps({"fold": fold, "completed": "stationary_fqi"}), flush=True)

    if include_stagewise:
        stagewise = train_stagewise_fqi(
            full_fit, full_loss, fit_index, costs, daily_budget, BASE_SEED
        )
        record(
            "algorithm", "stagewise_fqi", stagewise,
            gamma=1.0, backups=24,
        )
        print(json.dumps({"fold": fold, "completed": "stagewise_fqi"}), flush=True)

    for depth in (1, 3, 12):
        fitted = train_stationary_fqi(
            full_fit, full_loss, fit_index, costs, daily_budget,
            gamma=1.0, backups=depth, seed=BASE_SEED,
        )
        record(
            "backup_depth", f"depth_{depth}", fitted,
            gamma=1.0, backups=depth,
        )
    record(
        "backup_depth", "depth_6", stationary,
        gamma=1.0, backups=6,
    )
    print(json.dumps({"fold": fold, "completed": "depth_audit"}), flush=True)

    for state_variant in (
        "no_exogenous", "no_history", "no_calendar", "calendar"
    ):
        state_fit = cal_states[state_variant][:fit_end]
        for gamma, depth, label in (
            (0.0, 1, "contextual"), (1.0, 6, "fqi")
        ):
            fitted = train_stationary_fqi(
                state_fit, full_loss, fit_index, costs, daily_budget,
                gamma=gamma, backups=depth, seed=BASE_SEED,
            )
            record(
                "state_ablation", f"{label}_{state_variant}", fitted,
                state_variant=state_variant, gamma=gamma, backups=depth,
            )

    no_budget = train_stationary_fqi(
        full_fit, full_loss, fit_index, costs, daily_budget,
        gamma=0.0, backups=1, seed=BASE_SEED, include_budget=False,
    )
    record(
        "state_ablation", "contextual_no_budget", no_budget,
        include_budget=False,
    )
    no_budget_fqi = train_stationary_fqi(
        full_fit, full_loss, fit_index, costs, daily_budget,
        gamma=1.0, backups=6, seed=BASE_SEED, include_budget=False,
    )
    record(
        "state_ablation", "fqi_no_budget", no_budget_fqi,
        include_budget=False, gamma=1.0, backups=6,
    )
    print(json.dumps({"fold": fold, "completed": "state_ablation"}), flush=True)

    for gamma in (0.3, 0.6, 0.9):
        fitted = train_stationary_fqi(
            full_fit, full_loss, fit_index, costs, daily_budget,
            gamma=gamma, backups=6, seed=BASE_SEED,
        )
        record(
            "gamma", f"gamma_{gamma:g}", fitted,
            gamma=gamma, backups=6,
        )
    print(json.dumps({"fold": fold, "completed": "gamma_audit"}), flush=True)

    for trees in (8, 32, 64):
        fitted = train_stationary_fqi(
            full_fit, full_loss, fit_index, costs, daily_budget,
            gamma=1.0, backups=6, seed=BASE_SEED, n_estimators=trees,
        )
        record(
            "tree_count", f"trees_{trees}", fitted,
            gamma=1.0, backups=6, n_estimators=trees,
        )
    record(
        "tree_count", "trees_16", stationary,
        gamma=1.0, backups=6, n_estimators=16,
    )
    print(json.dumps({"fold": fold, "completed": "tree_audit"}), flush=True)

    record(
        "seed", f"seed_{BASE_SEED}", stationary,
        gamma=1.0, backups=6, seed=BASE_SEED,
    )
    for seed in (BASE_SEED + 1, BASE_SEED + 2, BASE_SEED + 3, BASE_SEED + 4):
        fitted = train_stationary_fqi(
            full_fit, full_loss, fit_index, costs, daily_budget,
            gamma=1.0, backups=6, seed=seed,
        )
        record(
            "seed", f"seed_{seed}", fitted,
            gamma=1.0, backups=6, seed=seed,
        )
    print(json.dumps({"fold": fold, "completed": "seed_audit"}), flush=True)

    if include_switching:
        for switch_penalty in (0.05,):
            contextual_switch = train_stationary_fqi(
                full_fit, full_loss, fit_index, costs, daily_budget,
                gamma=0.0, backups=1, seed=BASE_SEED,
                switching_penalty=switch_penalty,
            )
            record(
                "switching", "switch_contextual", contextual_switch,
                switching_penalty=switch_penalty,
            )
            fqi_switch = train_stationary_fqi(
                full_fit, full_loss, fit_index, costs, daily_budget,
                gamma=1.0, backups=6, seed=BASE_SEED,
                switching_penalty=switch_penalty,
            )
            record(
                "switching", "switch_fqi", fqi_switch,
                gamma=1.0, backups=6, switching_penalty=switch_penalty,
            )
        print(json.dumps({"fold": fold, "completed": "switching"}), flush=True)

    return rows, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["steel", "tetouan"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[12])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--daily-budget", type=int, default=108)
    parser.add_argument("--skip-stagewise", action="store_true")
    parser.add_argument("--skip-switching", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/rl_depth_audit_v1")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_traces: list[pd.DataFrame] = []
    for dataset in args.datasets:
        base_spec = common.SPECS[dataset]
        spec = replace(base_spec, calibration_days=84, folds=args.folds)
        hourly = common.load_hourly(spec)
        for horizon in args.horizons:
            frame = common.feature_frame(hourly, spec.target, horizon)
            for fold, train, calibration, test in common.chronological_folds(
                frame, spec, horizon
            ):
                rows, traces = evaluate_fold(
                    spec, horizon, fold, train, calibration, test,
                    args.daily_budget,
                    include_stagewise=not args.skip_stagewise,
                    include_switching=not args.skip_switching,
                )
                all_rows.extend(rows)
                all_traces.extend(traces)
                pd.DataFrame(all_rows).to_csv(
                    args.output / "policy_metrics.csv", index=False
                )
                if all_traces:
                    pd.concat(all_traces, ignore_index=True).to_csv(
                        args.output / "policy_traces.csv", index=False
                    )
                print(json.dumps({
                    "dataset": dataset,
                    "horizon": horizon,
                    "fold": fold,
                    "variants": len(rows),
                }), flush=True)
    metrics = pd.DataFrame(all_rows)
    summary = (
        metrics.groupby(["family", "variant", "switching_penalty"], dropna=False)
        .agg(
            units=("mae", "size"),
            mean_gain_pct=("gain_vs_static_pct", "mean"),
            mean_objective=("normalized_objective", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
            mean_cost=("mean_cost", "mean"),
            max_daily_spend=("max_daily_spend", "max"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "metadata.json").write_text(json.dumps({
        "datasets": args.datasets,
        "horizons": args.horizons,
        "folds": args.folds,
        "daily_budget": args.daily_budget,
        "test_selection": "none; all variants were fixed before test evaluation",
        "seed_values": [BASE_SEED + offset for offset in range(5)],
        "gamma_values": [0.0, 0.3, 0.6, 0.9, 1.0],
        "backup_depths": [1, 3, 6, 12],
        "tree_counts": [8, 16, 32, 64],
        "switching_penalties": [0.05],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
