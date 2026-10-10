"""Budget-constrained offline RL for load-forecast expert allocation.

Unlike a contextual router, the action here changes the next state by consuming
a finite daily inference budget.  This makes delayed return and gamma testable.
All candidate forecasts are computed offline for evaluation, but the policy is
charged the median measured inference cost of the action it selects.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rl_router_experiments as common


SEED = common.SEED
EXPERTS = ("daily", "weekly", "ridge", "hgb")
# Frozen relative inference-cost profile.  It is based on repeated warm-run
# measurements from the prototype machine and deliberately rounded to avoid
# fold-to-fold timing noise changing the definition of the budget treatment.
FROZEN_RELATIVE_COST = {
    "daily": 1, "weekly": 1, "ridge": 3, "hgb": 18,
    "random_forest": 120, "extra_trees": 120,
}


def model_pool():
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        ),
        "hgb": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=220, max_leaf_nodes=31,
                l2_regularization=0.2, random_state=SEED,
            ),
        ),
    }


def latency_profile(operation, sample_size: int, repeats: int = 21) -> dict[str, float | int]:
    """Measure a warm batch operation and retain its dispersion, not only its median."""
    if sample_size <= 0:
        raise ValueError("Latency measurement needs at least one origin")
    operation()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - start) / 1e6 / sample_size)
    values = np.asarray(values, dtype=float)
    return {
        "sample_origins": int(sample_size),
        "repetitions": int(repeats),
        "median_ms_per_origin": float(np.median(values)),
        "q25_ms_per_origin": float(np.quantile(values, 0.25)),
        "q75_ms_per_origin": float(np.quantile(values, 0.75)),
        "min_ms_per_origin": float(np.min(values)),
        "max_ms_per_origin": float(np.max(values)),
    }


def model_latency_profile(model, x: pd.DataFrame) -> dict[str, float | int]:
    sample = x.iloc[: min(336, len(x))]
    return latency_profile(lambda: model.predict(sample), len(sample))


def persistence_latency_profile(values: np.ndarray) -> dict[str, float | int]:
    sample_size = min(336, len(values))
    positions = np.arange(sample_size, dtype=int)
    return latency_profile(lambda: np.take(values, positions), sample_size)


def predictions_and_costs(train, calibration, test):
    excluded = {"target", "daily", "weekly"}
    columns = [column for column in train.columns if column not in excluded]
    calibration_predictions = {
        "daily": calibration.daily.to_numpy(float),
        "weekly": calibration.weekly.to_numpy(float),
    }
    test_predictions = {
        "daily": test.daily.to_numpy(float),
        "weekly": test.weekly.to_numpy(float),
    }
    latency = {
        "daily": persistence_latency_profile(calibration_predictions["daily"]),
        "weekly": persistence_latency_profile(calibration_predictions["weekly"]),
    }
    for name, model in model_pool().items():
        model.fit(train[columns], train.target)
        calibration_predictions[name] = model.predict(calibration[columns])
        test_predictions[name] = model.predict(test[columns])
        latency[name] = model_latency_profile(model, calibration[columns])
    measured_ms = {
        name: float(profile["median_ms_per_origin"])
        for name, profile in latency.items()
    }
    relative_cost = {name: FROZEN_RELATIVE_COST[name] for name in EXPERTS}
    return (
        columns, calibration_predictions, test_predictions,
        measured_ms, latency, relative_cost,
    )


def observable_state(frame, columns):
    """Deployment state available before any candidate forecast is computed."""
    state_columns = [
        column for column in columns
        if column.startswith(("lag_", "mean_", "std_", "hour_", "dow_", "x_"))
    ]
    state = frame[state_columns].to_numpy(float)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite observable policy state")
    return state


def action_rows(state, budget_fraction: float) -> np.ndarray:
    rows = []
    for action in range(len(EXPERTS)):
        action_hot = np.eye(len(EXPERTS))[action]
        rows.append(np.r_[state, budget_fraction, action_hot])
    return np.asarray(rows)


def calendar_budget_geometry(index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Return origins remaining after each row and whether the next row is a new day."""
    dates = pd.DatetimeIndex(index).normalize().to_numpy()
    remaining_after = np.zeros(len(dates), dtype=int)
    next_is_new_day = np.ones(len(dates), dtype=bool)
    for position in range(len(dates) - 2, -1, -1):
        if dates[position + 1] == dates[position]:
            remaining_after[position] = remaining_after[position + 1] + 1
            next_is_new_day[position] = False
    return remaining_after, next_is_new_day


def complete_calendar_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates containing all 24 hourly forecast origins."""
    dates = pd.DatetimeIndex(frame.index).normalize()
    counts = pd.Series(1, index=dates).groupby(level=0).sum()
    complete = counts.index[counts == 24]
    result = frame.loc[dates.isin(complete)]
    if result.empty:
        raise ValueError("No complete calendar day remains in the evaluation window")
    verification = pd.Series(1, index=pd.DatetimeIndex(result.index).normalize()).groupby(level=0).sum()
    if not (verification == 24).all():
        raise AssertionError("Incomplete day survived calendar-day filtering")
    return result


def calibration_day_boundaries(index: pd.Index) -> tuple[int, int]:
    """Return midnight-aligned 50% and 75% boundaries by complete days."""
    dates = pd.DatetimeIndex(index).normalize()
    unique_dates = pd.unique(dates)
    if len(unique_dates) < 8:
        raise ValueError("Calibration needs at least eight complete days")
    tune_day = max(1, int(np.floor(0.50 * len(unique_dates))))
    gate_day = max(tune_day + 1, int(np.floor(0.75 * len(unique_dates))))
    tune_start = int(np.flatnonzero(dates == unique_dates[tune_day])[0])
    gate_start = int(np.flatnonzero(dates == unique_dates[gate_day])[0])
    return tune_start, gate_start


def split_gate_boundaries(
    index: pd.Index, gate_start: int, embargo_days: int,
) -> tuple[int, int, int, int]:
    """Split the final gate period into disjoint chronological evidence blocks.

    The first block selects sequential versus contextual allocation.  After a
    whole-day outcome embargo, the second block evaluates the frozen selected
    dynamic policy against the static action.  No outcome enters both gates.
    """
    dates = pd.DatetimeIndex(index).normalize()
    gate_dates = pd.unique(dates[gate_start:])
    usable_days = len(gate_dates) - embargo_days
    if usable_days < 4:
        raise ValueError("Gate period is too short for two disjoint evidence blocks")
    sequential_days = usable_days // 2
    deployment_days = usable_days - sequential_days
    if sequential_days < 2 or deployment_days < 2:
        raise ValueError("Each gate needs at least two complete days")
    sequential_end_date = gate_dates[sequential_days]
    deployment_start_date = gate_dates[sequential_days + embargo_days]
    sequential_end = int(np.flatnonzero(dates == sequential_end_date)[0])
    deployment_start = int(np.flatnonzero(dates == deployment_start_date)[0])
    return gate_start, sequential_end, deployment_start, len(index)


def train_budget_fqi(
    states: np.ndarray,
    losses: np.ndarray,
    index: pd.Index,
    costs: np.ndarray,
    daily_budget: int,
    gamma: float,
    cost_penalty: float,
):
    x_rows, immediate, transitions = [], [], []
    remaining_after, next_is_new_day = calendar_budget_geometry(index)
    # Counterfactual coverage over multiple feasible starting budgets and prior actions.
    budget_grid = (0.33, 0.66, 1.0)
    for t in range(len(states) - 1):
        reserve = remaining_after[t] * int(np.min(costs))
        for budget_fraction in budget_grid:
            remaining = int(round(budget_fraction * daily_budget))
            for action in range(len(EXPERTS)):
                if costs[action] + reserve > remaining:
                    continue
                x_rows.append(action_rows(states[t], budget_fraction)[action])
                reward = -losses[t, action] - cost_penalty * costs[action] / daily_budget
                immediate.append(reward)
                next_remaining = remaining - costs[action]
                if next_is_new_day[t]:
                    next_remaining = daily_budget
                transitions.append((t + 1, next_remaining, remaining_after[t + 1]))
    x = np.asarray(x_rows)
    target = np.asarray(immediate)
    model = None
    # The episode ends at the calendar-day boundary, so the undiscounted
    # finite-horizon case (gamma=1) is well defined.  Six fitted backups
    # propagate value beyond the four-backup prototype without making the
    # complete rolling-origin audit prohibitively expensive.
    iterations = 1 if gamma == 0 else 6
    for iteration in range(iterations):
        model = ExtraTreesRegressor(
            n_estimators=60, min_samples_leaf=10, max_features=0.75,
            n_jobs=1, random_state=SEED + iteration,
        ).fit(x, target)
        if gamma == 0:
            break
        next_rows = np.vstack([
            action_rows(states[next_t], remaining / daily_budget)
            for next_t, remaining, _ in transitions
        ])
        q_all = model.predict(next_rows).reshape(len(transitions), len(EXPERTS))
        future = np.zeros(len(transitions))
        for i, (_, remaining, next_remaining_after) in enumerate(transitions):
            reserve = next_remaining_after * int(np.min(costs))
            feasible = costs + reserve <= remaining
            future[i] = np.max(q_all[i, feasible]) if feasible.any() else 0.0
        target = np.asarray(immediate) + gamma * future
    return model


def deploy(model, states, predictions, costs, daily_budget, index: pd.Index):
    selected, remaining_trace = [], []
    remaining = daily_budget
    remaining_after, next_is_new_day = calendar_budget_geometry(index)
    for t, state in enumerate(states):
        if t > 0 and next_is_new_day[t - 1]:
            remaining = daily_budget
        reserve = remaining_after[t] * int(np.min(costs))
        feasible = costs + reserve <= remaining
        rows = action_rows(state, remaining / daily_budget)
        q = np.asarray(model.predict(rows), dtype=float)
        q[~feasible] = -np.inf
        if not feasible.any():
            raise RuntimeError("Budget feasibility invariant violated")
        action = int(np.argmax(q))
        selected.append(action)
        remaining -= int(costs[action])
        remaining_trace.append(remaining)
    matrix = np.column_stack([predictions[name] for name in EXPERTS])
    selected = np.asarray(selected)
    selected_costs = costs[selected]
    daily_spend = pd.Series(selected_costs, index=pd.DatetimeIndex(index)).groupby(
        pd.DatetimeIndex(index).normalize()
    ).sum()
    if (daily_spend > daily_budget).any():
        raise RuntimeError("Daily budget exceeded after deployment")
    diagnostics = {
        "remaining_trace": np.asarray(remaining_trace),
        "max_daily_spend": int(daily_spend.max()),
        "mean_daily_spend": float(daily_spend.mean()),
        "budget_violation_count": int((daily_spend > daily_budget).sum()),
    }
    return matrix[np.arange(len(selected)), selected], selected, diagnostics


def static_budget_baseline(cal_losses, test_predictions, costs, daily_budget, test_index):
    max_origins_per_day = int(
        pd.Series(1, index=pd.DatetimeIndex(test_index)).groupby(
            pd.DatetimeIndex(test_index).normalize()
        ).sum().max()
    )
    feasible = costs * max_origins_per_day <= daily_budget
    if not feasible.any():
        action = int(np.argmin(costs))
    else:
        means = cal_losses.mean(axis=0)
        means[~feasible] = np.inf
        action = int(np.argmin(means))
    return test_predictions[EXPERTS[action]], action


def oracle_budget_mae(y: np.ndarray, predictions: dict[str, np.ndarray], costs, daily_budget, index):
    """Hindsight dynamic-programming lower bound under the identical budget."""
    loss = np.column_stack([np.abs(y - predictions[name]) for name in EXPERTS])
    selected_loss = []
    selected_cost = []
    dates = pd.DatetimeIndex(index).normalize()
    for day in pd.unique(dates):
        positions = np.flatnonzero(dates == day)
        block = loss[positions]
        steps = len(block)
        dp = np.full((steps + 1, daily_budget + 1), np.inf)
        parent = np.full((steps + 1, daily_budget + 1, 2), -1, dtype=int)
        dp[0, 0] = 0.0
        for t in range(steps):
            for used in range(daily_budget + 1):
                if not np.isfinite(dp[t, used]):
                    continue
                for action, cost in enumerate(costs):
                    next_used = used + int(cost)
                    if next_used > daily_budget:
                        continue
                    value = dp[t, used] + block[t, action]
                    if value < dp[t + 1, next_used]:
                        dp[t + 1, next_used] = value
                        parent[t + 1, next_used] = (used, action)
        used = int(np.argmin(dp[steps]))
        for t in range(steps, 0, -1):
            previous_used, action = parent[t, used]
            selected_loss.append(block[t - 1, action])
            selected_cost.append(costs[action])
            used = previous_used
    return float(np.mean(selected_loss)), float(np.mean(selected_cost))


def budgeted_upgrade_baselines(
    y_tune: np.ndarray,
    tune_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    costs: np.ndarray,
    daily_budget: int,
    tune_index: pd.Index,
    test_index: pd.Index,
):
    """Non-learning budget baselines with the identical feasible action set."""
    tune_loss = np.column_stack(
        [np.abs(y_tune - tune_predictions[name]) for name in EXPERTS]
    )
    cheap_feasible = costs * 24 <= daily_budget
    cheap_score = tune_loss.mean(axis=0).copy()
    cheap_score[~cheap_feasible] = np.inf
    cheap = int(np.argmin(cheap_score))

    affordable_once = costs <= daily_budget - 23 * costs[cheap]
    mean_gain = (tune_loss[:, cheap, None] - tune_loss).mean(axis=0)
    mean_gain[~affordable_once] = -np.inf
    mean_gain[cheap] = -np.inf
    upgrade = int(np.argmax(mean_gain))
    if not np.isfinite(mean_gain[upgrade]) or mean_gain[upgrade] <= 0:
        upgrade = cheap
    incremental_cost = int(costs[upgrade] - costs[cheap])
    upgrades_per_day = (
        0 if incremental_cost <= 0
        else max(0, int((daily_budget - 24 * costs[cheap]) // incremental_cost))
    )

    matrix = np.column_stack([test_predictions[name] for name in EXPERTS])
    disagreement = np.std(matrix / np.maximum(np.mean(np.abs(matrix), axis=1, keepdims=True), 1e-6), axis=1)
    disagreement_actions = np.full(len(matrix), cheap, dtype=int)
    if upgrades_per_day > 0 and upgrade != cheap:
        dates = pd.DatetimeIndex(test_index).normalize()
        for day in pd.unique(dates):
            positions = np.flatnonzero(dates == day)
            k = min(upgrades_per_day, len(positions))
            chosen = positions[np.argsort(disagreement[positions])[-k:]]
            disagreement_actions[chosen] = upgrade

    tune_hours = tune_index.hour.to_numpy()
    gain = tune_loss[:, cheap] - tune_loss[:, upgrade]
    hour_gain = np.asarray([
        gain[tune_hours == hour].mean() if np.any(tune_hours == hour) else -np.inf
        for hour in range(24)
    ])
    preferred_hours = set(np.argsort(hour_gain)[-upgrades_per_day:]) if upgrades_per_day else set()
    schedule_actions = np.asarray([
        upgrade if timestamp.hour in preferred_hours else cheap for timestamp in test_index
    ])

    return {
        "cheap_action": EXPERTS[cheap],
        "upgrade_action": EXPERTS[upgrade],
        "upgrades_per_day": upgrades_per_day,
        "disagreement_prediction": matrix[np.arange(len(matrix)), disagreement_actions],
        "schedule_prediction": matrix[np.arange(len(matrix)), schedule_actions],
        "disagreement_mean_cost": float(np.mean(costs[disagreement_actions])),
        "schedule_mean_cost": float(np.mean(costs[schedule_actions])),
    }


def run_fold(spec, horizon, fold, train, calibration, test, budget_multiplier, fixed_daily_budget=None):
    calibration = complete_calendar_days(calibration)
    test = complete_calendar_days(test)
    columns, cal_pred, test_pred, measured_ms, latency, relative = predictions_and_costs(
        train, calibration, test
    )
    y_cal = calibration.target.to_numpy(float)
    y_test = test.target.to_numpy(float)
    scale = max(np.mean(np.abs(y_cal)), 1e-6)
    cal_losses = np.column_stack(
        [np.abs(y_cal - cal_pred[name]) / scale for name in EXPERTS]
    )
    test_losses = np.column_stack(
        [np.abs(y_test - test_pred[name]) / scale for name in EXPERTS]
    )
    combined_frame = pd.concat((calibration, test))
    states = observable_state(combined_frame, columns)
    cal_states, test_states = states[:len(calibration)], states[len(calibration):]
    costs = np.asarray([relative[name] for name in EXPERTS], dtype=int)
    daily_budget = (
        int(fixed_daily_budget) if fixed_daily_budget is not None
        else max(int(24 * np.min(costs)), int(round(budget_multiplier * 24 * np.median(costs))))
    )
    tune_start, gate_start = calibration_day_boundaries(calibration.index)
    embargo_days = int(np.ceil(horizon / 24))
    embargo_rows = embargo_days * 24
    fit_end = tune_start - embargo_rows
    tune_end = gate_start - embargo_rows
    if fit_end <= 0 or tune_end <= tune_start:
        raise ValueError("Complete-day calibration blocks are too short after embargo")
    candidates = []
    # Include gamma=1: daily reset makes this a finite-horizon CMDP, for which
    # discounting is not required for return convergence.  Gamma=0 remains the
    # contextual counterfactual; positive values test sequential credit.
    for gamma in (0.0, 0.6, 0.9, 1.0):
        for penalty in (0.0, 0.05, 0.2):
            model = train_budget_fqi(
                cal_states[:fit_end], cal_losses[:fit_end], calibration.index[:fit_end],
                costs, daily_budget, gamma, penalty
            )
            pred, actions, _ = deploy(
                model, cal_states[tune_start:tune_end],
                {name: cal_pred[name][tune_start:tune_end] for name in EXPERTS},
                costs, daily_budget, calibration.index[tune_start:tune_end],
            )
            score = mean_absolute_error(y_cal[tune_start:tune_end], pred)
            candidates.append((score, gamma, penalty, model))
    _, seq_gamma, seq_penalty, seq_model = min(
        [row for row in candidates if row[1] > 0.0], key=lambda row: row[0]
    )
    _, myopic_penalty, myopic_model = min(
        [(score, penalty_value, fitted) for score, gamma_value, penalty_value, fitted in candidates if gamma_value == 0.0],
        key=lambda row: row[0],
    )
    seq_gate_start, seq_gate_end, deploy_gate_start, deploy_gate_end = split_gate_boundaries(
        calibration.index, gate_start, embargo_days
    )
    seq_gate_predictions = {
        name: cal_pred[name][seq_gate_start:seq_gate_end] for name in EXPERTS
    }
    seq_gate, _, _ = deploy(
        seq_model, cal_states[seq_gate_start:seq_gate_end], seq_gate_predictions,
        costs, daily_budget, calibration.index[seq_gate_start:seq_gate_end]
    )
    myopic_gate, _, _ = deploy(
        myopic_model, cal_states[seq_gate_start:seq_gate_end], seq_gate_predictions,
        costs, daily_budget, calibration.index[seq_gate_start:seq_gate_end]
    )
    y_seq_gate = y_cal[seq_gate_start:seq_gate_end]
    seq_minus_myopic = np.abs(y_seq_gate - seq_gate) - np.abs(y_seq_gate - myopic_gate)
    seq_gate_upper = common.moving_block_bootstrap_upper(seq_minus_myopic)
    use_sequential = bool(seq_gate_upper < 0.0)
    chosen_model = seq_model if use_sequential else myopic_model
    gamma = seq_gamma if use_sequential else 0.0
    penalty = seq_penalty if use_sequential else myopic_penalty

    rl_pred, actions, budget_diagnostics = deploy(
        chosen_model, test_states, test_pred, costs, daily_budget, test.index
    )
    myopic_pred, myopic_actions, myopic_budget_diagnostics = deploy(
        myopic_model, test_states, test_pred, costs, daily_budget, test.index
    )
    static_pred, static_action = static_budget_baseline(
        cal_losses[tune_start:tune_end], test_pred, costs, daily_budget, test.index
    )
    deploy_gate_predictions = {
        name: cal_pred[name][deploy_gate_start:deploy_gate_end] for name in EXPERTS
    }
    chosen_deploy_gate, _, _ = deploy(
        chosen_model, cal_states[deploy_gate_start:deploy_gate_end],
        deploy_gate_predictions, costs, daily_budget,
        calibration.index[deploy_gate_start:deploy_gate_end],
    )
    static_gate_pred = deploy_gate_predictions[EXPERTS[static_action]]
    y_deploy_gate = y_cal[deploy_gate_start:deploy_gate_end]
    chosen_minus_static = (
        np.abs(y_deploy_gate - chosen_deploy_gate)
        - np.abs(y_deploy_gate - static_gate_pred)
    )
    policy_gate_upper = common.moving_block_bootstrap_upper(chosen_minus_static)
    use_dynamic = bool(policy_gate_upper < 0.0)
    conservative_pred = rl_pred if use_dynamic else static_pred
    oracle_mae, oracle_cost = oracle_budget_mae(
        y_test, test_pred, costs, daily_budget, test.index
    )
    heuristic = budgeted_upgrade_baselines(
        y_cal[tune_start:tune_end],
        {name: cal_pred[name][tune_start:tune_end] for name in EXPERTS},
        test_pred, costs, daily_budget,
        calibration.index[tune_start:tune_end], test.index,
    )
    return {
        "dataset": spec.name, "horizon": horizon, "fold": fold,
        "fit_days": int(fit_end // 24),
        "tune_days": int((tune_end - tune_start) // 24),
        "gate_days": int(((seq_gate_end - seq_gate_start) + (deploy_gate_end - deploy_gate_start)) // 24),
        "sequential_gate_days": int((seq_gate_end - seq_gate_start) // 24),
        "deployment_gate_days": int((deploy_gate_end - deploy_gate_start) // 24),
        "gate_embargo_days": embargo_days,
        "test_days": int(len(test) // 24),
        "n_test": int(len(test)),
        "daily_budget": daily_budget, "gamma": gamma, "cost_penalty": penalty,
        "sequential_gamma_candidate": seq_gamma,
        "sequential_gate_upper": seq_gate_upper,
        "use_sequential": use_sequential,
        "policy_gate_upper": policy_gate_upper,
        "use_dynamic": use_dynamic,
        "myopic_cost_penalty": myopic_penalty,
        "static_action": EXPERTS[static_action],
        "mae_static_budget": mean_absolute_error(y_test, static_pred),
        "mae_myopic_budget": mean_absolute_error(y_test, myopic_pred),
        "mae_budget_rl": mean_absolute_error(y_test, rl_pred),
        "mae_conservative_budget": mean_absolute_error(y_test, conservative_pred),
        "mae_oracle_budget": oracle_mae,
        "mae_disagreement_trigger": mean_absolute_error(
            y_test, heuristic["disagreement_prediction"]
        ),
        "mae_hour_schedule": mean_absolute_error(y_test, heuristic["schedule_prediction"]),
        "heuristic_cheap_action": heuristic["cheap_action"],
        "heuristic_upgrade_action": heuristic["upgrade_action"],
        "heuristic_upgrades_per_day": heuristic["upgrades_per_day"],
        "disagreement_mean_relative_cost": heuristic["disagreement_mean_cost"],
        "schedule_mean_relative_cost": heuristic["schedule_mean_cost"],
        "rl_vs_static_pct": 100 * (
            mean_absolute_error(y_test, static_pred) - mean_absolute_error(y_test, rl_pred)
        ) / mean_absolute_error(y_test, static_pred),
        "mean_relative_cost": float(np.mean(costs[actions])),
        "myopic_mean_relative_cost": float(np.mean(costs[myopic_actions])),
        "oracle_mean_relative_cost": oracle_cost,
        "max_daily_spend": budget_diagnostics["max_daily_spend"],
        "mean_daily_spend": budget_diagnostics["mean_daily_spend"],
        "budget_violation_count": budget_diagnostics["budget_violation_count"],
        "myopic_max_daily_spend": myopic_budget_diagnostics["max_daily_spend"],
        "switch_rate": float(np.mean(actions[1:] != actions[:-1])),
        "measured_ms": json.dumps(measured_ms, sort_keys=True),
        "latency_profile": json.dumps(latency, sort_keys=True),
        "relative_cost": json.dumps(relative, sort_keys=True),
        **{f"share_{name}": float(np.mean(actions == i)) for i, name in enumerate(EXPERTS)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(common.SPECS), default="steel")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--fold-limit", type=int, default=1)
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    parser.add_argument("--daily-budget", type=int)
    parser.add_argument("--calibration-days", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/budget_rl_prototype"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = common.SPECS[args.dataset]
    if args.calibration_days is not None:
        spec = replace(spec, calibration_days=args.calibration_days)
    if args.folds is not None:
        spec = replace(spec, folds=args.folds)
    hourly = common.load_hourly(spec)
    frame = common.feature_frame(hourly, spec.target, args.horizon)
    rows = []
    for fold, train, calibration, test in common.chronological_folds(frame, spec, args.horizon):
        result = run_fold(
            spec, args.horizon, fold, train, calibration, test,
            args.budget_multiplier, args.daily_budget,
        )
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if fold >= args.fold_limit:
            break
    details = pd.DataFrame(rows)
    details.to_csv(args.output / "fold_metrics.csv", index=False)
    if not details.empty:
        summary = {
            "dataset": args.dataset,
            "horizon": args.horizon,
            "folds": len(details),
            "daily_budget": int(details.daily_budget.iloc[0]),
            "mae_static_budget": float(details.mae_static_budget.mean()),
            "mae_myopic_budget": float(details.mae_myopic_budget.mean()),
            "mae_selected_dynamic": float(details.mae_budget_rl.mean()),
            "mae_conservative_budget": float(details.mae_conservative_budget.mean()),
            "mae_oracle_budget": float(details.mae_oracle_budget.mean()),
            "dynamic_vs_static_pct": float(
                100 * (details.mae_static_budget.mean() - details.mae_budget_rl.mean())
                / details.mae_static_budget.mean()
            ),
            "conservative_vs_static_pct": float(
                100 * (details.mae_static_budget.mean() - details.mae_conservative_budget.mean())
                / details.mae_static_budget.mean()
            ),
            "sequential_activation_rate": float(details.use_sequential.mean()),
            "dynamic_activation_rate": float(details.use_dynamic.mean()),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
