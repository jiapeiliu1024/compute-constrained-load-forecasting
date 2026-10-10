"""Leakage-controlled offline-RL expert routing for public energy datasets.

The numerical forecasters are fitted on the training block.  A fitted-Q router
is then learned only from the calibration block, using the counterfactual loss
of every already-frozen expert.  Hyperparameters are selected on the final
part of calibration; the test block is touched exactly once.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260821
HORIZONS = (1, 6, 12, 24)
LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
BASE_EXPERTS = ("daily", "weekly", "ridge", "hgb")
ACTIONS = (*BASE_EXPERTS, "equal", "adaptive")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    target: str
    date_format: str | None
    min_train_days: int
    calibration_days: int
    test_days: int
    folds: int


SPECS = {
    "appliances": DatasetSpec(
        "appliances", Path("data/raw/uci_appliances/data.csv"), "Appliances",
        None, 60, 14, 14, 3,
    ),
    "steel": DatasetSpec(
        "steel", Path("data/raw/uci_steel/data.csv"), "Usage_kWh",
        "%d/%m/%Y %H:%M", 120, 28, 14, 5,
    ),
    "tetouan": DatasetSpec(
        "tetouan", Path("data/processed/uci_tetouan_hourly.csv"), "aggregate_load",
        None, 120, 84, 14, 5,
    ),
    "uci_ecl": DatasetSpec(
        "uci_ecl", Path("data/processed/uci_ecl_hourly.csv"), "aggregate_load_kw",
        None, 365, 84, 28, 5,
    ),
    "opsd": DatasetSpec(
        "opsd", Path("data/processed/opsd_aggregate_hourly.csv"), "aggregate_load",
        None, 365, 84, 28, 5,
    ),
}


def load_hourly(spec: DatasetSpec) -> pd.DataFrame:
    raw = pd.read_csv(spec.path)
    if spec.name == "appliances":
        # The UCI CSV omits the separator between the day and hour.
        raw["date"] = raw["date"].str.replace(
            r"^(\d{4}-\d{2}-\d{2})(\d{2}:)", r"\1 \2", regex=True
        )
    time_column = "date" if "date" in raw.columns else "timestamp"
    raw[time_column] = pd.to_datetime(raw[time_column], format=spec.date_format)
    raw = raw.set_index(time_column).sort_index()
    numeric = raw.select_dtypes(include=[np.number])
    hourly = numeric.resample("1h").mean()
    if spec.target not in hourly:
        raise ValueError(f"Missing target {spec.target}")
    if hourly.index.duplicated().any() or hourly[spec.target].isna().any():
        raise ValueError("Hourly time axis is not complete")
    # Random variables in the appliance dataset are documented negative controls.
    return hourly.drop(columns=["rv1", "rv2"], errors="ignore")


def feature_frame(hourly: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    y = hourly[target].astype(float)
    out = pd.DataFrame(index=hourly.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag)
    for window in (6, 24, 168):
        history = y.shift(1).rolling(window)
        out[f"mean_{window}"] = history.mean()
        out[f"std_{window}"] = history.std()
    hour = out.index.hour.to_numpy()
    dow = out.index.dayofweek.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    # Only contemporaneously observed covariates are used.  No future weather is assumed.
    for column in hourly.columns:
        if column != target:
            out[f"x_{column}"] = hourly[column]
    out["target"] = y.shift(-horizon)
    out["daily"] = y.shift(24 - horizon)
    out["weekly"] = y.shift(168 - horizon)
    return out.dropna()


def chronological_folds(frame: pd.DataFrame, spec: DatasetSpec, horizon: int):
    steps_train = spec.min_train_days * 24
    steps_cal = spec.calibration_days * 24
    steps_test = spec.test_days * 24
    required = steps_train + steps_cal + spec.folds * steps_test
    if len(frame) < required:
        raise ValueError(f"{spec.name}: {len(frame)} usable rows, need {required}")
    first_test_end = len(frame) - (spec.folds - 1) * steps_test
    for fold in range(spec.folds):
        test_end = first_test_end + fold * steps_test
        test_start = test_end - steps_test
        cal_start = test_start - steps_cal
        # Target-time embargo: an origin at position t predicts t+h.  Therefore
        # the final h origins of each preceding stage are not eligible for
        # fitting or policy learning at the next stage boundary.
        train = frame.iloc[:cal_start - horizon]
        calibration = frame.iloc[cal_start:test_start - horizon]
        test = frame.iloc[test_start:test_end]
        yield fold + 1, train, calibration, test


def frozen_predictions(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame):
    excluded = {"target", "daily", "weekly"}
    columns = [c for c in train.columns if c not in excluded]
    models = {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "hgb": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=220, max_leaf_nodes=31,
                l2_regularization=0.2, random_state=SEED,
            ),
        ),
    }
    cal_pred = {name: cal[name].to_numpy(float) for name in ("daily", "weekly")}
    test_pred = {name: test[name].to_numpy(float) for name in ("daily", "weekly")}
    for name, model in models.items():
        model.fit(train[columns], train["target"])
        cal_pred[name] = model.predict(cal[columns])
        test_pred[name] = model.predict(test[columns])
    return columns, cal_pred, test_pred


def base_router_state(frame: pd.DataFrame, columns: list[str], predictions: dict[str, np.ndarray]) -> np.ndarray:
    base_cols = [c for c in columns if c.startswith(("lag_", "mean_", "std_", "hour_", "dow_"))]
    base = frame[base_cols].to_numpy(float)
    if not np.isfinite(base).all():
        raise ValueError("Non-finite router state after chronological feature construction")
    pred = np.column_stack([predictions[name] for name in BASE_EXPERTS])
    scale = np.maximum(np.mean(np.abs(pred), axis=1, keepdims=True), 1e-6)
    normalized_pred = pred / scale
    disagreement = np.column_stack((np.std(normalized_pred, axis=1), np.ptp(normalized_pred, axis=1)))
    return np.column_stack((base, normalized_pred, disagreement))


def matured_error_memory(losses: np.ndarray, horizon: int) -> np.ndarray:
    """Causal expert-error memory available at each forecast origin.

    Loss at origin j becomes observable only when its target at j+h is
    realized.  Shifting by h therefore excludes every unresolved outcome.
    """
    loss_frame = pd.DataFrame(losses)
    available = loss_frame.shift(horizon)
    features = []
    for window in (24, 168):
        rolling = available.rolling(window, min_periods=max(4, min(window, horizon))).mean()
        features.append(rolling.to_numpy())
    recent = features[0]
    rank = pd.DataFrame(recent).rank(axis=1, method="average", pct=True).to_numpy()
    drift = features[0] - features[1]
    memory = np.column_stack((*features, rank, drift))
    # Missing early-history values contain no future information; a neutral
    # value of one denotes error equal to the calibration normalization scale.
    return np.nan_to_num(memory, nan=1.0, posinf=1.0, neginf=1.0)


def router_state(
    frame: pd.DataFrame,
    columns: list[str],
    predictions: dict[str, np.ndarray],
    losses: np.ndarray,
    horizon: int,
) -> np.ndarray:
    return np.column_stack((
        base_router_state(frame, columns, predictions),
        matured_error_memory(losses, horizon),
    ))


def action_loss(y: np.ndarray, predictions: dict[str, np.ndarray], scale: float) -> np.ndarray:
    return np.column_stack([np.abs(y - predictions[name]) / scale for name in ACTIONS])


def q_row(state: np.ndarray, previous: int, action: int) -> np.ndarray:
    previous_onehot = np.zeros(len(ACTIONS))
    action_onehot = np.zeros(len(ACTIONS))
    previous_onehot[previous] = 1.0
    action_onehot[action] = 1.0
    return np.r_[state, previous_onehot, action_onehot]


def fit_fqi(states: np.ndarray, losses: np.ndarray, tail: float, switch: float, gamma: float = 0.8):
    """Full-information fitted Q iteration with previous action as endogenous state."""
    n, actions = losses.shape
    cutoff = np.quantile(losses, 0.9)
    reward = -losses - tail * np.maximum(losses - cutoff, 0.0)
    rows, immediate, next_rows = [], [], []
    for t in range(n - 1):
        for previous in range(actions):
            for action in range(actions):
                rows.append(q_row(states[t], previous, action))
                immediate.append(reward[t, action] - switch * (action != previous))
                next_rows.append((t + 1, action))
    x = np.asarray(rows)
    q_target = np.asarray(immediate)
    model = None
    iterations = 1 if gamma == 0 else 5
    for iteration in range(iterations):
        model = ExtraTreesRegressor(
            n_estimators=80, min_samples_leaf=8, max_features=0.75,
            n_jobs=-1, random_state=SEED + iteration,
        ).fit(x, q_target)
        # Batch the Bellman backup.  Calling predict once per transition is
        # mathematically identical but prohibitively slow on the full matrix.
        candidate_rows = []
        for t_next, previous in next_rows:
            candidate_rows.extend(q_row(states[t_next], previous, action) for action in range(actions))
        q_next = model.predict(np.asarray(candidate_rows)).reshape(len(next_rows), actions)
        future = q_next.max(axis=1)
        q_target = np.asarray(immediate) + gamma * future
    return model


def deploy(model, states: np.ndarray, initial_action: int, uncertainty_beta: float = 0.0) -> np.ndarray:
    actions = len(ACTIONS)
    selected = []
    previous = initial_action
    for state in states:
        candidates = np.asarray([q_row(state, previous, action) for action in range(actions)])
        if uncertainty_beta > 0 and hasattr(model, "estimators_"):
            tree_q = np.vstack([tree.predict(candidates) for tree in model.estimators_])
            score = tree_q.mean(axis=0) - uncertainty_beta * tree_q.std(axis=0)
        else:
            score = model.predict(candidates)
        previous = int(np.argmax(score))
        selected.append(previous)
    return np.asarray(selected)


def fit_contextual_bandit(states: np.ndarray, losses: np.ndarray):
    """Direct-method baseline: predicts immediate loss, with no Bellman backup."""
    model = ExtraTreesRegressor(
        n_estimators=160, min_samples_leaf=8, max_features=0.75,
        n_jobs=-1, random_state=SEED + 100,
    )
    model.fit(states, losses)
    return model


def deploy_contextual_bandit(model, states: np.ndarray) -> np.ndarray:
    predicted_losses = model.predict(states)
    return np.argmin(predicted_losses, axis=1)


def moving_block_bootstrap_upper(diff: np.ndarray, block: int = 24, repeats: int = 999) -> float:
    """One-sided 95% upper bound for paired mean loss difference."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < block:
        return float("inf")
    rng = np.random.default_rng(SEED + len(diff))
    starts = np.arange(len(diff) - block + 1)
    means = np.empty(repeats)
    blocks_needed = int(np.ceil(len(diff) / block))
    for repeat in range(repeats):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([diff[start:start + block] for start in chosen])[:len(diff)]
        means[repeat] = sample.mean()
    return float(np.quantile(means, 0.95))


def selected_forecast(predictions: dict[str, np.ndarray], actions: np.ndarray) -> np.ndarray:
    matrix = np.column_stack([predictions[name] for name in ACTIONS])
    return matrix[np.arange(len(actions)), actions]


def causal_inverse_error_ensemble(
    predictions: dict[str, np.ndarray], losses: np.ndarray, horizon: int, window: int = 168
) -> np.ndarray:
    """Strong non-RL baseline using only outcome-matured rolling losses."""
    available = pd.DataFrame(losses).shift(horizon)
    rolling = available.rolling(window, min_periods=max(4, horizon)).mean()
    rolling = rolling.ffill().fillna(1.0).to_numpy()
    inverse = 1.0 / np.maximum(rolling, 1e-6)
    weights = inverse / inverse.sum(axis=1, keepdims=True)
    matrix = np.column_stack([predictions[name] for name in BASE_EXPERTS])
    return np.sum(weights * matrix, axis=1)


def add_mixture_actions(
    predictions: dict[str, np.ndarray], base_losses: np.ndarray, horizon: int
) -> dict[str, np.ndarray]:
    enriched = dict(predictions)
    enriched["equal"] = np.mean(
        np.column_stack([predictions[name] for name in BASE_EXPERTS]), axis=1
    )
    enriched["adaptive"] = causal_inverse_error_ensemble(predictions, base_losses, horizon)
    return enriched


def run_unit(spec: DatasetSpec, horizon: int, fold: int, train, cal, test) -> dict:
    columns, cal_pred, test_pred = frozen_predictions(train, cal, test)
    y_cal = cal["target"].to_numpy(float)
    y_test = test["target"].to_numpy(float)
    scale = max(float(np.mean(np.abs(y_cal))), 1e-6)
    tune_start = int(len(cal) * 0.50)
    gate_start = int(len(cal) * 0.75)
    fit_end = tune_start - horizon
    tune_end = gate_start - horizon
    if fit_end <= 0:
        raise ValueError("Calibration fit block is too short for the target-time embargo")
    base_cal_losses = np.column_stack(
        [np.abs(y_cal - cal_pred[name]) / scale for name in BASE_EXPERTS]
    )
    base_test_losses = np.column_stack(
        [np.abs(y_test - test_pred[name]) / scale for name in BASE_EXPERTS]
    )
    cal_pred = add_mixture_actions(cal_pred, base_cal_losses, horizon)
    combined_base_pred = {
        name: np.r_[cal_pred[name], test_pred[name]] for name in BASE_EXPERTS
    }
    combined_base_losses = np.vstack((base_cal_losses, base_test_losses))
    combined_enriched = add_mixture_actions(combined_base_pred, combined_base_losses, horizon)
    test_pred = {name: combined_enriched[name][len(cal):] for name in ACTIONS}
    losses = action_loss(y_cal, cal_pred, scale)
    test_losses = action_loss(y_test, test_pred, scale)
    cal_state = router_state(cal, columns, cal_pred, base_cal_losses, horizon)
    # At the beginning of test, mature calibration errors remain available;
    # later test errors enter memory only after their own horizon has elapsed.
    combined_losses = np.vstack((base_cal_losses, base_test_losses))
    combined_frame = pd.concat((cal, test))
    combined_pred = combined_enriched
    combined_state = router_state(combined_frame, columns, combined_pred, combined_losses, horizon)
    test_state = combined_state[len(cal):]

    best_expert = int(np.argmin(losses.mean(axis=0)))
    early_best = int(np.argmin(losses[:fit_end].mean(axis=0)))
    equal_test = test_pred["equal"]
    adaptive_test = test_pred["adaptive"]
    best_test = test_pred[ACTIONS[best_expert]]

    tune_candidates = {
        name: cal_pred[name][tune_start:tune_end] for name in ACTIONS
    }
    fallback_name = min(
        tune_candidates,
        key=lambda name: mean_absolute_error(y_cal[tune_start:tune_end], tune_candidates[name]),
    )
    fallback_test = test_pred[fallback_name]

    candidates = []
    # Compact, pre-declared grid for tractable multi-dataset evaluation.  The
    # endpoints retain the contextual (gamma=0) and sequential hypotheses.
    for gamma in (0.0, 0.6):
        for tail in (0.0, 0.5):
            for switch in (0.0, 0.03):
                model = fit_fqi(cal_state[:fit_end], losses[:fit_end], tail, switch, gamma)
                for beta in (0.0, 1.0):
                    actions = deploy(model, cal_state[tune_start:tune_end], early_best, beta)
                    pred = selected_forecast(
                        {k: v[tune_start:tune_end] for k, v in cal_pred.items()}, actions
                    )
                    candidates.append(
                        (mean_absolute_error(y_cal[tune_start:tune_end], pred), gamma, tail, switch, beta, model)
                    )
    tune_mae, gamma, tail, switch, beta, model = min(candidates, key=lambda row: row[0])
    gate_actions = deploy(model, cal_state[gate_start:], early_best, beta)
    gate_rl = selected_forecast({k: v[gate_start:] for k, v in cal_pred.items()}, gate_actions)
    gate_fallback = (
        cal_pred[fallback_name][gate_start:]
    )
    gate_abs_rl = np.abs(y_cal[gate_start:] - gate_rl)
    gate_abs_fallback = np.abs(y_cal[gate_start:] - gate_fallback)
    # Pre-test safety gate: the one-sided block-bootstrap confidence bound for
    # paired loss difference must favor RL, with no material tail deterioration.
    gate_upper = moving_block_bootstrap_upper(gate_abs_rl - gate_abs_fallback)
    tune_actions = deploy(model, cal_state[tune_start:tune_end], early_best, beta)
    tune_rl = selected_forecast(
        {k: v[tune_start:tune_end] for k, v in cal_pred.items()}, tune_actions
    )
    tune_fallback = tune_candidates[fallback_name]
    tune_gain = float(np.mean(np.abs(y_cal[tune_start:tune_end] - tune_fallback)) - np.mean(np.abs(y_cal[tune_start:tune_end] - tune_rl)))
    gate_gain = float(gate_abs_fallback.mean() - gate_abs_rl.mean())
    gate_mean = gate_upper < 0.0 and tune_gain > 0.0 and gate_gain > 0.0
    gate_tail = np.quantile(gate_abs_rl, 0.90) <= 1.01 * np.quantile(gate_abs_fallback, 0.90)
    gate_open = bool(gate_mean and gate_tail)

    actions = deploy(model, test_state, early_best, beta)
    rl_test = selected_forecast(test_pred, actions)
    safe_test = rl_test if gate_open else fallback_test
    bandit = fit_contextual_bandit(cal_state[:fit_end], losses[:fit_end])
    bandit_actions = deploy_contextual_bandit(bandit, test_state)
    bandit_test = selected_forecast(test_pred, bandit_actions)
    abs_rl = np.abs(y_test - rl_test)
    abs_safe = np.abs(y_test - safe_test)
    oracle = np.min(
        np.abs(y_test[:, None] - np.column_stack([test_pred[n] for n in ACTIONS])), axis=1
    )
    return {
        "dataset": spec.name,
        "horizon": horizon,
        "fold": fold,
        "n_test": len(test),
        "validation_best": ACTIONS[best_expert],
        "tail_penalty": tail,
        "switch_penalty": switch,
        "gamma": gamma,
        "uncertainty_beta": beta,
        "tune_mae": tune_mae,
        "tune_gain": tune_gain,
        "gate_gain": gate_gain,
        "safety_gate_open": gate_open,
        "gate_upper_loss_difference": gate_upper,
        "fallback": fallback_name,
        "mae_daily": mean_absolute_error(y_test, test_pred["daily"]),
        "mae_weekly": mean_absolute_error(y_test, test_pred["weekly"]),
        "mae_equal": mean_absolute_error(y_test, equal_test),
        "mae_adaptive_ensemble": mean_absolute_error(y_test, adaptive_test),
        "mae_validation_best": mean_absolute_error(y_test, best_test),
        "mae_contextual_bandit": mean_absolute_error(y_test, bandit_test),
        "mae_rl": mean_absolute_error(y_test, rl_test),
        "mae_safe_rl": mean_absolute_error(y_test, safe_test),
        "q90_rl": float(np.quantile(abs_rl, 0.90)),
        "q90_safe_rl": float(np.quantile(abs_safe, 0.90)),
        "q95_safe_rl": float(np.quantile(abs_safe, 0.95)),
        "mae_oracle_lower_bound": float(np.mean(oracle)),
        "rl_switch_rate": float(np.mean(actions[1:] != actions[:-1])) if len(actions) > 1 else 0.0,
        "bandit_switch_rate": float(np.mean(bandit_actions[1:] != bandit_actions[:-1])) if len(bandit_actions) > 1 else 0.0,
        **{f"rl_share_{name}": float(np.mean(actions == i)) for i, name in enumerate(ACTIONS)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--output", type=Path, default=Path("results/rl_router_prototype"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    profiles = {}
    for dataset in args.datasets:
        spec = SPECS[dataset]
        hourly = load_hourly(spec)
        profiles[dataset] = {
            "rows": len(hourly), "start": str(hourly.index.min()), "end": str(hourly.index.max()),
            "target": spec.target, "missing": int(hourly.isna().sum().sum()),
        }
        for horizon in args.horizons:
            frame = feature_frame(hourly, spec.target, horizon)
            for fold, train, cal, test in chronological_folds(frame, spec, horizon):
                result = run_unit(spec, horizon, fold, train, cal, test)
                rows.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    details = pd.DataFrame(rows)
    details.to_csv(args.output / "fold_metrics.csv", index=False)
    summary = details.groupby(["dataset", "horizon"], as_index=False).agg(
        folds=("fold", "count"), mae_daily=("mae_daily", "mean"),
        mae_equal=("mae_equal", "mean"), mae_adaptive_ensemble=("mae_adaptive_ensemble", "mean"),
        mae_validation_best=("mae_validation_best", "mean"),
        mae_contextual_bandit=("mae_contextual_bandit", "mean"),
        mae_rl=("mae_rl", "mean"), mae_safe_rl=("mae_safe_rl", "mean"),
        oracle_lower_bound=("mae_oracle_lower_bound", "mean"),
        switch_rate=("rl_switch_rate", "mean"),
    )
    summary["rl_vs_validation_pct"] = 100 * (summary.mae_validation_best - summary.mae_rl) / summary.mae_validation_best
    summary["rl_vs_equal_pct"] = 100 * (summary.mae_equal - summary.mae_rl) / summary.mae_equal
    summary["safe_rl_vs_validation_pct"] = 100 * (summary.mae_validation_best - summary.mae_safe_rl) / summary.mae_validation_best
    summary["safe_rl_vs_equal_pct"] = 100 * (summary.mae_equal - summary.mae_safe_rl) / summary.mae_equal
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "dataset_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
