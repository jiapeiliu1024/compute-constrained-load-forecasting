"""Fail-closed integrity audit for the expanded offline-policy experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results")
MAIN_DATASETS = ("steel", "tetouan", "uci_ecl", "opsd")
ALGORITHMS = {"contextual", "fqi", "double_fqi", "neural_fqi"}
HORIZONS = {6, 12, 24, 48}
BUDGETS = {60, 108, 180}
SHARES = ["share_daily", "share_weekly", "share_ridge", "share_hgb"]


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def audit_matrix(dataset: str, folds: int, boundary: bool = False):
    path = ROOT / f"rl_core_matrix_v2_{dataset}" / "policy_metrics.csv"
    frame = pd.read_csv(path)
    expected_units = len(HORIZONS) * len(BUDGETS) * folds
    keys = frame.groupby(["dataset", "horizon", "budget", "fold"], sort=False)
    require(len(keys) == expected_units, f"{dataset}: {len(keys)} units, expected {expected_units}")
    require(set(frame.horizon) == HORIZONS, f"{dataset}: horizon coverage")
    require(set(frame.budget) == BUDGETS, f"{dataset}: budget coverage")
    require(set(frame.fold) == set(range(1, folds + 1)), f"{dataset}: fold coverage")
    for key, group in keys:
        require(len(group) == 4, f"{dataset} {key}: algorithm row count")
        require(set(group.algorithm) == ALGORITHMS, f"{dataset} {key}: algorithms")
    numeric = [
        "mae", "static_mae", "oracle_mae", "gain_vs_static_pct",
        "oracle_gap_pct", "mean_cost", "switch_rate", "max_daily_spend",
        *SHARES,
    ]
    require(np.isfinite(frame[numeric].to_numpy(float)).all(), f"{dataset}: non-finite metric")
    require((frame.mae > 0).all() and (frame.static_mae > 0).all(), f"{dataset}: non-positive MAE")
    require((frame.max_daily_spend <= frame.budget).all(), f"{dataset}: budget violation")
    require((frame.oracle_mae <= frame.mae + 1e-9).all(), f"{dataset}: oracle exceeds policy")
    require((frame.oracle_mae <= frame.static_mae + 1e-9).all(), f"{dataset}: oracle exceeds static")
    share_sum = frame[SHARES].sum(axis=1)
    require(np.allclose(share_sum, 1.0, atol=1e-8), f"{dataset}: action shares do not sum to one")
    require(((frame.switch_rate >= 0) & (frame.switch_rate <= 1)).all(), f"{dataset}: switch rate")
    return {
        "dataset": dataset,
        "role": "boundary" if boundary else "main",
        "rows": int(len(frame)),
        "units": int(len(keys)),
        "budget_violations": int((frame.max_daily_spend > frame.budget).sum()),
        "max_share_error": float(np.max(np.abs(share_sum - 1))),
        "max_oracle_order_error": float(np.max(frame.oracle_mae - frame.mae)),
    }


def audit_depth():
    path = ROOT / "rl_depth_audit_v2" / "policy_metrics.csv"
    frame = pd.read_csv(path)
    require(set(frame.dataset) == {"steel", "tetouan"}, "depth: datasets")
    require(set(frame.fold) == {1, 2, 3, 4, 5}, "depth: folds")
    required_variants = {
        "contextual_full", "stationary_fqi", "depth_1", "depth_3",
        "depth_6", "depth_12", "contextual_no_exogenous",
        "fqi_no_exogenous", "contextual_no_history", "fqi_no_history",
        "contextual_no_calendar", "fqi_no_calendar", "contextual_calendar",
        "fqi_calendar", "contextual_no_budget", "fqi_no_budget",
        "gamma_0.3", "gamma_0.6", "gamma_0.9", "trees_8", "trees_16",
        "trees_32", "trees_64", "seed_20260821", "seed_20260822",
        "seed_20260823", "seed_20260824", "seed_20260825",
    }
    require(required_variants <= set(frame.variant), "depth: missing variants")
    require((frame.max_daily_spend <= 108).all(), "depth: budget violation")
    require(np.isfinite(frame.select_dtypes("number").to_numpy()).all(), "depth: non-finite")
    counts = frame.groupby(["dataset", "fold", "variant"]).size()
    require((counts == 1).all(), "depth: duplicate variant")
    return {"rows": int(len(frame)), "variants": int(frame.variant.nunique())}


def audit_hyper():
    frame = pd.read_csv(ROOT / "rl_hyper_grid_v1" / "grid_metrics.csv")
    require(set(frame.gamma) == {0.0, 0.3, 0.6, 0.9, 1.0}, "hyper: gamma grid")
    require(set(frame.backups) == {1, 3, 6, 12}, "hyper: backup grid")
    require(set(frame.dataset) == {"steel", "tetouan"}, "hyper: datasets")
    require(set(frame.fold) == {1, 2}, "hyper: folds")
    require(len(frame) == 80, "hyper: expected 80 cells")
    require((frame.max_daily_spend <= 108).all(), "hyper: budget violation")
    return {"rows": int(len(frame)), "cells": 20, "paired_units": 4}


def audit_sequential():
    frame = pd.read_csv(ROOT / "rl_sequential_audit_v1" / "policy_metrics.csv")
    required = {
        "contextual", "stationary_fqi", "finite_horizon_fqi",
        "switch_contextual", "switch_fqi",
    }
    require(set(frame.algorithm) == required, "sequential: algorithms")
    require(set(frame.dataset) == {"steel", "tetouan"}, "sequential: datasets")
    require(set(frame.fold) == {1, 2, 3}, "sequential: folds")
    require(len(frame) == 30, "sequential: expected 30 evaluations")
    require((frame.max_daily_spend <= 108).all(), "sequential: budget violation")
    require(np.isfinite(frame.select_dtypes("number").to_numpy()).all(), "sequential: non-finite")
    penalty_dirs = {
        "0.01": "rl_sequential_audit_switch_0p01",
        "0.05": "rl_sequential_audit_v1",
        "0.10": "rl_sequential_audit_switch_0p1",
    }
    penalty_rows = {}
    for penalty, directory in penalty_dirs.items():
        block = pd.read_csv(ROOT / directory / "policy_metrics.csv")
        switch = block[block.algorithm.isin({"switch_contextual", "switch_fqi"})]
        require(len(switch) == 12, f"sequential {penalty}: expected 12 switching evaluations")
        require((switch.max_daily_spend <= 108).all(), f"sequential {penalty}: budget violation")
        require(np.isfinite(switch.select_dtypes("number").to_numpy()).all(),
                f"sequential {penalty}: non-finite")
        penalty_rows[penalty] = int(len(switch))
    return {"rows": int(len(frame)), "paired_units": 6,
            "switching_penalty_rows": penalty_rows}


def audit_gate(dataset: str, folds: int):
    frame = pd.read_csv(ROOT / f"rl_gate_ablation_v1_{dataset}" / "gate_metrics.csv")
    expected = len(HORIZONS) * len(BUDGETS) * folds
    require(len(frame) == expected, f"gate {dataset}: expected {expected} rows")
    require(set(frame.horizon) == HORIZONS, f"gate {dataset}: horizons")
    require(set(frame.budget) == BUDGETS, f"gate {dataset}: budgets")
    require(set(frame.fold) == set(range(1, folds + 1)), f"gate {dataset}: folds")
    require((frame.max_dynamic_daily_spend <= frame.budget).all(),
            f"gate {dataset}: budget violation")
    require(set(frame.gate_open.unique()) <= {0, 1}, f"gate {dataset}: gate values")
    require(np.isfinite(frame.select_dtypes("number").to_numpy()).all(),
            f"gate {dataset}: non-finite")
    return {"dataset": dataset, "rows": int(len(frame)),
            "activation_rate": float(frame.gate_open.mean()),
            "budget_violations": int((frame.max_dynamic_daily_spend > frame.budget).sum())}


def main():
    report = {
        "main": [audit_matrix(dataset, 5) for dataset in MAIN_DATASETS],
        "boundary": audit_matrix("appliances", 2, boundary=True),
        "depth": audit_depth(),
        "hyperparameter_grid": audit_hyper(),
        "sequential": audit_sequential(),
        "gate_main": [audit_gate(dataset, 5) for dataset in MAIN_DATASETS],
        "gate_boundary": audit_gate("appliances", 2),
    }
    output = ROOT / "expanded_rl_integrity_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
