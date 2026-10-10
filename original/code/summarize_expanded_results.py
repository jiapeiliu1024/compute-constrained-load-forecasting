"""Create auditable descriptive summaries without test-set model selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results")
OUT = ROOT / "expanded_rl_summary"
DATASETS = ("steel", "tetouan", "uci_ecl", "opsd")
ALGORITHMS = ("contextual", "fqi", "double_fqi", "neural_fqi")
KEYS = ["dataset", "horizon", "budget", "fold"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.concat([
        pd.read_csv(ROOT / f"rl_core_matrix_v2_{dataset}" / "policy_metrics.csv")
        for dataset in DATASETS
    ], ignore_index=True)
    core.to_csv(OUT / "all_main_policy_metrics.csv", index=False)
    configuration = core.groupby(
        ["dataset", "horizon", "budget", "algorithm"], sort=False
    ).agg(
        folds=("fold", "nunique"),
        mean_mae=("mae", "mean"),
        mean_static_mae=("static_mae", "mean"),
        mean_gain_pct=("gain_vs_static_pct", "mean"),
        median_gain_pct=("gain_vs_static_pct", "median"),
        std_gain_pct=("gain_vs_static_pct", "std"),
        positive_folds=("gain_vs_static_pct", lambda x: int((x > 0).sum())),
        mean_oracle_gap_pct=("oracle_gap_pct", "mean"),
        mean_cost=("mean_cost", "mean"),
        mean_switch_rate=("switch_rate", "mean"),
        max_daily_spend=("max_daily_spend", "max"),
    ).reset_index()
    configuration.to_csv(OUT / "configuration_summary.csv", index=False)

    algorithm = core.groupby("algorithm", sort=False).agg(
        units=("gain_vs_static_pct", "size"),
        configurations=("dataset", lambda x: int(len(x) / 5)),
        mean_gain_pct=("gain_vs_static_pct", "mean"),
        median_gain_pct=("gain_vs_static_pct", "median"),
        positive_units=("gain_vs_static_pct", lambda x: int((x > 0).sum())),
        mean_oracle_gap_pct=("oracle_gap_pct", "mean"),
        mean_cost=("mean_cost", "mean"),
        max_daily_spend=("max_daily_spend", "max"),
    ).reset_index()
    algorithm["positive_rate"] = algorithm.positive_units / algorithm.units
    algorithm.to_csv(OUT / "algorithm_summary.csv", index=False)

    dataset_summary = core.groupby(["dataset", "algorithm"], sort=False).agg(
        units=("gain_vs_static_pct", "size"),
        mean_gain_pct=("gain_vs_static_pct", "mean"),
        median_gain_pct=("gain_vs_static_pct", "median"),
        positive_units=("gain_vs_static_pct", lambda x: int((x > 0).sum())),
        mean_oracle_gap_pct=("oracle_gap_pct", "mean"),
    ).reset_index()
    dataset_summary.to_csv(OUT / "dataset_algorithm_summary.csv", index=False)

    pivot = core.pivot(index=KEYS, columns="algorithm", values="gain_vs_static_pct")
    pair_rows = []
    for candidate in ALGORITHMS[1:]:
        delta = pivot[candidate] - pivot["contextual"]
        pair_rows.append({
            "candidate": candidate,
            "paired_units": int(delta.size),
            "mean_gain_difference_pct_points": float(delta.mean()),
            "median_gain_difference_pct_points": float(delta.median()),
            "candidate_better_units": int((delta > 0).sum()),
            "contextual_better_units": int((delta < 0).sum()),
            "ties": int(np.isclose(delta, 0).sum()),
        })
    pd.DataFrame(pair_rows).to_csv(OUT / "pairwise_vs_contextual.csv", index=False)

    best = core.loc[core.groupby(KEYS).mae.idxmin(), KEYS + ["algorithm", "mae"]]
    best.algorithm.value_counts().rename_axis("algorithm").reset_index(
        name="best_units"
    ).to_csv(OUT / "best_algorithm_counts.csv", index=False)

    boundary = pd.read_csv(
        ROOT / "rl_core_matrix_v2_appliances" / "policy_metrics.csv"
    )
    boundary.groupby(["horizon", "budget", "algorithm"], sort=False).agg(
        folds=("fold", "nunique"),
        mean_gain_pct=("gain_vs_static_pct", "mean"),
        mean_oracle_gap_pct=("oracle_gap_pct", "mean"),
        positive_folds=("gain_vs_static_pct", lambda x: int((x > 0).sum())),
        max_daily_spend=("max_daily_spend", "max"),
    ).reset_index().to_csv(OUT / "appliances_boundary_summary.csv", index=False)

    gate = pd.concat([
        pd.read_csv(ROOT / f"rl_gate_ablation_v1_{dataset}" / "gate_metrics.csv")
        for dataset in DATASETS
    ], ignore_index=True)
    gate.to_csv(OUT / "all_main_gate_metrics.csv", index=False)
    gate.groupby(["dataset", "horizon", "budget"]).agg(
        folds=("fold", "size"),
        mean_selected_gain_pct=("dynamic_gain_pct", "mean"),
        mean_gated_gain_pct=("gated_gain_pct", "mean"),
        activation_rate=("gate_open", "mean"),
        harmful_activations=("harmful_activation", "sum"),
        missed_gains=("missed_gain", "sum"),
    ).reset_index().to_csv(OUT / "gate_ablation_summary.csv", index=False)

    sequential_frames = []
    for penalty, directory in (
        (0.01, "rl_sequential_audit_switch_0p01"),
        (0.05, "rl_sequential_audit_v1"),
        (0.10, "rl_sequential_audit_switch_0p1"),
    ):
        sequential = pd.read_csv(ROOT / directory / "policy_metrics.csv")
        sequential = sequential[sequential.algorithm.str.startswith("switch_")].copy()
        sequential["penalty_setting"] = penalty
        sequential_frames.append(sequential)
    sequential_all = pd.concat(sequential_frames, ignore_index=True)
    sequential_all.to_csv(OUT / "sequential_penalty_metrics.csv", index=False)
    sequential_all.groupby(["penalty_setting", "algorithm"]).agg(
        paired_units=("fold", "size"),
        mean_objective=("normalized_objective", "mean"),
        mean_switch_rate=("switch_rate", "mean"),
        max_daily_spend=("max_daily_spend", "max"),
    ).reset_index().to_csv(OUT / "sequential_penalty_summary.csv", index=False)

    print(algorithm.to_string(index=False))
    print("\nPairwise gain differences versus contextual (percentage points):")
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    main()
