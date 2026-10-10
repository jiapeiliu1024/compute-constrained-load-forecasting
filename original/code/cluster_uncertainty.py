"""Cluster-bootstrap uncertainty for the expanded formal policy matrix.

The resampling unit is a dataset--chronological-fold block. Horizons and
budgets within the same block are kept together, so the 240 configuration rows
are not treated as independent replicates.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results")
OUT = ROOT / "expanded_rl_summary"
DATASETS = ("steel", "tetouan", "uci_ecl", "opsd")
ALGORITHMS = ("contextual", "fqi", "double_fqi", "neural_fqi")
SEED = 20260902
REPS = 20000


def load_main() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        frames.append(pd.read_csv(
            ROOT / f"rl_core_matrix_v2_{dataset}" / "policy_metrics.csv"
        ))
    return pd.concat(frames, ignore_index=True)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    frame = load_main()
    rng = np.random.default_rng(SEED)
    clusters = sorted(set(zip(frame.dataset, frame.fold)))

    # One mean per dataset--fold cluster and algorithm, retaining all twelve
    # horizon--budget configurations inside that cluster.
    means = (frame.groupby(["dataset", "fold", "algorithm"], as_index=False)
                  .gain_vs_static_pct.mean())
    wide = means.pivot(index=["dataset", "fold"], columns="algorithm",
                       values="gain_vs_static_pct").loc[clusters]

    draws = rng.integers(0, len(wide), size=(REPS, len(wide)))
    rows = []
    contextual = wide["contextual"].to_numpy(float)
    for algorithm in ALGORITHMS:
        values = wide[algorithm].to_numpy(float)
        boot = values[draws].mean(axis=1)
        lo, hi = percentile_interval(boot)
        delta = values - contextual
        delta_boot = delta[draws].mean(axis=1)
        dlo, dhi = percentile_interval(delta_boot)
        rows.append({
            "algorithm": algorithm,
            "cluster_count": len(wide),
            "mean_gain_pct": values.mean(),
            "ci95_low_pct": lo,
            "ci95_high_pct": hi,
            "mean_delta_vs_contextual_pp": delta.mean(),
            "delta_ci95_low_pp": dlo,
            "delta_ci95_high_pp": dhi,
            "bootstrap_reps": REPS,
            "seed": SEED,
        })

    # Dataset-specific intervals resample the five chronological folds and
    # preserve the twelve within-fold configurations.
    dataset_rows = []
    for dataset in DATASETS:
        block = wide.loc[dataset]
        local_draws = rng.integers(0, len(block), size=(REPS, len(block)))
        for algorithm in ALGORITHMS:
            values = block[algorithm].to_numpy(float)
            boot = values[local_draws].mean(axis=1)
            lo, hi = percentile_interval(boot)
            dataset_rows.append({
                "dataset": dataset,
                "algorithm": algorithm,
                "fold_clusters": len(block),
                "mean_gain_pct": values.mean(),
                "ci95_low_pct": lo,
                "ci95_high_pct": hi,
                "bootstrap_reps": REPS,
                "seed": SEED,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "cluster_bootstrap_algorithm.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(
        OUT / "cluster_bootstrap_dataset_algorithm.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
