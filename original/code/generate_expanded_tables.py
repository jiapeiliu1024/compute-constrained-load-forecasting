"""Generate MDPI-ready tabular fragments from audited expanded results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results")
SRC = Path("artifacts/tables")
LABEL = {
    "steel": "Steel", "tetouan": "Tetouan", "uci_ecl": "UCI--ECL",
    "opsd": "OPSD", "contextual": "Contextual", "fqi": "FQI",
    "double_fqi": "Double FQI", "neural_fqi": "Neural FQI",
}


def write(name: str, lines: list[str]):
    (SRC / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    profiles = pd.read_csv(ROOT / "expanded_rl_summary" / "dataset_profiles.csv")
    profile_lines = [
        r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"Dataset & Hourly rows & Period start & Period end & Formal folds & Role \\",
        r"\midrule",
    ]
    for row in profiles.itertuples():
        folds = 2 if row.dataset == "appliances" else 5
        role = "Boundary" if row.dataset == "appliances" else "Main"
        profile_lines.append(
            f"{LABEL.get(row.dataset, 'Appliances')} & {row.hourly_rows:,} & "
            f"{str(row.start)[:10]} & {str(row.end)[:10]} & {folds} & {role} \\\\" 
        )
    profile_lines += [r"\bottomrule", r"\end{tabular}"]
    write("expanded_dataset_tabular.tex", profile_lines)

    alg = pd.read_csv(ROOT / "expanded_rl_summary" / "algorithm_summary.csv")
    pair = pd.read_csv(ROOT / "expanded_rl_summary" / "pairwise_vs_contextual.csv").set_index("candidate")
    uncertainty = pd.read_csv(
        ROOT / "expanded_rl_summary" / "cluster_bootstrap_algorithm.csv"
    ).set_index("algorithm")
    alg_lines = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Policy & Mean gain (\%) & Cluster 95\% CI & Median gain (\%) & Positive units & Oracle gap (\%) & $\Delta$ vs. contextual (95\% CI; pp) \\",
        r"\midrule",
    ]
    for row in alg.itertuples():
        u = uncertainty.loc[row.algorithm]
        delta = 0.0 if row.algorithm == "contextual" else pair.loc[row.algorithm, "mean_gain_difference_pct_points"]
        delta_cell = "Reference" if row.algorithm == "contextual" else (
            f"{delta:+.2f} [{u.delta_ci95_low_pp:+.2f}, {u.delta_ci95_high_pp:+.2f}]"
        )
        alg_lines.append(
            f"{LABEL[row.algorithm]} & {row.mean_gain_pct:.2f} & "
            f"[{u.ci95_low_pct:.2f}, {u.ci95_high_pct:.2f}] & {row.median_gain_pct:.2f} & "
            f"{int(row.positive_units)}/{int(row.units)} & {row.mean_oracle_gap_pct:.2f} & "
            f"{delta_cell} \\\\"
        )
    alg_lines += [r"\bottomrule", r"\end{tabular}"]
    write("expanded_algorithm_tabular.tex", alg_lines)

    ds = pd.read_csv(ROOT / "expanded_rl_summary" / "dataset_algorithm_summary.csv")
    gain = ds.pivot(index="dataset", columns="algorithm", values="mean_gain_pct")
    pos = ds.pivot(index="dataset", columns="algorithm", values="positive_units")
    ds_lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Dataset & Contextual & FQI & Double FQI & Neural FQI \\", r"\midrule",
    ]
    for dataset in ("steel", "tetouan", "uci_ecl", "opsd"):
        cells = [f"{gain.loc[dataset, a]:+.2f} ({int(pos.loc[dataset, a])}/60)" for a in ("contextual", "fqi", "double_fqi", "neural_fqi")]
        ds_lines.append(f"{LABEL[dataset]} & " + " & ".join(cells) + r" \\")
    ds_lines += [r"\bottomrule", r"\end{tabular}"]
    write("expanded_dataset_algorithm_tabular.tex", ds_lines)

    depth = pd.read_csv(ROOT / "rl_depth_audit_v2" / "policy_metrics.csv")
    sensitivity_rows = []
    for family, parameter, variants in (
        ("Backup depth", "1, 3, 6, 12", ["depth_1", "depth_3", "depth_6", "depth_12"]),
        ("Discount $\\gamma$", "0, 0.3, 0.6, 0.9, 1", ["contextual_full", "gamma_0.3", "gamma_0.6", "gamma_0.9", "stationary_fqi"]),
        ("Tree count", "8, 16, 32, 64", ["trees_8", "trees_16", "trees_32", "trees_64"]),
        ("Random seed", "five fixed seeds", [f"seed_{20260821+i}" for i in range(5)]),
    ):
        means = [depth.loc[depth.variant == variant, "gain_vs_static_pct"].mean() for variant in variants]
        sensitivity_rows.append((family, parameter, min(means), max(means), max(means) - min(means)))
    sens_lines = [
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Audit & Values & Minimum mean gain (\%) & Maximum mean gain (\%) & Range (pp) \\", r"\midrule",
    ]
    for family, parameter, lo, hi, span in sensitivity_rows:
        sens_lines.append(f"{family} & {parameter} & {lo:.2f} & {hi:.2f} & {span:.2f} \\\\ ")
    sens_lines += [r"\bottomrule", r"\end{tabular}"]
    write("expanded_sensitivity_tabular.tex", sens_lines)

    seq0 = pd.read_csv(ROOT / "rl_sequential_audit_v1" / "summary.csv")
    penalty = pd.read_csv(ROOT / "expanded_rl_summary" / "sequential_penalty_summary.csv")
    seq_lines = [
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Environment & Policy & Mean objective & Mean switch rate \\", r"\midrule",
    ]
    for name in ("contextual", "stationary_fqi", "finite_horizon_fqi"):
        row = seq0[seq0.algorithm == name].iloc[0]
        seq_lines.append(f"Budget-only & {name.replace('_', ' ')} & {row.mean_objective:.5f} & {row.mean_switch_rate:.3f} \\\\ ")
    for setting in (0.01, 0.05, 0.10):
        for name in ("switch_contextual", "switch_fqi"):
            row = penalty[(penalty.penalty_setting == setting) & (penalty.algorithm == name)].iloc[0]
            seq_lines.append(f"Switch cost {setting:.2f} & {name.replace('switch_', '').upper() if name.endswith('fqi') else 'Contextual'} & {row.mean_objective:.5f} & {row.mean_switch_rate:.3f} \\\\ ")
    seq_lines += [r"\bottomrule", r"\end{tabular}"]
    write("expanded_sequential_tabular.tex", seq_lines)

    gate = pd.read_csv(ROOT / "expanded_rl_summary" / "gate_ablation_summary.csv")
    gate_ds = gate.groupby("dataset").agg(
        configurations=("horizon", "size"),
        activation_rate=("activation_rate", "mean"),
        selected_gain=("mean_selected_gain_pct", "mean"),
        gated_gain=("mean_gated_gain_pct", "mean"),
        harmful=("harmful_activations", "sum"),
        missed=("missed_gains", "sum"),
    ).reset_index()
    gate_lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Dataset & Activation (\%) & Selected gain (\%) & Gated gain (\%) & Harmful activations & Missed gains \\", r"\midrule",
    ]
    for row in gate_ds.itertuples():
        gate_lines.append(f"{LABEL[row.dataset]} & {100*row.activation_rate:.1f} & {row.selected_gain:.2f} & {row.gated_gain:.2f} & {int(row.harmful)} & {int(row.missed)} \\\\ ")
    gate_lines += [r"\bottomrule", r"\end{tabular}"]
    write("expanded_gate_tabular.tex", gate_lines)


if __name__ == "__main__":
    main()
