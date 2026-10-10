"""Generate eight vector figures from the expanded offline-policy audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
try:
    from reportlab.pdfgen import canvas
except ImportError:
    try:
        import simple_canvas as canvas
    except ImportError:
        from . import simple_canvas as canvas


ROOT = Path("results")
OUT = Path("artifacts/figures")
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1080, 650
ALGORITHMS = ["contextual", "fqi", "double_fqi", "neural_fqi"]
ALGORITHM_LABELS = ["Contextual", "FQI", "Double FQI", "Neural FQI"]
DATASETS = ["steel", "tetouan", "uci_ecl", "opsd"]
DATASET_LABELS = {
    "steel": "Steel", "tetouan": "Tetouan",
    "uci_ecl": "UCI-ECL", "opsd": "OPSD",
}
COLORS = {
    "contextual": (0.10, 0.34, 0.58),
    "fqi": (0.85, 0.38, 0.12),
    "double_fqi": (0.20, 0.55, 0.32),
    "neural_fqi": (0.48, 0.28, 0.65),
}


def load_core() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        path = ROOT / f"rl_core_matrix_v2_{dataset}" / "policy_metrics.csv"
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def base(path: Path, title: str, subtitle: str = ""):
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle(title)
    c.setFont("Helvetica-Bold", 25)
    c.drawString(42, H - 38, title)
    if subtitle:
        c.setFont("Helvetica", 15)
        c.setFillColorRGB(0.28, 0.28, 0.28)
        c.drawString(42, H - 55, subtitle)
        c.setFillColorRGB(0, 0, 0)
    return c


def heat_color(value: float, limit: float):
    value = float(np.clip(value / limit, -1, 1))
    if value >= 0:
        return (1 - 0.78 * value, 1 - 0.55 * value, 1 - 0.18 * value)
    value = -value
    return (1 - 0.14 * value, 1 - 0.55 * value, 1 - 0.72 * value)


def heatmap(c, matrix, row_labels, col_labels, x, y, width, height,
            limit, value_format="{:+.1f}", show_rows=True):
    matrix = np.asarray(matrix, dtype=float)
    rows, cols = matrix.shape
    cw, ch = width / cols, height / rows
    c.setFont("Helvetica-Bold", 16)
    for j, label in enumerate(col_labels):
        c.drawCentredString(x + (j + 0.5) * cw, y + height + 8, label)
    for i in range(rows):
        yy = y + height - (i + 1) * ch
        if show_rows:
            c.setFont("Helvetica", 16)
            c.setFillColorRGB(0, 0, 0)
            c.drawRightString(x - 6, yy + ch / 2 - 2.5, row_labels[i])
        for j in range(cols):
            value = matrix[i, j]
            c.setFillColorRGB(*heat_color(value, limit))
            c.rect(x + j * cw, yy, cw - 1, ch - 1, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(
                x + (j + 0.5) * cw, yy + ch / 2 - 2.5,
                value_format.format(value),
            )


def algorithm_comparison(core: pd.DataFrame):
    c = base(
        OUT / "algorithm_comparison.pdf",
        "Algorithm Comparison Across Horizons and Budgets",
        "Cells show mean MAE gain (%) relative to the tune-selected feasible static forecaster; five chronological folds per cell.",
    )
    row_keys = [(d, h) for d in DATASETS for h in (6, 12, 24, 48)]
    row_labels = [f"{DATASET_LABELS[d]} - {h} h" for d, h in row_keys]
    panel_w, panel_h, x0, gap, y0 = 262, 475, 135, 42, 45
    for panel, daily_budget in enumerate((60, 108, 180)):
        matrix = []
        for dataset, horizon in row_keys:
            subset = core[(core.dataset == dataset) &
                          (core.horizon == horizon) &
                          (core.budget == daily_budget)]
            matrix.append([
                subset.loc[subset.algorithm == algorithm,
                           "gain_vs_static_pct"].mean()
                for algorithm in ALGORITHMS
            ])
        x = x0 + panel * (panel_w + gap)
        c.setFont("Helvetica-Bold", 19)
        c.drawCentredString(x + panel_w / 2, y0 + panel_h + 25,
                            f"Daily budget {daily_budget}")
        heatmap(c, matrix, row_labels, ["Ctx", "FQI", "Double", "Neural"], x, y0,
                panel_w, panel_h, limit=35, show_rows=(panel == 0))
    c.setFont("Helvetica", 16)
    c.drawString(42, 18, "Ctx = contextual; Double = Double FQI; Neural = neural FQI. Blue = improvement; red = degradation.")
    c.save()


def hyperparameter_grid():
    data = pd.read_csv(ROOT / "rl_hyper_grid_v1" / "summary.csv")
    gammas = [0.0, 0.3, 0.6, 0.9, 1.0]
    depths = [1, 3, 6, 12]
    matrix = [[
        data.loc[(data.gamma == gamma) & (data.backups == backup),
                 "mean_gain_pct"].iloc[0]
        for backup in depths
    ] for gamma in gammas]
    c = base(
        OUT / "hyperparameter_heatmap.pdf",
        "Bellman Hyperparameter Surface",
        "Mean gain over pre-specified Steel and Tetouan folds at 12 h and budget 108; no cell is selected on test.",
    )
    heatmap(c, matrix, [f"gamma = {g:g}" for g in gammas],
            [f"{d} backup" if d == 1 else f"{d} backups" for d in depths],
            260, 135, 600, 360, limit=30, value_format="{:+.2f}")
    c.setFont("Helvetica", 15)
    c.drawString(260, 105, "For gamma = 0, additional backup labels repeat the identical contextual objective by construction.")
    c.save()


def hyperparameter_grid_ascii():
    """Render the gamma labels with PDF-core ASCII fonts for portable viewing."""
    data = pd.read_csv(ROOT / "rl_hyper_grid_v1" / "summary.csv")
    gammas = [0.0, 0.3, 0.6, 0.9, 1.0]
    depths = [1, 3, 6, 12]
    matrix = [[
        data.loc[(data.gamma == gamma) & (data.backups == backup),
                 "mean_gain_pct"].iloc[0]
        for backup in depths
    ] for gamma in gammas]
    c = base(
        OUT / "hyperparameter_heatmap.pdf",
        "Bellman Hyperparameter Surface",
        "Mean gain over pre-specified Steel and Tetouan folds at 12 h and budget 108; no cell is selected on test.",
    )
    heatmap(c, matrix, [f"gamma = {g:g}" for g in gammas],
            [f"{d} backup" if d == 1 else f"{d} backups" for d in depths],
            260, 135, 600, 360, limit=30, value_format="{:+.2f}")
    c.setFont("Helvetica", 15)
    c.drawString(260, 105, "For gamma = 0, additional backup labels repeat the identical contextual objective by construction.")
    c.save()


def action_distribution(core: pd.DataFrame):
    action_cols = ["share_daily", "share_weekly", "share_ridge", "share_hgb"]
    rows, labels = [], []
    subset = core[core.budget == 108]
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            block = subset[(subset.dataset == dataset) &
                           (subset.algorithm == algorithm)]
            rows.append([100 * block[column].mean() for column in action_cols])
            labels.append(f"{DATASET_LABELS[dataset]} - {ALGORITHM_LABELS[ALGORITHMS.index(algorithm)]}")
    c = base(
        OUT / "action_distribution_heatmap.pdf",
        "Action Allocation Under Budget 108",
        "Shares aggregate four horizons and five chronological folds; every row sums to 100% up to rounding.",
    )
    heatmap(c, rows, labels, ["Daily", "Weekly", "Ridge", "HGB"],
            225, 70, 700, 505, limit=100, value_format="{:.1f}%")
    c.save()


def budget_trajectories():
    c = base(
        OUT / "daily_budget_trajectories.pdf",
        "Representative Calendar-Day Budget Trajectories",
        "Fold 1, 12-hour horizon, daily budget 108. Curves show remaining credits after each action on the first complete test day.",
    )
    for panel, dataset in enumerate(("steel", "opsd")):
        traces = pd.read_csv(
            ROOT / f"rl_core_matrix_v2_{dataset}" / "policy_traces.csv"
        )
        traces = traces[(traces.horizon == 12) & (traces.budget == 108)]
        x0, y0, x1, y1 = 85 + panel * 505, 100, 520 + panel * 505, 520
        c.setStrokeColorRGB(0.1, 0.1, 0.1)
        c.line(x0, y0, x1, y0); c.line(x0, y0, x0, y1)
        for tick in (0, 36, 72, 108):
            yy = y0 + tick / 108 * (y1 - y0)
            c.setStrokeColorRGB(0.88, 0.88, 0.88); c.line(x0, yy, x1, yy)
            c.setFillColorRGB(0.2, 0.2, 0.2); c.setFont("Helvetica", 16)
            c.drawRightString(x0 - 6, yy - 3, str(tick))
        for hour in (0, 6, 12, 18, 23):
            xx = x0 + hour / 23 * (x1 - x0)
            c.setFillColorRGB(0.2, 0.2, 0.2); c.drawCentredString(xx, y0 - 18, str(hour))
        for algorithm in ALGORITHMS:
            block = traces[traces.algorithm == algorithm].sort_values("timestamp").head(24)
            values = block.remaining_budget.to_numpy(float)
            points = [(x0 + i / 23 * (x1 - x0), y0 + value / 108 * (y1 - y0))
                      for i, value in enumerate(values)]
            c.setStrokeColorRGB(*COLORS[algorithm]); c.setLineWidth(1.8)
            for p0, p1 in zip(points[:-1], points[1:]): c.line(*p0, *p1)
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 18)
        c.drawCentredString((x0 + x1) / 2, y1 + 18, DATASET_LABELS[dataset])
        c.setFont("Helvetica", 16); c.drawCentredString((x0 + x1) / 2, y0 - 35, "Hour of day")
    x, y = 150, 24
    for algorithm in ALGORITHMS:
        c.setStrokeColorRGB(*COLORS[algorithm]); c.setLineWidth(3); c.line(x, y, x + 25, y)
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 16)
        c.drawString(x + 30, y - 3, ALGORITHM_LABELS[ALGORITHMS.index(algorithm)])
        x += 210
    c.save()


def state_ablation():
    data = pd.read_csv(ROOT / "rl_depth_audit_v2" / "policy_metrics.csv")
    variants = [
        "stationary_fqi", "fqi_no_budget", "fqi_no_history",
        "fqi_no_exogenous", "fqi_no_calendar",
    ]
    labels = ["Full FQI", "No budget", "No load history", "No weather", "No calendar"]
    c = base(
        OUT / "state_ablation.pdf", "State Ablation at 12 Hours",
        "Bars are mean fold gains versus static at budget 108; dots retain individual chronological folds.",
    )
    for panel, dataset in enumerate(("steel", "tetouan")):
        x0, y0, x1, y1 = 90 + panel * 510, 115, 520 + panel * 510, 520
        block = data[data.dataset == dataset]
        values = [block.loc[block.variant == v, "gain_vs_static_pct"] for v in variants]
        all_values = pd.concat(values)
        ymin = min(-10, float(all_values.min()) - 3); ymax = max(30, float(all_values.max()) + 3)
        c.setStrokeColorRGB(0.1, 0.1, 0.1); c.line(x0, y0, x1, y0); c.line(x0, y0, x0, y1)
        for tick in np.linspace(np.ceil(ymin / 10) * 10, np.floor(ymax / 10) * 10,
                                int((np.floor(ymax / 10) - np.ceil(ymin / 10)) + 1)):
            yy = y0 + (tick - ymin) / (ymax - ymin) * (y1 - y0)
            c.setStrokeColorRGB(0.88, 0.88, 0.88); c.line(x0, yy, x1, yy)
            c.setFillColorRGB(0.2, 0.2, 0.2); c.setFont("Helvetica", 14)
            c.drawRightString(x0 - 5, yy - 2, f"{tick:g}%")
        slot = (x1 - x0) / len(variants)
        for j, series in enumerate(values):
            mean = series.mean(); xx = x0 + (j + 0.5) * slot
            zero = y0 + (0 - ymin) / (ymax - ymin) * (y1 - y0)
            yy = y0 + (mean - ymin) / (ymax - ymin) * (y1 - y0)
            c.setFillColorRGB(0.18, 0.43, 0.66); c.rect(xx - 18, min(zero, yy), 36, abs(yy - zero), fill=1, stroke=0)
            c.setFillColorRGB(0.05, 0.05, 0.05)
            for k, value in enumerate(series):
                py = y0 + (value - ymin) / (ymax - ymin) * (y1 - y0)
                c.circle(xx + (k - 2) * 4, py, 2, fill=1, stroke=0)
            c.setFont("Helvetica", 14); c.saveState(); c.translate(xx, y0 - 8); c.rotate(35)
            c.drawRightString(0, 0, labels[j]); c.restoreState()
        c.setFont("Helvetica-Bold", 18); c.drawCentredString((x0 + x1) / 2, y1 + 18, DATASET_LABELS[dataset])
    c.save()


def oracle_gap(core: pd.DataFrame):
    c = base(
        OUT / "oracle_gap.pdf", "Residual Gap to the Budget-Feasible Oracle",
        "Cells show oracle gap as a percentage of static MAE at budget 108; smaller is better and zero is the per-origin lower bound.",
    )
    row_keys = [(d, h) for d in DATASETS for h in (6, 12, 24, 48)]
    matrix = []
    for dataset, horizon in row_keys:
        block = core[(core.dataset == dataset) & (core.horizon == horizon) & (core.budget == 108)]
        matrix.append([block.loc[block.algorithm == a, "oracle_gap_pct"].mean() for a in ALGORITHMS])
    heatmap(c, matrix, [f"{DATASET_LABELS[d]} - {h} h" for d, h in row_keys],
            ALGORITHM_LABELS, 235, 60, 680, 470, limit=100, value_format="{:.1f}%")
    c.save()


def gate_coverage_risk():
    data = pd.concat([
        pd.read_csv(ROOT / f"rl_gate_ablation_v1_{dataset}" / "gate_metrics.csv")
        for dataset in DATASETS
    ], ignore_index=True)
    upper = data.gate_upper.to_numpy(float)
    beneficial = data.dynamic_better.astype(bool).to_numpy()
    thresholds = np.unique(np.r_[np.quantile(upper, np.linspace(0, 1, 31)), 0.0])
    points = []
    for threshold in thresholds:
        active = upper < threshold
        coverage = active.mean()
        risk = ((~beneficial) & active).sum() / max(active.sum(), 1)
        missed = (beneficial & ~active).sum() / max(beneficial.sum(), 1)
        points.append((threshold, coverage, risk, missed))
    c = base(
        OUT / "gate_coverage_risk.pdf", "Deployment-Gate Coverage-Risk Audit",
        "Thresholds are swept retrospectively over 240 formal units; the pre-specified operating point is upper confidence bound < 0.",
    )
    x0, y0, x1, y1 = 110, 95, 930, 535
    c.line(x0, y0, x1, y0); c.line(x0, y0, x0, y1)
    for tick in np.linspace(0, 1, 6):
        xx = x0 + tick * (x1 - x0); yy = y0 + tick * (y1 - y0)
        c.setStrokeColorRGB(0.88, 0.88, 0.88); c.line(x0, yy, x1, yy); c.line(xx, y0, xx, y1)
        c.setFillColorRGB(0.2, 0.2, 0.2); c.setFont("Helvetica", 15)
        c.drawCentredString(xx, y0 - 16, f"{100*tick:.0f}%")
        c.drawRightString(x0 - 7, yy - 3, f"{100*tick:.0f}%")
    ordered = sorted(points, key=lambda p: p[1])
    for metric, color in ((2, (0.75, 0.25, 0.14)), (3, (0.12, 0.39, 0.65))):
        pts = [(x0 + p[1] * (x1 - x0), y0 + p[metric] * (y1 - y0)) for p in ordered]
        c.setStrokeColorRGB(*color); c.setLineWidth(2.2)
        for p0, p1 in zip(pts[:-1], pts[1:]): c.line(*p0, *p1)
    operating = min(points, key=lambda p: abs(p[0]))
    ox = x0 + operating[1] * (x1 - x0); oy = y0 + operating[2] * (y1 - y0)
    c.setFillColorRGB(0, 0, 0); c.circle(ox, oy, 5, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 15); c.drawString(ox + 8, oy + 5, "pre-specified threshold 0")
    c.setFont("Helvetica", 16); c.drawCentredString((x0 + x1) / 2, y0 - 37, "Gate coverage (activated folds / all folds)")
    c.saveState(); c.translate(42, (y0 + y1) / 2); c.rotate(90); c.drawCentredString(0, 0, "Rate among relevant folds"); c.restoreState()
    c.setStrokeColorRGB(0.75, 0.25, 0.14); c.line(650, 26, 675, 26); c.setFillColorRGB(0, 0, 0); c.drawString(681, 22, "harmful activation rate")
    c.setStrokeColorRGB(0.12, 0.39, 0.65); c.line(850, 26, 875, 26); c.setFillColorRGB(0, 0, 0); c.drawString(881, 22, "missed-gain rate")
    c.save()


def sequential_environment():
    data = pd.read_csv(ROOT / "rl_sequential_audit_v1" / "policy_metrics.csv")
    penalties = pd.read_csv(
        ROOT / "expanded_rl_summary" / "sequential_penalty_summary.csv"
    )
    c = base(
        OUT / "sequential_environment.pdf", "When Actions Change Future Runtime State",
        "Left: ordinary budget environment. Right: previous-action cache state plus normalized switching cost 0.05; lower objective is better.",
    )
    algorithms = ["contextual", "stationary_fqi", "finite_horizon_fqi"]
    x0, y0, x1, y1 = 105, 150, 500, 520
    block = data[data.algorithm.isin(algorithms)]
    ymin = float(block.normalized_objective.min()) * 0.95
    ymax = float(block.normalized_objective.max()) * 1.05
    c.line(x0, y0, x1, y0); c.line(x0, y0, x0, y1)
    slot = (x1 - x0) / len(algorithms)
    for j, algorithm in enumerate(algorithms):
        series = block.loc[block.algorithm == algorithm, "normalized_objective"]
        mean = series.mean(); xx = x0 + (j + 0.5) * slot
        yy = y0 + (mean - ymin) / (ymax - ymin) * (y1 - y0)
        c.setFillColorRGB(0.18, 0.44, 0.67); c.rect(xx - 25, y0, 50, yy - y0, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 15)
        for k, value in enumerate(series):
            py = y0 + (value - ymin) / (ymax - ymin) * (y1 - y0)
            c.circle(xx + (k - (len(series)-1)/2) * 4, py, 2, fill=1, stroke=0)
        label = algorithm.replace("stationary_", "").replace("finite_horizon_", "finite-")
        c.drawCentredString(xx, y0 - 17, label)
        c.setFont("Helvetica-Bold", 15); c.drawCentredString(xx, yy + 7, f"{mean:.4f}")
    c.setFont("Helvetica-Bold", 18); c.drawCentredString((x0 + x1) / 2, y1 + 18, "Ordinary budget state")
    c.setFont("Helvetica", 15); c.drawRightString(x0 - 5, y0 - 2, f"{ymin:.3f}"); c.drawRightString(x0 - 5, y1 - 2, f"{ymax:.3f}")

    x0, y0, x1, y1 = 625, 150, 1020, 520
    ymin = float(penalties.mean_objective.min()) * 0.97
    ymax = float(penalties.mean_objective.max()) * 1.03
    c.line(x0, y0, x1, y0); c.line(x0, y0, x0, y1)
    penalty_values = [0.01, 0.05, 0.10]
    line_specs = [
        ("switch_contextual", (0.10, 0.34, 0.58), "Contextual"),
        ("switch_fqi", (0.85, 0.38, 0.12), "FQI"),
    ]
    for algorithm, color, label in line_specs:
        subset = penalties[penalties.algorithm == algorithm].set_index("penalty_setting")
        pts = []
        for j, penalty in enumerate(penalty_values):
            value = float(subset.loc[penalty, "mean_objective"])
            xx = x0 + j / 2 * (x1 - x0)
            yy = y0 + (value - ymin) / (ymax - ymin) * (y1 - y0)
            pts.append((xx, yy)); c.setFillColorRGB(*color); c.circle(xx, yy, 4, fill=1, stroke=0)
            c.setFont("Helvetica", 15); c.drawCentredString(xx, yy + 9, f"{value:.4f}")
        c.setStrokeColorRGB(*color); c.setLineWidth(2.2)
        for p0, p1 in zip(pts[:-1], pts[1:]): c.line(*p0, *p1)
    for j, penalty in enumerate(penalty_values):
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 16)
        c.drawCentredString(x0 + j / 2 * (x1 - x0), y0 - 17, f"{penalty:.2f}")
    c.setFont("Helvetica-Bold", 18); c.drawCentredString((x0 + x1) / 2, y1 + 18, "Previous-action state and switching cost")
    c.setFont("Helvetica", 16); c.drawCentredString((x0 + x1) / 2, y0 - 36, "Normalized switching penalty")
    c.setStrokeColorRGB(0.10, 0.34, 0.58); c.line(735, 62, 760, 62); c.setFillColorRGB(0, 0, 0); c.drawString(766, 59, "Contextual")
    c.setStrokeColorRGB(0.85, 0.38, 0.12); c.line(840, 62, 865, 62); c.setFillColorRGB(0, 0, 0); c.drawString(871, 59, "FQI")
    c.save()


def main():
    core = load_core()
    algorithm_comparison(core)
    hyperparameter_grid_ascii()
    action_distribution(core)
    budget_trajectories()
    state_ablation()
    oracle_gap(core)
    gate_coverage_risk()
    sequential_environment()
    print("Generated expanded vector figures in", OUT)


if __name__ == "__main__":
    main()
