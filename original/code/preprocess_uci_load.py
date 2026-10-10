"""Create a leakage-neutral hourly aggregate from the official UCI ECL data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/uci_electricity/LD2011_2014.txt")
    ap.add_argument("--output", type=Path, default=CODE_ROOT / "processed_data/uci_ecl_hourly.csv")
    ap.add_argument("--start", default="2012-01-01 00:00:00")
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pieces = []
    for chunk in pd.read_csv(args.input, sep=";", decimal=",", index_col=0, chunksize=4096):
        chunk.index = pd.to_datetime(chunk.index)
        chunk = chunk.loc[chunk.index >= pd.Timestamp(args.start)]
        if len(chunk):
            # Values are client mean demand in kW over each 15-minute period.
            pieces.append(pd.DataFrame({"timestamp": chunk.index,
                                        "aggregate_load_kw": chunk.sum(axis=1).to_numpy(float)}))
    quarter_hour = pd.concat(pieces, ignore_index=True).set_index("timestamp").sort_index()
    # UCI timestamps mark the end of each 15-minute delivery interval.  A
    # natural clock hour therefore contains the four interval ends in
    # ``(hour - 1 h, hour]``.  The --start boundary otherwise creates a
    # one-observation first bin, so retain complete hours explicitly instead
    # of allowing a partial mean to enter the benchmark.
    hourly_groups = quarter_hour.resample("1h", label="right", closed="right")
    hourly_values = hourly_groups.mean()
    observations_per_hour = hourly_groups.size()
    complete = observations_per_hour.eq(4)
    hourly = hourly_values.loc[complete].dropna().reset_index()
    if hourly.timestamp.duplicated().any() or not hourly.timestamp.is_monotonic_increasing:
        raise RuntimeError("invalid hourly timestamps")
    hourly.to_csv(args.output, index=False)
    profile = {
        "source": str(args.input.resolve()), "output": str(args.output.resolve()),
        "aggregation": "sum across 370 clients at 15 min, then arithmetic mean within each clock hour",
        "unit": "kW", "start": str(hourly.timestamp.min()), "end": str(hourly.timestamp.max()),
        "source_quarter_hour_rows": int(len(quarter_hour)),
        "clock_hour_bins": int(len(observations_per_hour)),
        "complete_hour_bins": int(complete.sum()),
        "dropped_incomplete_hour_bins": int((~complete).sum()),
        "observations_per_complete_hour": 4,
        "rows": int(len(hourly)), "missing": int(hourly.aggregate_load_kw.isna().sum()),
        "mean_kw": float(hourly.aggregate_load_kw.mean()), "std_kw": float(hourly.aggregate_load_kw.std()),
        "min_kw": float(hourly.aggregate_load_kw.min()), "max_kw": float(hourly.aggregate_load_kw.max()),
    }
    args.output.with_suffix(".profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))

if __name__ == "__main__": main()
