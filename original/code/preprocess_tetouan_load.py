"""Create an hourly, city-level Tetouan load and weather table from UCI data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
ZONE_COLUMNS = [
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
]
WEATHER_RENAME = {
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flows",
    "diffuse flows": "diffuse_flows",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/uci_tetouan_power/Tetuan City power consumption.csv",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "data/raw/uci_tetouan_power.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CODE_ROOT / "processed_data/uci_tetouan_hourly.csv",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    raw["timestamp"] = pd.to_datetime(raw.pop("DateTime"), format="%m/%d/%Y %H:%M")
    raw["aggregate_load"] = raw[ZONE_COLUMNS].sum(axis=1)
    raw = raw.rename(columns=WEATHER_RENAME).set_index("timestamp").sort_index()
    keep = ["aggregate_load", *WEATHER_RENAME.values()]
    # These observations are stamped at 00, 10, ..., 50 minutes.  Natural
    # clock hours are therefore left-closed intervals [hour, hour + 1 h).
    # Requiring all six observations prevents partial boundary bins (or an
    # interior source gap) from being silently averaged.
    hourly_groups = raw[keep].resample("1h", label="left", closed="left")
    hourly_values = hourly_groups.mean()
    observations_per_hour = hourly_groups.size()
    complete = observations_per_hour.eq(6)
    hourly = hourly_values.loc[complete].dropna().reset_index()

    if hourly.timestamp.duplicated().any() or not hourly.timestamp.is_monotonic_increasing:
        raise RuntimeError("invalid hourly timestamps")
    if hourly[keep].isna().any().any():
        raise RuntimeError("missing value after hourly aggregation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.output, index=False)
    profile = {
        "dataset": "UCI Power Consumption of Tetouan City",
        "doi": "10.24432/C5B034",
        "license": "CC BY 4.0",
        "archive_sha256": sha256(args.archive),
        "source_rows": int(len(raw)),
        "clock_hour_bins": int(len(observations_per_hour)),
        "complete_hour_bins": int(complete.sum()),
        "dropped_incomplete_hour_bins": int((~complete).sum()),
        "observations_per_complete_hour": 6,
        "hourly_rows": int(len(hourly)),
        "start": str(hourly.timestamp.min()),
        "end": str(hourly.timestamp.max()),
        "missing": int(hourly[keep].isna().sum().sum()),
        "aggregation": "sum three zones at 10 min, then arithmetic mean within each clock hour",
        "load_unit": "reported dataset power-consumption unit; no conversion applied",
    }
    args.output.with_suffix(".profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
