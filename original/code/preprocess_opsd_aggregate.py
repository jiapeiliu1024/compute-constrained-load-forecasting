"""Create a reproducible aggregate OPSD load series for the budget-policy study."""

from pathlib import Path
import json

import pandas as pd


SOURCE = Path("data/raw/opsd_time_series_60min_singleindex.csv")
OUTPUT = Path("data/processed/opsd_aggregate_hourly.csv")
PROFILE = Path("data/processed/opsd_aggregate_hourly.profile.json")
# Fixed non-overlapping national systems with 100% paired actual/forecast
# coverage over 2016--2019 in the archived OPSD release.  Restricting the
# aggregate to complete systems preserves a continuous hourly axis without
# imputing either targets or the published operator forecast.
REGIONS = ("AT", "DK", "ES", "NL", "PT")


def main() -> None:
    actual = [f"{region}_load_actual_entsoe_transparency" for region in REGIONS]
    forecast = [f"{region}_load_forecast_entsoe_transparency" for region in REGIONS]
    frame = pd.read_csv(
        SOURCE,
        usecols=["utc_timestamp", *actual, *forecast],
        parse_dates=["utc_timestamp"],
    )
    frame = frame[(frame.utc_timestamp >= "2016-01-01") & (frame.utc_timestamp < "2020-01-01")]
    complete = frame[[*actual, *forecast]].notna().all(axis=1)
    frame = frame.loc[complete].copy()
    frame["aggregate_load"] = frame[actual].sum(axis=1)
    frame["official_forecast"] = frame[forecast].sum(axis=1)
    result = frame[["utc_timestamp", "aggregate_load", "official_forecast"]].rename(
        columns={"utc_timestamp": "timestamp"}
    )
    if result.timestamp.duplicated().any():
        raise ValueError("Duplicate OPSD timestamps")
    result.to_csv(OUTPUT, index=False)
    PROFILE.write_text(json.dumps({
        "source": str(SOURCE),
        "regions": list(REGIONS),
        "selection_rule": "non-overlapping national systems with 100% paired actual/forecast coverage over 2016--2019",
        "aggregation": "hourly sum; no target or official-forecast imputation",
        "rows": len(result),
        "start": str(result.timestamp.min()),
        "end": str(result.timestamp.max()),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
