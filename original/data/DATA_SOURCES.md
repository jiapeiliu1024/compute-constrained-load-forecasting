# Dataset sources and expected paths

The experiment code expects the following files relative to the package root. `download_and_prepare_data.ps1` creates this layout.

## Formal datasets

1. **Steel Industry Energy Consumption** — UCI dataset 851, DOI `10.24432/C52G8C`, CC BY 4.0. Expected file: `data/raw/uci_steel/data.csv`.
2. **Power Consumption of Tetouan City** — UCI dataset 849, DOI `10.24432/C5B034`, CC BY 4.0. The 10-minute source is aggregated to `data/processed/uci_tetouan_hourly.csv` using complete six-observation clock hours.
3. **ElectricityLoadDiagrams20112014** — UCI dataset 321, DOI `10.24432/C58C86`, CC BY 4.0. The 15-minute client series is aggregated to `data/processed/uci_ecl_hourly.csv` using complete four-observation clock hours.
4. **Open Power System Data Time Series, version 2020-10-06** — DOI `10.25832/time_series/2020-10-06`. Expected source: `data/raw/opsd_time_series_60min_singleindex.csv`. The study uses a fixed aggregate of AT, DK, ES, NL, and PT with paired realized-load and published-forecast coverage during 2016-2019, producing `data/processed/opsd_aggregate_hourly.csv`.

## Boundary dataset

5. **Appliances Energy Prediction** — UCI dataset 374, DOI `10.24432/C5VC8G`, CC BY 4.0. Expected file: `data/raw/uci_appliances/data.csv`.

## Official source pages

- <https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption>
- <https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city>
- <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>
- <https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction>
- <https://data.open-power-system-data.org/time_series/2020-10-06/>

OPSD incorporates primary data from sources including ENTSO-E. Consult the OPSD package documentation and the primary-source terms before redistributing the raw or processed file. This code package therefore supplies acquisition and preprocessing scripts rather than redistributing those data.

