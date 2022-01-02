# Download and store STOOQ data
# This notebook contains information on downloading the STOOQ stock and ETF price data that we use in [Chapter 09]
# (../09_time_series_models) for a pairs trading strategy based on cointegration and [Chapter 11]
# (../11_decision_trees_random_forests) for a long-short strategy using Random Forest return predictions.

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import warnings
import requests

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, BadZipFile
from sklearn.datasets import fetch_openml

pd.set_option("display.expand_frame_repr", False)
warnings.filterwarnings("ignore")

## Set Data Store path
DATA_STORE = Path("../data/assets.h5")

## Stooq Historical Market Data
# > Note that the below downloading details may change at any time as Stooq updates their website; if you encounter
# errors, please inspect their website and raise a GitHub issue to let us know, so we can update the information.
# > Update 12/2020: please note that STOOQ will disable automatic downloads and require CAPTCHA starting Dec 10,
# 2020 so that the code that downloads and unpacks the zip files will no longer work; please navigate to their
# website [here](https://stooq.com/db/h/) for manual download.

### Download price data
# 1. Download **price data** for the selected combination of asset class, market and frequency from [the Stooq website]
#    (https://stooq.com/db/h/)
# 2. Store the result under `stooq` using the preferred folder structure outlined on the website. It has the structure:
# `/data/freq/market/asset_class`, such as `/data/daily/us/nasdaq etfs`.

stooq_path = Path("../data/stooq")
if not stooq_path.exists():
    stooq_path.mkdir()

# # Use the symbol for the market you want to download price data for. In this book we'll be useing `us` and `jp`.
# def download_price_data(market="us"):
#     data_url = f"https://stooq.com/db/d/?b=d_{market}_txt"  # https://stooq.com/db/d/?b=d_us_txt
#     response = requests.get(data_url).content
#     with ZipFile(BytesIO(response)) as zip_file:
#         for i, file in enumerate(zip_file.namelist()):
#             if not file.endswith(".txt"):
#                 continue
#             local_file = stooq_path / file
#             local_file.parent.mkdir(parents=True, exist_ok=True)
#             with local_file.open("wb") as output:
#                 for line in zip_file.open(file).readlines():
#                     output.write(line)
#
#
# for market in ["us", "jp"]:
#     download_price_data(market=market)

### Add symbols
# Add the corresponding **symbols**, i.e., tickers and names by following the directory tree on the same site.
# You can also adapt the following code snippet using the appropriate asset code that you find by inspecting the url;
# this example works for NASDAQ ETFs that have code `g=69`:
metadata_dict = {
    ("jp", "tse etfs"): 34,
    ("jp", "tse stocks"): 32,
    ("us", "nasdaq etfs"): 69,
    ("us", "nasdaq stocks"): 27,
    ("us", "nyse etfs"): 70,
    ("us", "nyse stocks"): 28,
    ("us", "nysemkt stocks"): 26,
}

for (market, asset_class), code in metadata_dict.items():
    # df = pd.read_csv(f"https://stooq.com/db/l/?g={code}", sep="        ").apply(
    #     lambda x: x.str.strip()
    # )
    df = pd.read_csv(f"{stooq_path}/{market}_{asset_class}.csv", sep="        ").apply(
        lambda x: x.str.strip()
    )
    df.columns = ["ticker", "name"]
    df = df.drop_duplicates("ticker").dropna()
    print(market, asset_class, f"# tickers: {df.shape[0]:,.0f}")
    path = stooq_path / "tickers" / market
    if not path.exists():
        path.mkdir(parents=True)
    df.to_csv(path / f"{asset_class}.csv", index=False)

### Store price data in HDF5 format
# To speed up loading, we store the price data in HDF format. The function `get_stooq_prices_and_symbols` loads data
# assuming the directory structure described above and takes the following arguments:
# - frequency (see Stooq website for options as these may change; default is `daily`
# - market (default: `us`), and
# - asset class (default: `nasdaq etfs`.
# It removes files that do not have data or do not appear in the corresponding list of symbols.


def get_stooq_prices_and_tickers(frequency="daily", market="us", asset_class="nasdaq etfs"):
    prices = []
    tickers = pd.read_csv(stooq_path / "tickers" / market / f"{asset_class}.csv")
    if frequency in ["5 min", "hourly"]:
        parse_dates = [["date", "time"]]
        date_label = "date_time"
    else:
        parse_dates = ["date"]
        date_label = "date"
    names = ["ticker", "freq", "date", "time", "open", "high", "low", "close", "volume", "openint"]

    usecols = ["ticker", "open", "high", "low", "close", "volume"] + parse_dates
    path = stooq_path / "data" / frequency / market / asset_class
    print(path.as_posix())
    files = path.glob("**/*.txt")
    for i, file in enumerate(files, 1):
        if i % 500 == 0:
            print(i)
        if file.stem not in set(tickers.ticker.str.lower()):
            print(file.stem, "not available")
            file.unlink()
        else:
            try:
                df = pd.read_csv(
                    file, names=names, usecols=usecols, header=0, parse_dates=parse_dates
                )
                prices.append(df)
            except pd.errors.EmptyDataError:
                print("\tdata missing", file.stem)
                file.unlink()

    prices = (
        pd.concat(prices, ignore_index=True)
        .rename(columns=str.lower)
        .set_index(["ticker", date_label])
        .apply(lambda x: pd.to_numeric(x, errors="coerce"))
    )
    return prices, tickers


# We'll be using US equities and ETFs in [Chapter 9](../09_time_series_models) and Japanese equities in
# [Chapter 11](../11_decision_trees_random_forests). The following code collects the price data for the period 2000-2019
# and stores it with the corresponding symbols in the global `assets.h5` store:
# load some Japanese and all US assets for 2000-2019
markets = {
    "jp": ["tse stocks"],
    "us": ["nasdaq etfs", "nasdaq stocks", "nyse etfs", "nyse stocks", "nysemkt stocks"],
}
frequency = "daily"

idx = pd.IndexSlice
for market, asset_classes in markets.items():
    for asset_class in asset_classes:
        print(f"\n{asset_class}")
        prices, tickers = get_stooq_prices_and_tickers(
            frequency=frequency, market=market, asset_class=asset_class
        )
        prices = prices.sort_index().loc[idx[:, "2000":"2019"], :]
        names = prices.index.names
        prices = prices.reset_index().drop_duplicates().set_index(names).sort_index()
        print("\nNo. of observations per asset")
        print(prices.groupby("ticker").size().describe())
        key = f'stooq/{market}/{asset_class.replace(" ", "/")}/'
        print(prices.info(null_counts=True))
        prices.to_hdf(DATA_STORE, key + "prices", format="t")
        print(tickers.info())
        tickers.to_hdf(DATA_STORE, key + "tickers", format="t")
