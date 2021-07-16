import requests
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import warnings

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, BadZipFile
from sklearn.datasets import fetch_openml


def wiki_prices(asset_file):
    df = pd.read_csv(
        "../data/wiki_prices.csv",
        parse_dates=["date"],
        index_col=["date", "ticker"],
        infer_datetime_format=True,
    ).sort_index()
    df = df[14500000:]

    print(df.info(show_counts=True))
    with pd.HDFStore(asset_file) as store:
        store.put("quandl/wiki/prices", df)

    df = pd.read_csv("../data/wiki_stocks.csv")
    print(df.info(show_counts=True))
    with pd.HDFStore(asset_file) as store:
        store.put("quandl/wiki/stocks", df)


def sp500_prices(asset_file):
    sp500_stooq = (
        pd.read_csv("../data/^spx_d.csv", index_col=0, parse_dates=True)
        .loc["1950":"2019"]
        .rename(columns=str.lower)
    )
    print(sp500_stooq.info())
    with pd.HDFStore(asset_file) as store:
        store.put("sp500/stooq", sp500_stooq)


def sp500_stocks(asset_file):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, header=None)[0]
    print(df.head())
    df.columns = [
        "ticker",
        "name",
        "sec_filings",
        "gics_sector",
        "gics_sub_industry",
        "location",
        "first_added",
        "cik",
        "founded",
    ]
    df = df.drop("sec_filings", axis=1).set_index("ticker")
    print(df.info())
    with pd.HDFStore(asset_file) as store:
        store.put("sp500/stocks", df)


def us_metadata(asset_file):
    # https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange={}&render=download
    exchanges = ["nasdaq", "amex", "nyse"]
    df = pd.concat([pd.read_csv(f"../data/{ex}.csv") for ex in exchanges]).dropna(axis=1, how="all")
    # df = df.rename(columns=str.lower).set_index("symbol").drop("summary quote", axis=1)
    df = df[~df.index.duplicated()]
    print(df.info())
    print(df.head())

    # mcap = df[["Market Cap"]].dropna()
    # mcap["suffix"] = mcap["Market Cap"].str[-1]
    # mcap.suffix.value_counts()
    # mcap = mcap[mcap.suffix.str.endswith(("B", "M"))]
    # mcap.marketcap = pd.to_numeric(mcap.marketcap.str[1:-1])
    # mcaps = {"M": 1e6, "B": 1e9}
    # for symbol, factor in mcaps.items():
    #     mcap.loc[mcap.suffix == symbol, "marketcap"] *= factor
    # print(mcap.info())
    #
    # df["marketcap"] = mcap.marketcap
    # df.marketcap.describe(percentiles=np.arange(0.1, 1, 0.1).round(1)).apply(
    #     lambda x: f"{int(x):,d}"
    # )

    df = pd.read_csv("../data/us_equities_meta_data.csv")
    print(df.info())

    with pd.HDFStore(asset_file) as store:
        store.put("us_equities/stocks", df.set_index("ticker"))


def mnist_data():
    mnist = fetch_openml("mnist_784", version=1)
    print(mnist.DESCR)
    print(mnist.keys())

    mnist_path = Path("../data/mnist")
    if not mnist_path.exists():
        mnist_path.mkdir()

    np.save(mnist_path / "data", mnist.data.astype(np.uint8))
    np.save(mnist_path / "labels", mnist.target.astype(np.uint8))

    fashion_mnist = fetch_openml(name="Fashion-MNIST")
    print(fashion_mnist.DESCR)

    label_dict = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    fashion_path = Path("../data/fashion_mnist")
    if not fashion_path.exists():
        fashion_path.mkdir()

    pd.Series(label_dict).to_csv(fashion_path / "label_dict.csv", index=False, header=None)

    np.save(fashion_path / "data", fashion_mnist.data.astype(np.uint8))
    np.save(fashion_path / "labels", fashion_mnist.target.astype(np.uint8))


def bond_prices(asset_file):
    securities = {
        "BAMLCC0A0CMTRIV": "US Corp Master TRI",
        "BAMLHYH0A0HYM2TRIV": "US High Yield TRI",
        "BAMLEMCBPITRIV": "Emerging Markets Corporate Plus TRI",
        "GOLDAMGBD228NLBM": "Gold (London, USD)",
        "DGS10": "10-Year Treasury CMR",
    }

    df = web.DataReader(name=list(securities.keys()), data_source="fred", start=2000)
    df = df.rename(columns=securities).dropna(how="all").resample("B").mean()
    print(df.info())
    print(df.describe())
    with pd.HDFStore(asset_file) as store:
        store.put("fred/assets", df)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    asset_file = Path("../data/assets1.h5")
    wiki_prices(asset_file)
    sp500_prices(asset_file)
    sp500_stocks(asset_file)
    us_metadata(asset_file)
    # mnist_data()
    # bond_prices(asset_file)
