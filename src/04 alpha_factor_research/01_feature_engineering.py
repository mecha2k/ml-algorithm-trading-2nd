import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from pyfinance.ols import PandasRollingOLS
from datetime import datetime

idx = pd.IndexSlice
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16

if __name__ == "__main__":
    asset_file = "../data/assets.h5"
    START = 2000
    END = 2018

    with pd.HDFStore(asset_file) as store:
        prices = (
            store["quandl/wiki/prices"]
            .loc[idx[str(START) : str(END), :], "adj_close"]
            .unstack("ticker")
        )
        stocks = store["us_equities/stocks"].loc[:, ["marketcap", "ipoyear", "sector"]]
    print(prices.info())
    print(stocks.info())
    print(~stocks.index.duplicated())
    stocks = stocks[~stocks.index.duplicated()]
    stocks.index.name = "ticker"

    shared = prices.columns.intersection(stocks.index)
    stocks = stocks.loc[shared, :]
    print(stocks.info())
    prices = prices.loc[:, shared]
    print(prices.info())
    assert prices.shape[1] == stocks.shape[0]

    monthly_prices = prices.resample("M").last()
    print(monthly_prices.info())

    outlier_cutoff = 0.01
    data = pd.DataFrame()
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        data[f"return_{lag}m"] = (
            monthly_prices.pct_change(lag)
            .stack()
            .pipe(
                lambda x: x.clip(
                    lower=x.quantile(outlier_cutoff), upper=x.quantile(1 - outlier_cutoff)
                )
            )
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )
    data = data.swaplevel().dropna()
    print(data.info())

    min_obs = 120
    nobs = data.groupby(level="ticker").size()
    keep = nobs[nobs > min_obs].index

    data = data.loc[idx[keep, :], :]
    print(data.info())
    print(data.describe())
    # cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.clustermap(data.corr("spearman"), annot=True, center=0, cmap="Blues")
    plt.savefig("images/01-01.png", bboxinches="tight")

    data.index.get_level_values("ticker").nunique()
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    factor_data = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start="2000")[
        0
    ].drop("RF", axis=1)
    factor_data.index = factor_data.index.to_timestamp()
    factor_data = factor_data.resample("M").last().div(100)
    factor_data.index.name = "date"
    factor_data.info()

    factor_data = factor_data.join(data["return_1m"]).sort_index()
    factor_data.info()

    T = 24
    betas = factor_data.groupby(level="ticker", group_keys=False).apply(
        lambda x: PandasRollingOLS(
            window=min(T, x.shape[0] - 1), y=x.return_1m, x=x.drop("return_1m", axis=1)
        ).beta
    )
    print(betas.describe().join(betas.sum(1).describe().to_frame("total")))

    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.clustermap(betas.corr(), annot=True, cmap=cmap, center=0)
    plt.savefig("images/01-02.png", bboxinches="tight")

    data = data.join(betas.groupby(level="ticker").shift())
    data.info()

    data.loc[:, factors] = data.groupby("ticker")[factors].apply(lambda x: x.fillna(x.mean()))
    data.info()

    for lag in [2, 3, 6, 9, 12]:
        data[f"momentum_{lag}"] = data[f"return_{lag}m"].sub(data.return_1m)
    data[f"momentum_3_12"] = data[f"return_12m"].sub(data.return_3m)

    dates = data.index.get_level_values("date")
    data["year"] = dates.year
    data["month"] = dates.month

    for t in range(1, 7):
        data[f"return_1m_t-{t}"] = data.groupby(level="ticker").return_1m.shift(t)
    data.info()

    for t in [1, 2, 3, 6, 12]:
        data[f"target_{t}m"] = data.groupby(level="ticker")[f"return_{t}m"].shift(-t)

    cols = [
        "target_1m",
        "target_2m",
        "target_3m",
        "return_1m",
        "return_2m",
        "return_3m",
        "return_1m_t-1",
        "return_1m_t-2",
        "return_1m_t-3",
    ]

    data[cols].dropna().sort_index().head(10)
    data.info()

    data = data.join(
        pd.qcut(stocks.ipoyear, q=5, labels=list(range(1, 6)))
        .astype(float)
        .fillna(0)
        .astype(int)
        .to_frame("age")
    )
    data.age = data.age.fillna(-1)
    stocks.info()

    size_factor = (
        monthly_prices.loc[
            data.index.get_level_values("date").unique(),
            data.index.get_level_values("ticker").unique(),
        ]
        .sort_index(ascending=False)
        .pct_change()
        .fillna(0)
        .add(1)
        .cumprod()
    )
    size_factor.info()

    msize = (size_factor.mul(stocks.loc[size_factor.columns, "marketcap"])).dropna(
        axis=1, how="all"
    )

    data["msize"] = (
        msize.apply(lambda x: pd.qcut(x, q=10, labels=list(range(1, 11))).astype(int), axis=1)
        .stack()
        .swaplevel()
    )
    data.msize = data.msize.fillna(-1)
    data = data.join(stocks[["sector"]])
    data.sector = data.sector.fillna("Unknown")
    data.info()

    with pd.HDFStore(asset_file) as store:
        store.put("engineered_features", data.sort_index().loc[idx[:, : datetime(2018, 3, 1)], :])
        print(store.info())

    dummy_data = pd.get_dummies(
        data,
        columns=["year", "month", "msize", "age", "sector"],
        prefix=["year", "month", "msize", "age", ""],
        prefix_sep=["_", "_", "_", "_", ""],
    )
    dummy_data = dummy_data.rename(columns={c: c.replace(".0", "") for c in dummy_data.columns})
    dummy_data.info()
