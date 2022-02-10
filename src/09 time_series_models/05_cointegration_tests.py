import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV

import numpy as np
from numpy.linalg import LinAlgError

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from pathlib import Path
import warnings

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

DATA_PATH = Path("..", "data")
STORE = DATA_PATH / "assets.h5"

### Remove highly correlated assets
def remove_correlated_assets(df, cutoff=0.99):
    corr = df.corr().stack()
    corr = corr[corr < 1]
    to_check = corr[corr.abs() > cutoff].index
    keep, drop = set(), set()
    for s1, s2 in to_check:
        if s1 not in keep:
            if s2 not in keep:
                keep.add(s1)
                drop.add(s2)
            else:
                drop.add(s1)
        else:
            keep.discard(s2)
            drop.add(s2)
    return df.drop(drop, axis=1)


### Remove stationary series
def check_stationarity(df):
    results = []
    for ticker, prices in df.items():
        results.append([ticker, adfuller(prices, regression="ct")[1]])
    return pd.DataFrame(results, columns=["ticker", "adf"]).sort_values("adf")


def remove_stationary_assets(df, pval=0.05):
    test_result = check_stationarity(df)
    stationary = test_result.loc[test_result.adf <= pval, "ticker"].tolist()
    return df.drop(stationary, axis=1).sort_index()


### Select Assets
def select_assets(asset_class="stocks", n=500, start=2010, end=2019):
    idx = pd.IndexSlice
    with pd.HDFStore(STORE) as store:
        df = (
            pd.concat(
                [
                    store[f"stooq/us/nasdaq/{asset_class}/prices"],
                    store[f"stooq/us/nyse/{asset_class}/prices"],
                ]
            )
            .sort_index()
            .loc[idx[:, str(start) : str(end)], :]
        )
        df = df.reset_index().drop_duplicates().set_index(["ticker", "date"])
        df["dv"] = df.close.mul(df.volume)
        dv = df.groupby(level="ticker").dv.mean().nlargest(n=n).index
        df = df.loc[idx[dv, :], "close"].unstack("ticker").ffill(limit=5).dropna(axis=1)

    df = remove_correlated_assets(df)
    return remove_stationary_assets(df).sort_index()


### Get ticker dictionary
def get_ticker_dict():
    with pd.HDFStore(STORE) as store:
        return (
            pd.concat(
                [
                    store["stooq/us/nyse/stocks/tickers"],
                    store["stooq/us/nyse/etfs/tickers"],
                    store["stooq/us/nasdaq/etfs/tickers"],
                    store["stooq/us/nasdaq/stocks/tickers"],
                ]
            )
            .drop_duplicates()
            .set_index("ticker")
            .squeeze()
            .to_dict()
        )


def compute_pair_metrics(security, candidates):
    security = security.div(security.iloc[0])
    ticker = security.name
    candidates = candidates.div(candidates.iloc[0])
    spreads = candidates.sub(security, axis=0)
    n, m = spreads.shape
    X = np.ones(shape=(n, 2))
    X[:, 1] = np.arange(1, n + 1)
    drift = (np.linalg.inv(X.T @ X) @ X.T @ spreads).iloc[1].to_frame("drift")
    vol = spreads.std().to_frame("vol")
    corr_ret = candidates.pct_change().corrwith(security.pct_change()).to_frame("corr_ret")
    corr = candidates.corrwith(security).to_frame("corr")
    metrics = drift.join(vol).join(corr).join(corr_ret).assign(n=n)
    tests = []
    for candidate, prices in candidates.items():
        df = pd.DataFrame({"s1": security, "s2": prices})
        var = VAR(df.values)
        lags = var.select_order()  # select VAR order
        k_ar_diff = lags.selected_orders["aic"]
        # Johansen Test with constant Term and estd. lag order
        cj0 = coint_johansen(df, det_order=0, k_ar_diff=k_ar_diff)
        # Engle-Granger Tests
        t1, p1 = coint(security, prices, trend="c")[:2]
        t2, p2 = coint(prices, security, trend="c")[:2]
        tests.append([ticker, candidate, t1, p1, t2, p2, k_ar_diff, *cj0.lr1])
    columns = ["s1", "s2", "t1", "p1", "t2", "p2", "k_ar_diff", "trace0", "trace1"]
    tests = pd.DataFrame(tests, columns=columns).set_index("s2")
    return metrics.join(tests)


if __name__ == "__main__":
    ### Johansen Test Critical Values
    critical_values = {
        0: {0.9: 13.4294, 0.95: 15.4943, 0.99: 19.9349},
        1: {0.9: 2.7055, 0.95: 3.8415, 0.99: 6.6349},
    }

    trace0_cv = critical_values[0][0.95]  # critical value for 0 cointegration relationships
    trace1_cv = critical_values[1][0.95]  # critical value for 1 cointegration relationship

    ## Load & Clean Stock & ETF Data
    for asset_class, n in [("etfs", 500), ("stocks", 250)]:
        df = select_assets(asset_class=asset_class, n=n)
        df.to_hdf("../data/data09.h5", f"{asset_class}/close")

    names = get_ticker_dict()

    stocks = pd.read_hdf("../data/data09.h5", "stocks/close")
    stocks.info()
    etfs = pd.read_hdf("../data/data09.h5", "etfs/close")
    etfs.info()

    tickers = {k: v for k, v in names.items() if k in etfs.columns.union(stocks.columns)}
    pd.Series(tickers).to_hdf("../data/data09.h5", "tickers")

    corr = pd.DataFrame(index=stocks.columns)
    for etf, data in etfs.items():
        corr[etf] = stocks.corrwith(data)
    corr.info()

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.clustermap(corr, cmap=cmap, center=0)
    plt.savefig("images/05_01.png")

    ## Candidate Selection using Heuristics
    ### Computational Complexity: Comparing running times
    print(stocks.shape, etfs.shape)
    stocks.info()
    etfs.info()

    security = etfs["AAXJ.US"].loc["2010":"2013"]
    candidates = stocks.loc["2010":"2013"]

    security = security.div(security.iloc[0])
    candidates = candidates.div(candidates.iloc[0])
    spreads = candidates.sub(security, axis=0)

    n, m = spreads.shape
    X = np.ones(shape=(n, 2))
    X[:, 1] = np.arange(1, n + 1)

    # #### Heuristics
    # get_ipython().run_cell_magic("timeit", "", "np.linalg.inv(X.T @ X) @ X.T @ spreads")
    # get_ipython().run_cell_magic("timeit", "", "spreads.std()")
    # get_ipython().run_cell_magic("timeit", "", "candidates.corrwith(security)")
    #
    # #### Cointegration Tests
    # get_ipython().run_cell_magic(
    #     "timeit",
    #     "",
    #     "for candidate, prices in candidates.items():\n    df = pd.DataFrame({'s1': security,\n
    #     's2': prices})\n    var = VAR(df.values)\n    lags = var.select_order()\n
    #     k_ar_diff = lags.selected_orders['aic']\n    coint_johansen(df, det_order=0, k_ar_diff=k_ar_diff)\n
    #     coint(security, prices, trend='c')[:2]\n    coint(prices, security, trend='c')[:2]",
    # )

    ### Compute Heuristics
    spreads = []
    start = 2010
    stop = 2019
    etf_candidates = etfs.loc[str(start) : str(stop), :]
    stock_candidates = stocks.loc[str(start) : str(stop), :]
    s = time()
    for i, (etf_ticker, etf_prices) in enumerate(etf_candidates.items(), 1):
        df = compute_pair_metrics(etf_prices, stock_candidates)
        spreads.append(df.set_index("s1", append=True))
        if i % 10 == 0:
            print(f"{i:>3} {time() - s:.1f}")
            s = time()

    names = get_ticker_dict()
    spreads = pd.concat(spreads)
    spreads.index.names = ["s2", "s1"]
    spreads = spreads.swaplevel()
    spreads["name1"] = spreads.index.get_level_values("s1").map(names)
    spreads["name2"] = spreads.index.get_level_values("s2").map(names)

    spreads["t"] = spreads[["t1", "t2"]].min(axis=1)
    spreads["p"] = spreads[["p1", "p2"]].min(axis=1)

    ### Engle-Granger vs Johansen: how do their findings compare?
    spreads["trace_sig"] = ((spreads.trace0 > trace0_cv) & (spreads.trace1 > trace1_cv)).astype(int)
    spreads["eg_sig"] = (spreads.p < 0.05).astype(int)
    print(pd.crosstab(spreads.eg_sig, spreads.trace_sig))

    spreads["coint"] = (spreads.trace_sig & spreads.eg_sig).astype(int)
    spreads.info()
    spreads = spreads.reset_index()

    sns.scatterplot(
        x=np.log1p(spreads.t.abs()),
        y=np.log1p(spreads.trace1),
        hue="coint",
        data=spreads[spreads.trace0 > trace0_cv],
    )
    plt.savefig("images/05_02.png")

    spreads.to_hdf("../data/heuristics.h5", "spreads")
    spreads = pd.read_hdf("../data/heuristics.h5", "spreads")

    ### Evaluate Heuristics
    spreads.drift = spreads.drift.abs()
    print(pd.crosstab(spreads.eg_sig, spreads.trace_sig))

    pd.set_option("display.float_format", lambda x: f"{x:.2%}")
    print(pd.crosstab(spreads.eg_sig, spreads.trace_sig, normalize=True))

    fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
    for i, heuristic in enumerate(["drift", "vol", "corr", "corr_ret"]):
        sns.boxplot(x="coint", y=heuristic, data=spreads, ax=axes[i])
    fig.tight_layout()
    plt.savefig("images/05_03.png")

    ### How well do the heuristics predict significant cointegration?
    print(
        spreads.groupby(spreads.coint)["drift", "vol", "corr"]
        .describe()
        .stack(level=0)
        .swaplevel()
        .sort_index()
    )
    print(spreads.coint.value_counts())

    #### Logistic Regression
    y = spreads.coint
    X = spreads[["drift", "vol", "corr", "corr_ret"]]
    # X = spreads[['drift', 'vol']]

    kf = StratifiedKFold(n_splits=5, shuffle=True)

    log_reg = LogisticRegressionCV(
        Cs=np.logspace(-10, 10, 21), class_weight="balanced", scoring="roc_auc"
    )

    log_reg.fit(X=X, y=y)
    Cs = log_reg.Cs_
    scores = pd.DataFrame(log_reg.scores_[True], columns=Cs).mean()
    scores.plot(logx=True)
    print(f"C:{np.log10(scores.idxmax()):.2f}, AUC: {scores.max():.2%}")
    print(log_reg.coef_)

    y_pred = log_reg.predict_proba(X)[:, 1]
    print(confusion_matrix(y_true=spreads.coint, y_pred=(y_pred > 0.5)))
    print(
        spreads.assign(y_pred=log_reg.predict_proba(X)[:, 1])
        .groupby(spreads.coint)
        .y_pred.describe()
    )

    #### Decision Tree Classifier
    model = DecisionTreeClassifier(class_weight="balanced")
    decision_tree = GridSearchCV(
        model, param_grid={"max_depth": list(range(1, 10))}, cv=5, scoring="roc_auc"
    )
    decision_tree.fit(X=X, y=y)
    print(f'{decision_tree.best_score_:.2%}, Depth: {decision_tree.best_params_["max_depth"]}')

    pd.Series(
        data=decision_tree.best_estimator_.feature_importances_, index=X.columns
    ).sort_values().plot.barh(title="Feature Importance")
    sns.despine()
    plt.savefig("images/05_04.png")

    data = spreads.assign(y_pred=decision_tree.predict_proba(X)[:, 1])
    print(data.groupby(spreads.coint).y_pred.describe())
    sns.catplot(x="coint", y="y_pred", data=data, kind="box")
    plt.savefig("images/05_05.png")
