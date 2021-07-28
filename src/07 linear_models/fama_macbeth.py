import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as pdr

from pathlib import Path
from statsmodels.api import OLS, add_constant
from linearmodels.asset_pricing import LinearFactorModel


if __name__ == "__main__":
    # ff_factor = "F-F_Research_Data_5_Factors_2x3"
    # ff_factor_data = pdr.DataReader(ff_factor, "famafrench", start="2010", end="2017-12")[0]
    #
    # ff_portfolio = "17_Industry_Portfolios"
    # ff_portfolio_data = pdr.DataReader(ff_portfolio, "famafrench", start="2010", end="2017-12")[0]
    # ff_portfolio_data = ff_portfolio_data.sub(ff_factor_data.RF, axis=0)
    #
    # with pd.HDFStore("../data/fama.h5") as store:
    #     store.put("fama/factor", ff_factor_data)
    #     store.put("fama/portfolio", ff_portfolio_data)

    with pd.HDFStore("../data/fama.h5") as store:
        ff_factor_data = store["fama/factor"]
        ff_portfolio_data = store["fama/portfolio"]

    ff_factor_data.info()
    print(ff_factor_data.describe())
    ff_portfolio_data.info()
    print(ff_portfolio_data.describe())

    with pd.HDFStore("../data/assets.h5") as store:
        # prices = store["/quandl/wiki/prices"].adj_close
        # prices = prices.unstack().loc["2010":"2010"]
        prices = store["/quandl/wiki/prices"].adj_close.unstack().loc["2010":"2017"]
        equities = store["/us_equities/stocks"].drop_duplicates()

    sectors = equities.filter(prices.columns, axis=0).sector.to_dict()
    prices = prices.filter(sectors.keys()).dropna(how="all", axis=1)

    returns = prices.resample("1M").last().pct_change().mul(100).to_period("M")
    returns = returns.dropna(how="all").dropna(axis=1)
    returns.info()

    ff_factor_data = ff_factor_data.loc[returns.index]
    ff_portfolio_data = ff_portfolio_data.loc[returns.index]
    print(ff_factor_data.describe())

    excess_returns = returns.sub(ff_factor_data.RF, axis=0)
    excess_returns.info()
    excess_returns = excess_returns.clip(
        lower=np.percentile(excess_returns, 1), upper=np.percentile(excess_returns, 99)
    )

    ff_portfolio_data.info()
    ff_factor_data = ff_factor_data.drop("RF", axis=1)
    ff_factor_data.info()

    betas = []
    for industry in ff_portfolio_data:
        step1 = OLS(
            endog=ff_portfolio_data.loc[ff_factor_data.index, industry],
            exog=add_constant(ff_factor_data),
        ).fit()
        betas.append(step1.params.drop("const"))

    betas = pd.DataFrame(betas, columns=ff_factor_data.columns, index=ff_portfolio_data.columns)
    betas.info()

    lambdas = []
    for period in ff_portfolio_data.index:
        step2 = OLS(endog=ff_portfolio_data.loc[period, betas.index], exog=betas).fit()
        lambdas.append(step2.params)
    lambdas = pd.DataFrame(lambdas, index=ff_portfolio_data.index, columns=betas.columns.tolist())
    lambdas.info()

    lambdas.mean().sort_values().plot.barh(figsize=(12, 4))
    sns.despine()
    plt.tight_layout()
    plt.show()

    t = lambdas.mean().div(lambdas.std())

    window = 24
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    lambdas.mean().sort_values().plot.barh(ax=ax1)
    lambdas.rolling(window).mean().dropna().plot(lw=1, figsize=(14, 5), sharey=True, ax=ax2)
    plt.show()

    lambdas.rolling(window).mean().dropna().plot(lw=2, figsize=(14, 7), subplots=True, sharey=True)
    plt.show()

    mod = LinearFactorModel(portfolios=ff_portfolio_data, factors=ff_factor_data)
    res = mod.fit()
    print(res)
    print(res.full_summary)
    print(lambdas.mean())
