import warnings
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.random import random, uniform, dirichlet, choice
from numpy.linalg import inv
from scipy.optimize import minimize
from matplotlib.ticker import FuncFormatter


np.random.seed(42)
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
cmap = sns.diverging_palette(10, 240, n=9, as_cmap=True)


def simulate_portfolios(mean_ret, cov, rf_rate, short=True):
    alpha = np.full(shape=n_assets, fill_value=0.05)
    weights = dirichlet(alpha=alpha, size=NUM_PF)
    if short:
        weights *= choice([-1, 1], size=weights.shape)

    returns = weights @ mean_ret.values + 1
    returns = returns ** periods_per_year - 1
    std = (weights @ weekly_returns.T).std(1)
    std *= np.sqrt(periods_per_year)
    sharpe = (returns - rf_rate) / std
    return (
        pd.DataFrame(
            {
                "Annualized Standard Deviation": std,
                "Annualized Returns": returns,
                "Sharpe Ratio": sharpe,
            }
        ),
        weights,
    )


def portfolio_std(wt, rt=None, cov=None):
    return np.sqrt(wt @ cov @ wt * periods_per_year)


def portfolio_returns(wt, rt=None, cov=None):
    return (wt @ rt + 1) ** periods_per_year - 1


def portfolio_performance(wt, rt, cov):
    r = portfolio_returns(wt, rt=rt)
    sd = portfolio_std(wt, cov=cov)
    return r, sd


def neg_sharpe_ratio(weights, mean_ret, cov):
    r, sd = portfolio_performance(weights, mean_ret, cov)
    return -(r - rf_rate) / sd


def max_sharpe_ratio(mean_ret, cov, short=False):
    return minimize(
        fun=neg_sharpe_ratio,
        x0=x0,
        args=(mean_ret, cov),
        method="SLSQP",
        bounds=((-1 if short else 0, 1),) * n_assets,
        constraints=weight_constraint,
        options={"tol": 1e-10, "maxiter": 1e4},
    )


def min_vol_target(mean_ret, cov, target, short=False):
    def ret_(wt):
        return portfolio_returns(wt, mean_ret)

    constraints = [{"type": "eq", "fun": lambda x: ret_(x) - target}, weight_constraint]

    bounds = ((-1 if short else 0, 1),) * n_assets
    return minimize(
        portfolio_std,
        x0=x0,
        args=(mean_ret, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"tol": 1e-10, "maxiter": 1e4},
    )


def min_vol(mean_ret, cov, short=False):
    bounds = ((-1 if short else 0, 1),) * n_assets

    return minimize(
        fun=portfolio_std,
        x0=x0,
        args=(mean_ret, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=weight_constraint,
        options={"tol": 1e-10, "maxiter": 1e4},
    )


def efficient_frontier(mean_ret, cov, ret_range, short=False):
    return [min_vol_target(mean_ret, cov, ret) for ret in ret_range]


if __name__ == "__main__":
    with pd.HDFStore("../data/assets.h5") as store:
        sp500_stocks = store["sp500/stocks"]
    print(sp500_stocks.head())
    with pd.HDFStore("../data/assets.h5") as store:
        prices = (
            store["quandl/wiki/prices"]
            .adj_close.unstack("ticker")
            .filter(sp500_stocks.index)
            .sample(n=30, axis=1)
        )

    start = 2008
    end = 2017

    # Create month-end monthly returns and drop dates that have no observations:
    weekly_returns = (
        prices.loc[f"{start}":f"{end}"].resample("W").last().pct_change().dropna(how="all")
    )
    weekly_returns = weekly_returns.dropna(axis=1)
    print(weekly_returns.info())

    stocks = weekly_returns.columns
    n_obs, n_assets = weekly_returns.shape
    print(n_assets, n_obs)

    NUM_PF = 100000  # no of portfolios to simulate
    x0 = uniform(0, 1, n_assets)
    x0 /= np.sum(np.abs(x0))
    periods_per_year = round(weekly_returns.resample("A").size().mean())
    print(periods_per_year)

    mean_returns = weekly_returns.mean()
    cov_matrix = weekly_returns.cov()
    precision_matrix = pd.DataFrame(inv(cov_matrix), index=stocks, columns=stocks)
    treasury_10yr_monthly = (
        web.DataReader("DGS10", "fred", start, end)
        .resample("M")
        .last()
        .div(periods_per_year)
        .div(100)
        .squeeze()
    )
    rf_rate = treasury_10yr_monthly.mean()
    simul_perf, simul_wt = simulate_portfolios(
        mean_returns, cov_matrix, short=False, rf_rate=rf_rate
    )

    df = pd.DataFrame(simul_wt)
    print(df.describe())

    ax = simul_perf.plot.scatter(
        x=0,
        y=1,
        c=2,
        cmap="Blues",
        alpha=0.5,
        figsize=(14, 9),
        colorbar=True,
        title=f"{NUM_PF:,d} Simulated Portfolios",
    )

    max_sharpe_idx = simul_perf.iloc[:, 2].idxmax()
    sd, r = simul_perf.iloc[max_sharpe_idx, :2].values
    print(f"Max Sharpe: {sd:.2%}, {r:.2%}")
    ax.scatter(sd, r, marker="*", color="darkblue", s=500, label="Max. Sharpe Ratio")
    plt.show()

    min_vol_idx = simul_perf.iloc[:, 0].idxmin()
    sd, r = simul_perf.iloc[min_vol_idx, :2].values
    ax.scatter(sd, r, marker="*", color="green", s=500, label="Min Volatility")
    plt.legend(labelspacing=1, loc="upper left")
    plt.tight_layout()
    plt.show()

    weight_constraint = {"type": "eq", "fun": lambda x: np.sum(np.abs(x)) - 1}
    simul_perf, simul_wt = simulate_portfolios(
        mean_returns, cov_matrix, rf_rate=rf_rate, short=False
    )
    print(simul_perf.describe())
    simul_max_sharpe = simul_perf.iloc[:, 2].idxmax()
    print(simul_perf.iloc[simul_max_sharpe])

    max_sharpe_pf = max_sharpe_ratio(mean_returns, cov_matrix, short=False)
    max_sharpe_perf = portfolio_performance(max_sharpe_pf.x, mean_returns, cov_matrix)

    r, sd = max_sharpe_perf
    pd.Series({"ret": r, "sd": sd, "sr": (r - rf_rate) / sd})

    min_vol_pf = min_vol(mean_returns, cov_matrix, short=False)
    min_vol_perf = portfolio_performance(min_vol_pf.x, mean_returns, cov_matrix)

    ret_range = np.linspace(simul_perf.iloc[:, 1].min(), simul_perf.iloc[:, 1].max(), 50)
    eff_pf = efficient_frontier(mean_returns, cov_matrix, ret_range, short=True)
    eff_pf = pd.Series(dict(zip([p["fun"] for p in eff_pf], ret_range)))

    fig, ax = plt.subplots()
    simul_perf.plot.scatter(
        x=0, y=1, c=2, ax=ax, cmap="Blues", alpha=0.25, figsize=(14, 9), colorbar=True
    )
    eff_pf[eff_pf.index.min() :].plot(
        linestyle="--", lw=2, ax=ax, c="k", label="Efficient Frontier"
    )
    plt.show()

    r, sd = max_sharpe_perf
    ax.scatter(sd, r, marker="*", color="k", s=500, label="Max Sharpe Ratio PF")
    r, sd = min_vol_perf
    ax.scatter(sd, r, marker="v", color="k", s=200, label="Min Volatility PF")

    kelly_wt = precision_matrix.dot(mean_returns).clip(lower=0).values
    kelly_wt /= np.sum(np.abs(kelly_wt))
    r, sd = portfolio_performance(kelly_wt, mean_returns, cov_matrix)
    ax.scatter(sd, r, marker="D", color="k", s=150, label="Kelly PF")

    std = weekly_returns.std()
    std /= std.sum()
    r, sd = portfolio_performance(std, mean_returns, cov_matrix)
    ax.scatter(sd, r, marker="X", color="k", s=250, label="Risk Parity PF")

    r, sd = portfolio_performance(np.full(n_assets, 1 / n_assets), mean_returns, cov_matrix)
    ax.scatter(sd, r, marker="o", color="k", s=200, label="1/n PF")

    ax.legend(labelspacing=0.8)
    ax.set_xlim(0, eff_pf.max() + 0.4)
    ax.set_title("Mean-Variance Efficient Frontier", fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    sns.despine()
    fig.tight_layout()
    plt.show()
