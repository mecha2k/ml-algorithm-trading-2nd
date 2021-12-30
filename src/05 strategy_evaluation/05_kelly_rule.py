# How to size your bets - The Kelly Rule
# The Kelly rule has a long history in gambling because it provides guidance on how much to stake on each of
# an (infinite) sequence of bets with varying (but favorable) odds to maximize terminal wealth. It was published
# as A New Interpretation of the Information Rate in 1956 by John Kelly who was a colleague of Claude Shannon's
# at Bell Labs. He was intrigued by bets placed on candidates at the new quiz show The $64,000 Question,
# where a viewer on the west coast used the three-hour delay to obtain insider information about the winners.
# Kelly drew a connection to Shannon's information theory to solve for the bet that is optimal for long-term capital
# growth when the odds are favorable, but uncertainty remains. His rule maximizes logarithmic wealth as a function
# of the odds of success of each game, and includes implicit bankruptcy protection since log(0) is negative infinity
# so that a Kelly gambler would naturally avoid losing everything.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from pathlib import Path
from numpy.linalg import inv
from numpy.random import dirichlet
from sympy import symbols, solve, log, diff
from scipy.optimize import minimize_scalar, newton, minimize
from scipy.integrate import quad
from scipy.stats import norm


np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    DATA_STORE = Path("..", "data", "assets.h5")

    ## The optimal size of a bet
    # Kelly began by analyzing games with a binary win-lose outcome. The key variables are:
    # - b: The odds define the amount won for a \\$1 bet. Odds = 5/1 implies a \\$5 gain if the bet wins, plus recovery
    #      of the \\$1 capital.
    # - p: The probability defines the likelihood of a favorable outcome.
    # - f: The share of the current capital to bet.
    # - V: The value of the capital as a result of betting.
    #
    # The Kelly rule aims to maximize the value's growth rate, G, of infinitely-repeated bets.
    # We can maximize the rate of growth G by maximizing G with respect to f, as illustrated using sympy as follows:

    share, odds, probability = symbols("share odds probability")
    Value = probability * log(1 + odds * share) + (1 - probability) * log(1 - share)
    solve(diff(Value, share), share)

    f, p = symbols("f p")
    y = p * log(1 + f) + (1 - p) * log(1 - f)
    solve(diff(y, f), f)

    with pd.HDFStore(DATA_STORE) as store:
        sp500 = store["sp500/stooq"].close

    ### Compute Returns & Standard Deviation
    annual_returns = sp500.resample("A").last().pct_change().dropna().to_frame("sp500")
    return_params = annual_returns.sp500.rolling(25).agg(["mean", "std"]).dropna()
    return_ci = (
        return_params[["mean"]]
        .assign(lower=return_params["mean"].sub(return_params["std"].mul(2)))
        .assign(upper=return_params["mean"].add(return_params["std"].mul(2)))
    )

    return_ci.plot(lw=2, figsize=(14, 8))
    plt.tight_layout()
    plt.savefig("images/05-01.png", bboxinches="tight")

    ### Kelly Rule for a Single Asset - Index Returns
    # In a financial market context, both outcomes and alternatives are more complex, but the Kelly rule logic does
    # still apply. It was made popular by Ed Thorp, who first applied it profitably to gambling (described in Beat
    # the Dealer) and later started the successful hedge fund Princeton/Newport Partners.
    # With continuous outcomes, the growth rate of capital is defined by an integrated over the probability
    # distribution of the different returns that can be optimized numerically.
    # We can solve this expression (see book) for the optimal f* using the `scipy.optimize` module:

    def norm_integral(f, mean, std):
        val, er = quad(
            lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std), mean - 3 * std, mean + 3 * std
        )
        return -val

    def norm_dev_integral(f, mean, std):
        val, er = quad(
            lambda s: (s / (1 + f * s)) * norm.pdf(s, mean, std), m - 3 * std, mean + 3 * std
        )
        return val

    def get_kelly_share(data):
        solution = minimize_scalar(
            norm_integral, args=(data["mean"], data["std"]), bounds=[0, 2], method="bounded"
        )
        return solution.x

    annual_returns["f"] = return_params.apply(get_kelly_share, axis=1)
    return_params.plot(subplots=True, lw=2, figsize=(14, 8))
    plt.savefig("images/05-02.png", bboxinches="tight")
    print(annual_returns.tail())

    ### Performance Evaluation
    sp500_annual = (
        annual_returns[["sp500"]]
        .assign(kelly=annual_returns.sp500.mul(annual_returns.f.shift()))
        .dropna()
        .loc["1900":]
        .add(1)
        .cumprod()
        .sub(1)
    )
    sp500_annual.plot(lw=2)
    plt.savefig("images/05-03.png", bboxinches="tight")

    print(annual_returns.f.describe())
    print(return_ci.head())

    ### Compute Kelly Fraction
    m = 0.058
    s = 0.216

    # Option 1: minimize the expectation integral
    sol = minimize_scalar(norm_integral, args=(m, s), bounds=[0.0, 2.0], method="bounded")
    print("Optimal Kelly fraction: {:.4f}".format(sol.x))

    # Option 2: take the derivative of the expectation and make it null
    x0 = newton(norm_dev_integral, 0.1, args=(m, s))
    print("Optimal Kelly fraction: {:.4f}".format(x0))

    ## Kelly Rule for Multiple Assets
    # We will use an example with various equities. [E. Chan (2008)]
    # (https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889) illustrates how to arrive
    # at a multi-asset application of the Kelly Rule, and that the result is equivalent to the (potentially levered)
    # maximum Sharpe ratio portfolio from the mean-variance optimization.
    # The computation involves the dot product of the precision matrix, which is the inverse of the covariance matrix,
    # and the return matrix:

    with pd.HDFStore(DATA_STORE) as store:
        sp500_stocks = store["sp500/stocks"].index
        prices = store["quandl/wiki/prices"].adj_close.unstack("ticker").filter(sp500_stocks)
    prices.info()

    monthly_returns = (
        prices.loc["1988":"2017"].resample("M").last().pct_change().dropna(how="all").dropna(axis=1)
    )
    stocks = monthly_returns.columns
    monthly_returns.info()

    ### Compute Precision Matrix
    cov = monthly_returns.cov()
    precision_matrix = pd.DataFrame(inv(cov), index=stocks, columns=stocks)
    kelly_allocation = monthly_returns.mean().dot(precision_matrix)
    print(kelly_allocation.describe())
    print(kelly_allocation.sum())

    ### Largest Portfolio Allocation
    # The plot shows the tickers that receive an allocation weight > 5x their value:
    kelly_allocation[kelly_allocation.abs() > 5].sort_values(ascending=False).plot.barh(
        figsize=(8, 10)
    )
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("images/05-04.png", bboxinches="tight")

    ### Performance vs SP500
    # The Kelly rule does really well. But it has also been computed from historical data..
    ax = (
        monthly_returns.loc["2010":]
        .mul(kelly_allocation.div(kelly_allocation.sum()))
        .sum(1)
        .to_frame("Kelly")
        .add(1)
        .cumprod()
        .sub(1)
        .plot(figsize=(14, 4))
    )
    sp500.filter(monthly_returns.loc["2010":].index).pct_change().add(1).cumprod().sub(1).to_frame(
        "SP500"
    ).plot(ax=ax, legend=True)
    plt.tight_layout()
    plt.savefig("images/05-05.png", bboxinches="tight")
