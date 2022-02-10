# How to use the VAR model for macro fundamentals forecasts
# The vector autoregressive VAR(p) model extends the AR(p) model to k series by creating a system of k equations where
# each contains p lagged values of all k series. The coefficients on the own lags provide information about the dynamics
# of the series itself, whereas the cross-variable coefficients offer some insight into the interactions across the series.
import numpy as np
import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns

from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import minmax_scale
from scipy.stats import probplot, moment
from sklearn.metrics import mean_absolute_error
import warnings

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")


def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x) / 5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    x.plot(ax=axes[0][0], title="Residuals")
    x.rolling(21).mean().plot(ax=axes[0][0], c="k", lw=1)
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f"Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}"
    axes[0][0].text(x=0.02, y=0.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f"Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}"
    axes[0][1].text(x=0.02, y=0.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel("Lag")
    axes[1][1].set_xlabel("Lag")
    fig.suptitle(title, fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)


def test_unit_root(df):
    return df.apply(lambda x: f"{pd.Series(adfuller(x)).iloc[1]:.2%}").to_frame("p-value")


if __name__ == "__main__":
    # We will extend the univariate example of a single time series of monthly data on industrial production and add
    # a monthly time series on consumer sentiment, both provided by the Federal Reserve's data service. We will use
    # the familiar pandas-datareader library to retrieve data from 1970 through 2017:
    sent = "UMCSENT"
    df = web.DataReader(["UMCSENT", "IPGMFN"], "fred", "1970", "2019-12").dropna()
    df.columns = ["sentiment", "ip"]
    df.info()

    df.plot(subplots=True, figsize=(14, 8), rot=0)
    plt.tight_layout()
    plt.savefig("images/04_01.png")

    plot_correlogram(df.sentiment, lags=24)
    plt.savefig("images/04_02.png")

    plot_correlogram(df.ip, lags=24)
    plt.savefig("images/04_03.png")

    ## Stationarity Transform
    # Log-transforming the industrial production series and seasonal differencing using lag 12 of both series yields
    # stationary results:
    df_transformed = pd.DataFrame(
        {"ip": np.log(df.ip).diff(12), "sentiment": df.sentiment.diff(12)}
    ).dropna()

    plot_correlogram(df_transformed.sentiment, lags=24)
    plt.savefig("images/04_04.png")

    plot_correlogram(df_transformed.ip, lags=24)
    plt.savefig("images/04_05.png")

    test_unit_root(df_transformed)
    df_transformed.plot(
        subplots=True,
        figsize=(14, 8),
        title=["Industrial Production", "Consumer Sentiment"],
        legend=False,
        rot=0,
    )
    plt.tight_layout()
    plt.savefig("images/04_06.png")

    ## VAR Model
    # To limit the size of the output, we will just estimate a VAR(1) model using the statsmodels VARMAX implementation
    # (which allows for optional exogenous variables) with a constant trend through 2017. The output contains the
    # coefficients for both time series equations.

    df_transformed = df_transformed.apply(minmax_scale)
    model = VARMAX(df_transformed.loc[:"2017"], order=(1, 1), trend="c").fit(maxiter=1000)
    print(model.summary())

    ### Plot Diagnostics
    # `statsmodels` provides diagnostic plots to check whether the residuals meet the white noise assumptions, which are
    # not exactly met in this simple case:

    #### Industrial Production
    model.plot_diagnostics(variable=0, figsize=(14, 8), lags=24)
    plt.gcf().suptitle("Industrial Production - Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig("images/04_07.png")

    #### Sentiment
    model.plot_diagnostics(variable=1, figsize=(14, 8), lags=24)
    plt.title("Sentiment - Diagnostics")
    plt.savefig("images/04_08.png")

    ### Impulse-Response Function
    median_change = df_transformed.diff().quantile(0.5).tolist()
    model.impulse_responses(steps=12, impulse=median_change).plot.bar(
        subplots=True, figsize=(12, 6), rot=0, legend=False
    )
    plt.tight_layout()
    plt.savefig("images/04_09.png")

    ### Generate Predictions
    # Out-of-sample predictions can be generated as follows:
    n = len(df_transformed)
    start = n - 24
    preds = model.predict(start=start + 1, end=n)
    preds.index = df_transformed.index[start:]

    fig, axes = plt.subplots(nrows=2, figsize=(14, 8), sharex=True)
    df_transformed.ip.loc["2010":].plot(ax=axes[0], label="actual", title="Industrial Production")
    preds.ip.plot(label="predicted", ax=axes[0])
    trans = mtransforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    axes[0].legend()
    axes[0].fill_between(
        x=df_transformed.index[start + 1 :], y1=0, y2=1, transform=trans, color="grey", alpha=0.5
    )

    trans = mtransforms.blended_transform_factory(axes[0].transData, axes[1].transAxes)
    df_transformed.sentiment.loc["2010":].plot(ax=axes[1], label="actual", title="Sentiment")
    preds.sentiment.plot(label="predicted", ax=axes[1])
    axes[1].fill_between(
        x=df_transformed.index[start + 1 :], y1=0, y2=1, transform=trans, color="grey", alpha=0.5
    )
    axes[1].set_xlabel("")
    fig.tight_layout()
    plt.savefig("images/04_10.png")

    ### Out-of-sample forecasts
    # A visualization of actual and predicted values shows how the prediction lags the actual values and does not
    # capture non-linear out-of-sample patterns well:

    forecast = model.forecast(steps=24)

    fig, axes = plt.subplots(nrows=2, figsize=(14, 8), sharex=True)
    df_transformed["2010":].ip.plot(ax=axes[0], label="actual", title="Liquor")
    preds.ip.plot(label="predicted", ax=axes[0])
    axes[0].legend()
    df_transformed["2010":].sentiment.plot(ax=axes[1], label="actual", title="Sentiment")
    preds.sentiment.plot(label="predicted", ax=axes[1])
    axes[1].legend()
    axes[1].set_xlabel("")
    fig.tight_layout()
    plt.savefig("images/04_11.png")

    print(mean_absolute_error(forecast, df_transformed.iloc[492:]))
