# Long-Short Strategy, Part 3: Evaluating our Boosting Model Signals
# Cross-validation of numerous configurations has produced a large number of results. Now, we need to evaluate the
# predictive performance to identify the model that generates the most reliable and profitable signals
# for our prospective trading strategy.

from time import time
from io import StringIO
import sys, os
import warnings
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr

from alphalens import plotting
from alphalens import performance as perf
from alphalens.utils import get_clean_factor_and_forward_returns, rate_of_return, std_conversion
from alphalens.tears import create_summary_tear_sheet, create_full_tear_sheet


YEAR = 252

idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.2f}".format

results_path = Path("../data/ch12", "us_stocks")
if not results_path.exists():
    results_path.mkdir(parents=True)


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(
        self,
        n_splits=3,
        train_period_length=126,
        test_period_length=21,
        lookahead=None,
        date_idx="date",
        shuffle=False,
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[
                (dates[self.date_idx] > days[train_start])
                & (dates[self.date_idx] <= days[train_end])
            ].index
            test_idx = dates[
                (dates[self.date_idx] > days[test_start]) & (dates[self.date_idx] <= days[test_end])
            ].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


if __name__ == "__main__":
    scope_params = ["lookahead", "train_length", "test_length"]
    daily_ic_metrics = ["daily_ic_mean", "daily_ic_mean_n", "daily_ic_median", "daily_ic_median_n"]
    lgb_train_params = ["learning_rate", "num_leaves", "feature_fraction", "min_data_in_leaf"]
    catboost_train_params = ["max_depth", "min_child_samples"]

    ## Collect Data
    # We produced a larger number of LightGBM models because it runs an order of magnitude faster than CatBoost
    # and will demonstrate some evaluation strategies accordingly.

    ### LightGBM
    #### Summary Metrics by Fold
    # First, we collect the summary metrics computed for each fold and hyperparameter combination:
    with pd.HDFStore(results_path / "tuning_lgb.h5") as store:
        for i, key in enumerate([k[1:] for k in store.keys() if k[1:].startswith("metrics")]):
            _, t, train_length, test_length = key.split("/")[:4]
            attrs = {"lookahead": t, "train_length": train_length, "test_length": test_length}
            s = store[key].to_dict()
            s.update(attrs)
            if i == 0:
                lgb_metrics = pd.Series(s).to_frame(i)
            else:
                lgb_metrics[i] = pd.Series(s)

    id_vars = scope_params + lgb_train_params + daily_ic_metrics
    lgb_metrics = (
        pd.melt(
            lgb_metrics.T.drop("t", axis=1),
            id_vars=id_vars,
            value_name="ic",
            var_name="boost_rounds",
        )
        .dropna()
        .apply(pd.to_numeric)
    )

    lgb_metrics.to_hdf(results_path / "model_tuning.h5", "lgb/metrics")
    lgb_metrics.info()
    print(lgb_metrics.groupby(scope_params).size())

    #### Information Coefficient by Day
    # Next, we retrieve the IC per day computed during cross-validation:
    int_cols = ["lookahead", "train_length", "test_length", "boost_rounds"]

    lgb_ic = []
    with pd.HDFStore(results_path / "tuning_lgb.h5") as store:
        keys = [k[1:] for k in store.keys()]
        for key in keys:
            _, t, train_length, test_length = key.split("/")[:4]
            if key.startswith("daily_ic"):
                df = (
                    store[key]
                    .drop(["boosting", "objective", "verbose"], axis=1)
                    .assign(lookahead=t, train_length=train_length, test_length=test_length)
                )
                lgb_ic.append(df)
        lgb_ic = pd.concat(lgb_ic).reset_index()

    id_vars = ["date"] + scope_params + lgb_train_params
    lgb_ic = pd.melt(lgb_ic, id_vars=id_vars, value_name="ic", var_name="boost_rounds").dropna()
    lgb_ic.loc[:, int_cols] = lgb_ic.loc[:, int_cols].astype(int)

    lgb_ic.to_hdf(results_path / "model_tuning.h5", "lgb/ic")
    lgb_ic.info(show_counts=True)

    lgb_daily_ic = (
        lgb_ic.groupby(id_vars[1:] + ["boost_rounds"]).ic.mean().to_frame("ic").reset_index()
    )
    lgb_daily_ic.to_hdf(results_path / "model_tuning.h5", "lgb/daily_ic")
    lgb_daily_ic.info()

    lgb_ic = pd.read_hdf(results_path / "model_tuning.h5", "lgb/ic")
    lgb_daily_ic = pd.read_hdf(results_path / "model_tuning.h5", "lgb/daily_ic")

    ### CatBoost
    # We proceed similarly for CatBoost:

    #### Summary Metrics
    with pd.HDFStore(results_path / "tuning_catboost.h5") as store:
        for i, key in enumerate([k[1:] for k in store.keys() if k[1:].startswith("metrics")]):
            _, t, train_length, test_length = key.split("/")[:4]
            attrs = {"lookahead": t, "train_length": train_length, "test_length": test_length}
            s = store[key].to_dict()
            s.update(attrs)
            if i == 0:
                catboost_metrics = pd.Series(s).to_frame(i)
            else:
                catboost_metrics[i] = pd.Series(s)

    id_vars = scope_params + catboost_train_params + daily_ic_metrics
    catboost_metrics = (
        pd.melt(
            catboost_metrics.T.drop("t", axis=1),
            id_vars=id_vars,
            value_name="ic",
            var_name="boost_rounds",
        )
        .dropna()
        .apply(pd.to_numeric)
    )
    catboost_metrics.info()
    print(catboost_metrics.groupby(scope_params).size())

    #### Daily Information Coefficient
    catboost_ic = []
    with pd.HDFStore(results_path / "tuning_catboost.h5") as store:
        keys = [k[1:] for k in store.keys()]
        for key in keys:
            _, t, train_length, test_length = key.split("/")[:4]
            if key.startswith("daily_ic"):
                df = (
                    store[key]
                    .drop("task_type", axis=1)
                    .assign(lookahead=t, train_length=train_length, test_length=test_length)
                )
                catboost_ic.append(df)
        catboost_ic = pd.concat(catboost_ic).reset_index()

    id_vars = ["date"] + scope_params + catboost_train_params
    catboost_ic = pd.melt(
        catboost_ic, id_vars=id_vars, value_name="ic", var_name="boost_rounds"
    ).dropna()
    catboost_ic.loc[:, int_cols] = catboost_ic.loc[:, int_cols].astype(int)

    catboost_ic.to_hdf(results_path / "model_tuning.h5", "catboost/ic")
    catboost_ic.info(show_counts=True)

    catboost_daily_ic = (
        catboost_ic.groupby(id_vars[1:] + ["boost_rounds"]).ic.mean().to_frame("ic").reset_index()
    )
    catboost_daily_ic.to_hdf(results_path / "model_tuning.h5", "catboost/daily_ic")
    catboost_daily_ic.info()

    catboost_ic = pd.read_hdf(results_path / "model_tuning.h5", "catboost/ic")
    catboost_daily_ic = pd.read_hdf(results_path / "model_tuning.h5", "catboost/daily_ic")

    ## Validation Performance: Daily vs Overall Information Coefficient
    # The following image shows that LightGBM (in orange) performs (slightly) better than CatBoost, especially
    # for longer horizons. This is not an entirely fair comparison because we ran more configurations for LightGBM,
    # which also, unsurprisingly, shows a wider dispersion of outcomes:
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5), sharey=True)
    sns.boxenplot(
        x="lookahead",
        y="ic",
        hue="model",
        data=catboost_metrics.assign(model="catboost").append(lgb_metrics.assign(model="lightgbm")),
        ax=axes[0],
    )
    axes[0].axhline(0, ls="--", lw=1, c="k")
    axes[0].set_title("Overall IC")
    sns.boxenplot(
        x="lookahead",
        y="ic",
        hue="model",
        data=catboost_daily_ic.assign(model="catboost").append(
            lgb_daily_ic.assign(model="lightgbm")
        ),
        ax=axes[1],
    )
    axes[1].axhline(0, ls="--", lw=1, c="k")
    axes[1].set_title("Daily IC")
    fig.tight_layout()
    plt.savefig("images/06-01.png")

    ## HyperParameter Impact: Linear Regression
    # Next, we'd like to understand if there's a systematic, statistical relationship between the hyperparameters and
    # the outcomes across daily predictions. To this end, we will run a linear regression using the various LightGBM
    # hyperparameter settings as dummy variables and the daily validation IC as the outcome.
    # The below chart shows the coefficient estimates and their confidence intervals for 1- and 21-day forecast horizons.
    # - For the shorter horizon, a longer lookback period, a higher learning rate, and deeper trees (more leaf nodes)
    #   have a positive impact.
    # - For the longer horizon, the picture is a little less clear: shorter trees do better, but the lookback period
    #   is not significant. A higher feature sampling rate also helps. In both cases, a larger ensemble does better.
    # Note that these results apply to this specific example only.
    lin_reg = {}
    for t in [1, 21]:
        df_ = lgb_ic[lgb_ic.lookahead == t]
        y, X = df_.ic, df_.drop(["ic"], axis=1)
        X = sm.add_constant(pd.get_dummies(X, columns=X.columns, drop_first=True))
        model = sm.OLS(endog=y, exog=X)
        lin_reg[t] = model.fit()
        s = lin_reg[t].summary()
        coefs = pd.read_csv(StringIO(s.tables[1].as_csv())).rename(columns=lambda x: x.strip())
        coefs.columns = ["variable", "coef", "std_err", "t", "p_value", "ci_low", "ci_high"]
        coefs.to_csv(f"results/linreg_result_{t:02}.csv", index=False)

    def visualize_lr_result(model, ax):
        ci = model.conf_int()
        errors = ci[1].sub(ci[0]).div(2)

        coefs = (
            model.params.to_frame("coef")
            .assign(error=errors)
            .reset_index()
            .rename(columns={"index": "variable"})
        )
        coefs = coefs[~coefs["variable"].str.startswith("date") & (coefs.variable != "const")]

        coefs.plot(
            x="variable",
            y="coef",
            kind="bar",
            ax=ax,
            color="none",
            capsize=3,
            yerr="error",
            legend=False,
        )
        ax.set_ylabel("IC")
        ax.set_xlabel("")
        ax.scatter(x=pd.np.arange(len(coefs)), marker="_", s=120, y=coefs["coef"], color="black")
        ax.axhline(y=0, linestyle="--", color="black", linewidth=1)
        ax.xaxis.set_ticks_position("none")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
    axes = axes.flatten()
    for i, t in enumerate([1, 21]):
        visualize_lr_result(lin_reg[t], axes[i])
        axes[i].set_title(f"Lookahead: {t} Day(s)")
    fig.suptitle("OLS Coefficients & Confidence Intervals", fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig("images/06-02.png")

    ## Cross-validation Result: Best Hyperparameters
    ### LightGBM
    # The top-performing LightGBM models use the following parameters for the three different prediction horizons.
    group_cols = scope_params + lgb_train_params + ["boost_rounds"]
    print(lgb_daily_ic.groupby("lookahead", group_keys=False).apply(lambda x: x.nlargest(3, "ic")))

    print(lgb_metrics.groupby("lookahead", group_keys=False).apply(lambda x: x.nlargest(3, "ic")))
    lgb_metrics.groupby("lookahead", group_keys=False).apply(lambda x: x.nlargest(3, "ic")).to_csv(
        results_path / "best_lgb_model.csv", index=False
    )

    print(
        lgb_metrics.groupby("lookahead", group_keys=False).apply(
            lambda x: x.nlargest(3, "daily_ic_mean")
        )
    )

    ### CatBoost
    group_cols = scope_params + catboost_train_params + ["boost_rounds"]
    print(
        catboost_daily_ic.groupby("lookahead", group_keys=False).apply(
            lambda x: x.nlargest(3, "ic")
        )
    )
    print(
        catboost_metrics.groupby("lookahead", group_keys=False).apply(lambda x: x.nlargest(3, "ic"))
    )
    print(
        catboost_metrics.groupby("lookahead", group_keys=False).apply(
            lambda x: x.nlargest(3, "daily_ic_mean")
        )
    )

    fig = plt.figure(figsize=(10, 6))
    sns.jointplot(x=lgb_metrics.daily_ic_mean, y=lgb_metrics.ic)
    plt.savefig("images/06-03.png")

    ### Visualization
    #### LightGBM
    fig = plt.figure(figsize=(10, 6))
    g = sns.catplot(
        x="lookahead", y="ic", col="train_length", row="test_length", data=lgb_metrics, kind="box"
    )
    plt.savefig("images/06-04.png")

    t = 1
    fig = plt.figure(figsize=(10, 6))
    g = sns.catplot(
        x="boost_rounds",
        y="ic",
        col="train_length",
        row="test_length",
        data=lgb_daily_ic[lgb_daily_ic.lookahead == t],
        kind="box",
    )
    plt.savefig("images/06-05.png")

    #### CatBoost
    # Some figures are empty because we did not run those parameter combinations.
    t = 1
    fig = plt.figure(figsize=(10, 6))
    g = sns.catplot(
        x="boost_rounds",
        y="ic",
        col="train_length",
        row="test_length",
        data=catboost_metrics[catboost_metrics.lookahead == t],
        kind="box",
    )
    plt.savefig("images/06-06.png")

    t = 1
    train_length = 1134
    test_length = 63
    fig = plt.figure(figsize=(10, 6))
    g = sns.catplot(
        x="boost_rounds",
        y="ic",
        col="max_depth",
        hue="min_child_samples",
        data=catboost_daily_ic[
            (catboost_daily_ic.lookahead == t)
            & (catboost_daily_ic.train_length == train_length)
            & (catboost_daily_ic.test_length == test_length)
        ],
        kind="swarm",
    )
    plt.savefig("images/06-07.png")

    ## AlphaLens Analysis - Validation Performance
    ### LightGBM
    #### Select Parameters
    lgb_daily_ic = pd.read_hdf(results_path / "model_tuning.h5", "lgb/daily_ic")
    lgb_daily_ic.info()

    def get_lgb_params(data, t=5, best=0):
        param_cols = scope_params[1:] + lgb_train_params + ["boost_rounds"]
        df = data[data.lookahead == t].sort_values("ic", ascending=False).iloc[best]
        return df.loc[param_cols]

    def get_lgb_key(t, p):
        key = f"{t}/{int(p.train_length)}/{int(p.test_length)}/{p.learning_rate}/"
        return key + f"{int(p.num_leaves)}/{p.feature_fraction}/{int(p.min_data_in_leaf)}"

    best_params = get_lgb_params(lgb_daily_ic, t=1, best=0)
    print(best_params)

    best_params.to_hdf("../data/12_data.h5", "best_params")

    #### Plot rolling IC
    def select_ic(params, ic_data, lookahead):
        return ic_data.loc[
            (ic_data.lookahead == lookahead)
            & (ic_data.train_length == params.train_length)
            & (ic_data.test_length == params.test_length)
            & (ic_data.learning_rate == params.learning_rate)
            & (ic_data.num_leaves == params.num_leaves)
            & (ic_data.feature_fraction == params.feature_fraction)
            & (ic_data.boost_rounds == params.boost_rounds),
            ["date", "ic"],
        ].set_index("date")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    axes = axes.flatten()
    for i, t in enumerate([1, 21]):
        params = get_lgb_params(lgb_daily_ic, t=t)
        data = select_ic(params, lgb_ic, lookahead=t).sort_index()
        rolling = data.rolling(63).ic.mean().dropna()
        avg = data.ic.mean()
        med = data.ic.median()
        rolling.plot(
            ax=axes[i], title=f"Horizon: {t} Day(s) | IC: Mean={avg*100:.2f}   Median={med*100:.2f}"
        )
        axes[i].axhline(avg, c="darkred", lw=1)
        axes[i].axhline(0, ls="--", c="k", lw=1)

    fig.suptitle("3-Month Rolling Information Coefficient", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig("images/06-08.png")

    #### Get Predictions for Validation Period
    # We retrieve the predictions for the 10 validation runs:
    lookahead = 1
    topn = 10
    best_predictions = []
    for best in range(topn):
        best_params = get_lgb_params(lgb_daily_ic, t=lookahead, best=best)
        key = get_lgb_key(lookahead, best_params)
        rounds = str(int(best_params.boost_rounds))
        if best == 0:
            best_predictions = pd.read_hdf(results_path / "tuning_lgb.h5", "predictions/" + key)
            best_predictions = best_predictions[rounds].to_frame(best)
        else:
            best_predictions[best] = pd.read_hdf(
                results_path / "tuning_lgb.h5", "predictions/" + key
            )[rounds]
    best_predictions = best_predictions.sort_index()

    best_predictions.to_hdf(results_path / "predictions.h5", f"lgb/train/{lookahead:02}")
    best_predictions.info()

    #### Get Trade Prices
    # Using next available prices.
    def get_trade_prices(tickers):
        idx = pd.IndexSlice
        DATA_STORE = "../data/assets.h5"
        prices = pd.read_hdf(DATA_STORE, "quandl/wiki/prices").swaplevel().sort_index()
        prices.index.names = ["symbol", "date"]
        return (
            prices.loc[idx[tickers, "2015":"2017"], "adj_open"]
            .unstack("symbol")
            .sort_index()
            .shift(-1)
            .tz_localize("UTC")
        )

    test_tickers = best_predictions.index.get_level_values("symbol").unique()
    trade_prices = get_trade_prices(test_tickers)
    trade_prices.info()

    # persist result in case we want to rerun:
    trade_prices.to_hdf(results_path / "model_tuning.h5", "trade_prices/model_selection")
    trade_prices = pd.read_hdf(results_path / "model_tuning.h5", "trade_prices/model_selection")

    # We average the top five models and provide the corresponding prices to Alphalens, in order to compute the mean
    # period-wise return earned on an equal-weighted portfolio invested in the daily factor quintiles for various
    # holding periods:
    factor = (
        best_predictions.iloc[:, :5].mean(1).dropna().tz_localize("UTC", level="date").swaplevel()
    )

    #### Create AlphaLens Inputs
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    )

    #### Compute Alphalens metrics
    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=True,
        group_adjust=False,
    )
    factor_returns = perf.factor_returns(factor_data)

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=True
    )

    mean_quant_rateret = mean_quant_ret.apply(
        rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=True,
        group_adjust=False,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        rate_of_return,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        std_conversion, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(factor_data, demeaned=True)

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    print(
        mean_ret_spread_quant.mean()
        .mul(10000)
        .to_frame("Mean Period Wise Spread (bps)")
        .join(alpha_beta.T)
        .T
    )

    fig, axes = plt.subplots(ncols=3, figsize=(18, 4))
    plotting.plot_quantile_returns_bar(mean_quant_rateret, ax=axes[0])
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)
    axes[0].set_xlabel("Quantile")

    plotting.plot_cumulative_returns_by_quantile(
        mean_quant_ret_bydate["1D"], freq=pd.tseries.offsets.BDay(), period="1D", ax=axes[1]
    )
    axes[1].set_title("Cumulative Return by Quantile (1D Period)")

    title = "Cumulative Return - Factor-Weighted Long/Short PF (1D Period)"
    plotting.plot_cumulative_returns(
        factor_returns["1D"], period="1D", freq=pd.tseries.offsets.BDay(), title=title, ax=axes[2]
    )

    fig.suptitle("Alphalens - Validation Set Performance", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig("images/06-09.png")

    #### Summary Tearsheet
    create_summary_tear_sheet(factor_data)
    create_full_tear_sheet(factor_data)

    ### CatBoost
    #### Select Parameters
    catboost_daily_ic = pd.read_hdf(results_path / "model_tuning.h5", "catboost/daily_ic")
    catboost_daily_ic.info()

    def get_cb_params(data, t=5, best=0):
        param_cols = scope_params[1:] + catboost_train_params + ["boost_rounds"]
        df = data[data.lookahead == t].sort_values("ic", ascending=False).iloc[best]
        return df.loc[param_cols]

    def get_cb_key(t, p):
        key = f"{t}/{int(p.train_length)}/{int(p.test_length)}/"
        return key + f"{int(p.max_depth)}/{int(p.min_child_samples)}"

    best_params = get_cb_params(catboost_daily_ic, t=1, best=0)
    print(best_params)

    def select_cb_ic(params, ic_data, lookahead):
        return ic_data.loc[
            (ic_data.lookahead == lookahead)
            & (ic_data.train_length == params.train_length)
            & (ic_data.test_length == params.test_length)
            & (ic_data.max_depth == params.max_depth)
            & (ic_data.min_child_samples == params.min_child_samples)
        ].set_index("date")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    axes = axes.flatten()
    for i, t in enumerate([1, 21]):
        params = get_cb_params(catboost_daily_ic, t=t)
        data = select_cb_ic(params, catboost_ic, lookahead=t).sort_index()
        rolling = data.rolling(63).ic.mean().dropna()
        avg = data.ic.mean()
        med = data.ic.median()
        rolling.plot(
            ax=axes[i], title=f"Horizon: {t} Day(s) | IC: Mean={avg*100:.2f}   Median={med*100:.2f}"
        )
        axes[i].axhline(avg, c="darkred", lw=1)
        axes[i].axhline(0, ls="--", c="k", lw=1)

    fig.suptitle("3-Month Rolling Information Coefficient", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig("images/06-10.png")

    #### Get Predictions
    lookahead = 1
    topn = 10
    for best in range(topn):
        best_params = get_cb_params(catboost_daily_ic, t=lookahead, best=best)
        key = get_cb_key(lookahead, best_params)
        rounds = str(int(best_params.boost_rounds))
        if best == 0:
            best_predictions = pd.read_hdf(
                results_path / "tuning_catboost.h5", "predictions/" + key
            )
            best_predictions = best_predictions[rounds].to_frame(best)
        else:
            best_predictions[best] = pd.read_hdf(
                results_path / "tuning_catboost.h5", "predictions/" + key
            )[rounds]
    best_predictions = best_predictions.sort_index()

    best_predictions.to_hdf(results_path / "predictions.h5", f"catboost/train/{lookahead:02}")
    best_predictions.info()

    #### Get Trade Prices
    # Using next available prices.
    def get_trade_prices(tickers):
        idx = pd.IndexSlice
        DATA_STORE = "../data/assets.h5"
        prices = pd.read_hdf(DATA_STORE, "quandl/wiki/prices").swaplevel().sort_index()
        prices.index.names = ["symbol", "date"]
        return (
            prices.loc[idx[tickers, "2015":"2017"], "adj_open"]
            .unstack("symbol")
            .sort_index()
            .shift(-1)
            .tz_localize("UTC")
        )

    test_tickers = best_predictions.index.get_level_values("symbol").unique()
    trade_prices = get_trade_prices(test_tickers)
    trade_prices.info()

    # only generate once to save time
    trade_prices.to_hdf(results_path / "model_tuning.h5", "trade_prices/model_selection")

    trade_prices = pd.read_hdf(results_path / "model_tuning.h5", "trade_prices/model_selection")
    factor = (
        best_predictions.iloc[:, :5].mean(1).dropna().tz_localize("UTC", level="date").swaplevel()
    )

    #### Create AlphaLens Inputs
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    )

    #### Summary Tearsheet
    create_summary_tear_sheet(factor_data)
    create_full_tear_sheet(factor_data)
