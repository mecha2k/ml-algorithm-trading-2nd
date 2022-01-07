# How to generate long-short trading signals with a Random Forest

from time import time
from io import StringIO
import sys, os
from tqdm import tqdm

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr


idx = pd.IndexSlice
np.random.seed(seed=42)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
pd.options.display.float_format = "{:,.2f}".format

DATA_DIR = Path("..", "data")

results_path = Path("../data/ch11", "return_predictions")
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
    YEAR = 252
    # See the notebook [japanese_equity_features](03_japanese_equity_features.ipynb) in this directory
    # for data preparation.
    data = pd.read_hdf("../data/11_data.h5", "stooq/japan/equities")
    data.info(show_counts=True)

    # We start with 941 tickers.
    print(len(data.index.unique("ticker")))

    ### Select universe of 250 most-liquid stocks
    # We rank the stocks by their daily average dollar volume and select those with the 250 lowest average ranks and
    # thus highest average volumes for the 2010-2017 period.
    prices = pd.read_hdf(DATA_DIR / "assets.h5", "stooq/jp/tse/stocks/prices").loc[
        idx[:, "2010":"2017"], :
    ]

    dollar_vol = prices.close.mul(prices.volume)
    dollar_vol_rank = dollar_vol.groupby(level="date").rank(ascending=False)
    universe = dollar_vol_rank.groupby(level="ticker").mean().nsmallest(250).index

    ## MultipleTimeSeriesCV
    # See [Chapter 7 - Linear Models](../07_linear_models) for details.
    cv = MultipleTimeSeriesCV(
        n_splits=36, test_period_length=21, lookahead=5, train_period_length=2 * 252
    )

    # For each fold, the train and test periods are separated by a `lookahead` number of periods and thus do not overlap:
    for i, (train_idx, test_idx) in enumerate(cv.split(X=data)):
        train = data.iloc[train_idx]
        train_dates = train.index.get_level_values("date")
        test = data.iloc[test_idx]
        test_dates = test.index.get_level_values("date")
        df = train.reset_index().append(test.reset_index())
        n = len(df)
        assert n == len(df.drop_duplicates())
        msg = f"Training: {train_dates.min().date()}-{train_dates.max().date()} "
        msg += f' ({train.groupby(level="ticker").size().value_counts().index[0]:,.0f} days) | '
        msg += f"Test: {test_dates.min().date()}-{test_dates.max().date()} "
        msg += f'({test.groupby(level="ticker").size().value_counts().index[0]:,.0f} days)'
        print(msg)
        if i == 3:
            break

    ## Model Selection: Time Period and Horizon
    # For the model selection step, we restrict training and validation sets to the 2010-2017 period.
    cv_data = data.loc[idx[universe, :"2017"], :]
    tickers = cv_data.index.unique("ticker")

    # Persist the data to save some time when running another experiment:
    cv_data.to_hdf("../data/11_data.h5", "stooq/japan/equities/cv_data")

    with pd.HDFStore("../data/11_data.h5") as store:
        print(store.info())

    # We're picking prediction horizons of 1, 5, 10 and 21 days:
    lookaheads = [1, 5, 10, 21]

    ## Baseline: Linear Regression
    # Since it's quick to run and quite informative, we generate linear regression baseline predictions.
    # See [Chapter 7 - Linear Models](../07_linear_models) for details.
    lr = LinearRegression()

    labels = sorted(cv_data.filter(like="fwd").columns)
    features = cv_data.columns.difference(labels).tolist()

    ### CV Parameters
    # We set five different training lengths from 3 months to 5 years, and two test periods as follows:
    train_lengths = [5 * YEAR, 3 * YEAR, YEAR, 126, 63]
    test_lengths = [5, 21]

    # Since linear regression has no hyperparameters, our CV parameters are the cartesian product of prediction horizon
    # and train/test period lengths:
    test_params = list(product(lookaheads, train_lengths, test_lengths))

    # Now we iterate over these parameters and train/validate the linear regression model while capturing the
    # information coefficient of the model predictions, measure both on a daily basis and for each complete fold:
    lr_metrics = []
    for lookahead, train_length, test_length in tqdm(test_params):
        label = f"fwd_ret_{lookahead:02}"
        df = cv_data.loc[:, features + [label]].dropna()
        X, y = df.drop(label, axis=1), df[label]

        n_splits = int(2 * YEAR / test_length)
        cv = MultipleTimeSeriesCV(
            n_splits=n_splits,
            test_period_length=test_length,
            lookahead=lookahead,
            train_period_length=train_length,
        )

        ic, preds = [], []
        for i, (train_idx, test_idx) in enumerate(cv.split(X=X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            preds.append(y_test.to_frame("y_true").assign(y_pred=y_pred))
            ic.append(spearmanr(y_test, y_pred)[0])
        preds = pd.concat(preds)
        lr_metrics.append(
            [
                lookahead,
                train_length,
                test_length,
                np.mean(ic),
                spearmanr(preds.y_true, preds.y_pred)[0],
            ]
        )

    columns = ["lookahead", "train_length", "test_length", "ic_by_day", "ic"]
    lr_metrics = pd.DataFrame(lr_metrics, columns=columns)
    lr_metrics.info()

    ### Information Coefficient distribution by Lookahead
    # Convert the data to long `seaborn`-friendly format:
    lr_metrics_long = pd.concat(
        [
            (
                lr_metrics.drop("ic", axis=1)
                .rename(columns={"ic_by_day": "ic"})
                .assign(Measured="By Day")
            ),
            lr_metrics.drop("ic_by_day", axis=1).assign(Measured="Overall"),
        ]
    )
    lr_metrics_long.columns = ["Lookahead", "Train Length", "Test Length", "IC", "Measure"]
    lr_metrics_long.info()

    # Plot both IC measures for the various CV parameters:
    sns.catplot(
        x="Train Length",
        y="IC",
        hue="Test Length",
        col="Lookahead",
        row="Measure",
        data=lr_metrics_long,
        kind="bar",
    )
    plt.tight_layout()
    plt.savefig("images/05-01.png", bboxinches="tight")

    # Compare the distributions of each IC metric for the different prediction horizons:
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), sharey=True)
    sns.boxplot(x="lookahead", y="ic_by_day", data=lr_metrics, ax=axes[0])
    axes[0].set_title("IC by Day")
    sns.boxplot(x="lookahead", y="ic", data=lr_metrics, ax=axes[1])
    axes[1].set_title("IC Overall")
    axes[0].set_ylabel("Information Coefficient")
    axes[1].set_ylabel("")
    fig.tight_layout()
    plt.savefig("images/05-02.png", bboxinches="tight")

    ### Best Train/Test Period Lengths
    # Show the best train/test period settings for the four prediction horizons:
    print(lr_metrics.groupby("lookahead", group_keys=False).apply(lambda x: x.nlargest(3, "ic")))

    lr_metrics.to_csv(results_path / "lin_reg_performance.csv", index=False)

    ## LightGBM Random Forest Model Tuning
    # Helper function to obtain the LightGBM feature importance metrics:
    def get_fi(model):
        fi = model.feature_importance(importance_type="gain")
        return pd.Series(fi / fi.sum(), index=model.feature_name())

    # LightGBM base parameter settings that are independent of hyperparameter tuning:
    base_params = dict(boosting_type="rf", objective="regression", bagging_freq=1, verbose=-1)

    ### Hyperparameter Options
    # We run this experiment with different parameters for the bagging and feature fractions that determine the degree
    # of randomization as well as the minimum number of samples for a split to control overfitting:
    bagging_fraction_opts = [0.5, 0.75, 0.95]
    feature_fraction_opts = [0.75, 0.95]
    min_data_in_leaf_opts = [250, 500, 1000]

    # This gives us 3x2x3=18 parameter combinations:
    cv_params = list(product(bagging_fraction_opts, feature_fraction_opts, min_data_in_leaf_opts))
    n_cv_params = len(cv_params)
    print(n_cv_params)

    #### Random Sample
    # To limit the running time, we can randomly sample a subset of the parameter combinations (here: 50%):
    sample_proportion = 0.5
    sample_size = int(sample_proportion * n_cv_params)

    cv_param_sample = np.random.choice(
        list(range(n_cv_params)), size=int(sample_size), replace=False
    )
    cv_params_ = [cv_params[i] for i in cv_param_sample]
    print("# CV parameters:", len(cv_params_))

    # We tune the number of trees by evaluating a fully grown forest for various smaller sizes:
    num_iterations = [25] + list(range(50, 501, 25))
    num_boost_round = num_iterations[-1]

    ### Train/Test Period Lenghts
    # As above for linear regression, we define a range of train/test period length:

    #### Define parameters
    train_lengths = [5 * YEAR, 3 * YEAR, YEAR, 126, 63]
    test_lengths = [5, 21]

    test_params = list(product(train_lengths, test_lengths))
    n_test_params = len(test_params)

    #### Random sample
    # Just as for the model parameters, we can randomly sample from the 5 x 2 = 8 training configurations (here: 50%):
    sample_proportion = 1.0
    sample_size = int(sample_proportion * n_test_params)

    test_param_sample = np.random.choice(
        list(range(n_test_params)), size=int(sample_size), replace=False
    )
    test_params_ = [test_params[i] for i in test_param_sample]
    print("Train configs:", len(test_params_))
    print("CV Iterations:", len(cv_params_) * len(test_params_))

    ### Categorical Variables
    # To leverage LightGBM's ability to handle categorical variables, we need to define them; we'll also `factorize`
    # them so they are both integer-encoded and start at zero (optional, but otherwise throws a warning) as expected
    # by LightGBM:
    categoricals = ["year", "weekday", "month"]
    for feature in categoricals:
        data[feature] = pd.factorize(data[feature], sort=True)[0]

    ### Run Cross-Validation
    # Set up some helper variabels and storage locations to faciliate the CV process and result storage:
    labels = sorted(cv_data.filter(like="fwd").columns)
    features = cv_data.columns.difference(labels).tolist()
    label_dict = dict(zip(lookaheads, labels))

    cv_store = Path(results_path / "parameter_tuning.h5")

    ic_cols = ["bagging_fraction", "feature_fraction", "min_data_in_leaf", "t"] + [
        str(n) for n in num_iterations
    ]

    # Now we take the following steps:
    # - we iterate over the prediction horizons and train/test period length,
    # - set up the `MultipleTimeSeriesCV` accordingly
    # - create the binary LightGBM dataset with the appropriate target, and
    # - iterate over the model hyperparamters to train and validate the model while capturing the relevant performance metrics:

    for lookahead in lookaheads:
        for train_length, test_length in test_params_:
            n_splits = int(2 * YEAR / test_length)
            print(
                f"Lookahead: {lookahead:2.0f} | Train: {train_length:3.0f} | "
                f"Test: {test_length:2.0f} | Params: {len(cv_params_):3.0f}"
            )

            cv = MultipleTimeSeriesCV(
                n_splits=n_splits,
                test_period_length=test_length,
                train_period_length=train_length,
                lookahead=lookahead,
            )

            label = label_dict[lookahead]
            outcome_data = data.loc[:, features + [label]].dropna()

            lgb_data = lgb.Dataset(
                data=outcome_data.drop(label, axis=1),
                label=outcome_data[label],
                categorical_feature=categoricals,
                free_raw_data=False,
            )
            predictions, daily_ic, ic, feature_importance = [], [], [], []
            key = f"{lookahead}/{train_length}/{test_length}"
            T = 0
            for p, (bagging_fraction, feature_fraction, min_data_in_leaf) in enumerate(cv_params_):
                params = base_params.copy()
                params.update(
                    dict(
                        bagging_fraction=bagging_fraction,
                        feature_fraction=feature_fraction,
                        min_data_in_leaf=min_data_in_leaf,
                    )
                )

                start = time()
                cv_preds, nrounds = [], []
                for i, (train_idx, test_idx) in enumerate(cv.split(X=outcome_data)):
                    lgb_train = lgb_data.subset(train_idx.tolist()).construct()
                    lgb_test = lgb_data.subset(test_idx.tolist()).construct()

                    model = lgb.train(
                        params=params,
                        train_set=lgb_train,
                        num_boost_round=num_boost_round,
                        verbose_eval=False,
                    )
                    if i == 0:
                        fi = get_fi(model).to_frame()
                    else:
                        fi[i] = get_fi(model)

                    test_set = outcome_data.iloc[test_idx, :]
                    X_test = test_set.loc[:, model.feature_name()]
                    y_test = test_set.loc[:, label]
                    y_pred = {
                        str(n): model.predict(X_test, num_iteration=n) for n in num_iterations
                    }
                    cv_preds.append(y_test.to_frame("y_test").assign(**y_pred).assign(i=i))
                    nrounds.append(model.best_iteration)
                feature_importance.append(
                    fi.T.describe().T.assign(
                        bagging_fraction=bagging_fraction,
                        feature_fraction=feature_fraction,
                        min_data_in_leaf=min_data_in_leaf,
                    )
                )
                cv_preds = pd.concat(cv_preds).assign(
                    bagging_fraction=bagging_fraction,
                    feature_fraction=feature_fraction,
                    min_data_in_leaf=min_data_in_leaf,
                )

                predictions.append(cv_preds)
                by_day = cv_preds.groupby(level="date")
                ic_by_day = pd.concat(
                    [
                        by_day.apply(lambda x: spearmanr(x.y_test, x[str(n)])[0]).to_frame(n)
                        for n in num_iterations
                    ],
                    axis=1,
                )

                daily_ic.append(
                    ic_by_day.assign(
                        bagging_fraction=bagging_fraction,
                        feature_fraction=feature_fraction,
                        min_data_in_leaf=min_data_in_leaf,
                    )
                )

                cv_ic = [spearmanr(cv_preds.y_test, cv_preds[str(n)])[0] for n in num_iterations]

                T += time() - start
                ic.append([bagging_fraction, feature_fraction, min_data_in_leaf, lookahead] + cv_ic)

                msg = f"{p:3.0f} | {format_time(T)} | "
                msg += f"{bagging_fraction:3.0%} | {feature_fraction:3.0%} | {min_data_in_leaf:5,.0f} | "
                msg += f"{max(cv_ic):6.2%} | {ic_by_day.mean().max(): 6.2%} | {ic_by_day.median().max(): 6.2%}"
                print(msg)

            m = pd.DataFrame(ic, columns=ic_cols)
            m.to_hdf(cv_store, "ic/" + key)
            pd.concat(daily_ic).to_hdf(cv_store, "daily_ic/" + key)
            pd.concat(feature_importance).to_hdf(cv_store, "fi/" + key)
            pd.concat(predictions).to_hdf(cv_store, "predictions/" + key)

    # ## Analyse Cross-Validation Results

    # ### Collect Data

    # We'll now combine the CV results that we stored separately for each fold (to avoid loosing results in case something goes wrong along the way):

    # In[45]:

    id_vars = [
        "train_length",
        "test_length",
        "bagging_fraction",
        "feature_fraction",
        "min_data_in_leaf",
        "t",
        "date",
    ]

    # We'll look at the financial performance in the notebook `alphalens_signal_quality`.

    # In[46]:

    daily_ic, ic = [], []
    for t in lookaheads:
        print(t)
        with pd.HDFStore(cv_store) as store:
            keys = [k[1:] for k in store.keys() if k.startswith(f"/fi/{t}")]
            for key in keys:
                train_length, test_length = key.split("/")[2:]
                print(train_length, test_length)
                k = f"{t}/{train_length}/{test_length}"
                cols = {"t": t, "train_length": int(train_length), "test_length": int(test_length)}

                ic.append(
                    pd.melt(
                        store["ic/" + k].assign(**cols),
                        id_vars=id_vars[:-1],
                        value_name="ic",
                        var_name="rounds",
                    ).apply(pd.to_numeric)
                )

                df = store["daily_ic/" + k].assign(**cols).reset_index()
                daily_ic.append(
                    pd.melt(df, id_vars=id_vars, value_name="daily_ic", var_name="rounds")
                    .set_index("date")
                    .apply(pd.to_numeric)
                    .reset_index()
                )
    ic = pd.concat(ic, ignore_index=True)
    daily_ic = pd.concat(daily_ic, ignore_index=True)

    # ### Predictive Performance: CV Information Coefficient by Day

    # We first look at the daily IC, the metric we ultimately care about for a daily trading strategy. The best results for all prediction horizons are typically achieved with three years of training; the shorter horizons work better with 21 day testing period length. More regularization often improves the result but the impact of the bagging and feature fraction parameters are a little less clear cut and likely depend on other parameters.

    # In[47]:

    group_cols = [
        "t",
        "train_length",
        "test_length",
        "bagging_fraction",
        "feature_fraction",
        "min_data_in_leaf",
    ]
    daily_ic_avg = (
        daily_ic.groupby(group_cols + ["rounds"]).daily_ic.mean().to_frame("ic").reset_index()
    )
    daily_ic_avg.groupby("t", group_keys=False).apply(lambda x: x.nlargest(3, "ic"))

    # In[48]:

    daily_ic_avg.info(show_counts=True)

    # For a 1-day forecast horizon, over 75% of the predictions yield a positive daily IC; the same is true for 21 days which, unsurprisingly, also shows a wider range.

    # In[49]:

    ax = sns.boxenplot(x="t", y="ic", data=daily_ic_avg)
    ax.axhline(0, ls="--", lw=1, c="k")

    # In[50]:

    g = sns.catplot(
        x="t",
        y="ic",
        col="train_length",
        row="test_length",
        data=daily_ic_avg[(daily_ic_avg.test_length == 21)],
        kind="boxen",
    )
    g.savefig(results_path / "daily_ic_test_21", dpi=300)

    # ### HyperParameter Impact: Linear Regression

    # To get a better idea of how the various CV parameters impact the forecast quality, we can run a linear regression with the daily IC as outcome and the one-hot encoded hyperparameters as inputs:

    # In[51]:

    lin_reg = {}
    for t in [1, 5]:
        df_ = daily_ic_avg[(daily_ic_avg.t == t) & (daily_ic_avg.rounds <= 250)].dropna()
        y, X = df_.ic, df_.drop(["ic", "t"], axis=1)
        X = sm.add_constant(pd.get_dummies(X, columns=X.columns, drop_first=True))
        model = sm.OLS(endog=y, exog=X)
        lin_reg[t] = model.fit()
        s = lin_reg[t].summary()
        coefs = pd.read_csv(StringIO(s.tables[1].as_csv())).rename(columns=lambda x: x.strip())
        coefs.columns = ["variable", "coef", "std_err", "t", "p_value", "ci_low", "ci_high"]
        coefs.to_csv(results_path / f"lr_result_{t:02}.csv", index=False)

    # In[52]:

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
        coefs.variable = coefs.variable.str.split("_").str[-1]

        coefs.plot(
            x="variable",
            y="coef",
            kind="bar",
            ax=ax,
            color="none",
            capsize=3,
            yerr="error",
            legend=False,
            rot=0,
        )
        ax.set_ylabel("IC")
        ax.set_xlabel("")
        ax.scatter(x=pd.np.arange(len(coefs)), marker="_", s=120, y=coefs["coef"], color="black")
        ax.axhline(y=0, linestyle="--", color="black", linewidth=1)
        ax.xaxis.set_ticks_position("none")

        ax.annotate(
            "Train\nLength",
            xy=(0.09, -0.1),
            xytext=(0.09, -0.2),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", ec="black"),
            arrowprops=dict(arrowstyle="-[, widthB=5, lengthB=0.8", lw=1.0, color="black"),
        )

        ax.annotate(
            "Test\nLength",
            xy=(0.23, -0.1),
            xytext=(0.23, -0.2),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", ec="black"),
            arrowprops=dict(arrowstyle="-[, widthB=2, lengthB=0.8", lw=1.0, color="black"),
        )

        ax.annotate(
            "Bagging\nFraction",
            xy=(0.32, -0.1),
            xytext=(0.32, -0.2),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", ec="black"),
            arrowprops=dict(arrowstyle="-[, widthB=2.7, lengthB=0.8", lw=1.0, color="black"),
        )

        ax.annotate(
            "Feature\nFraction",
            xy=(0.44, -0.1),
            xytext=(0.44, -0.2),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", ec="black"),
            arrowprops=dict(arrowstyle="-[, widthB=3.4, lengthB=1.0", lw=1.0, color="black"),
        )

        ax.annotate(
            "Min.\nSamples",
            xy=(0.55, -0.1),
            xytext=(0.55, -0.2),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", ec="black"),
            arrowprops=dict(arrowstyle="-[, widthB=2.5, lengthB=1.0", lw=1.0, color="black"),
        )

        ax.annotate(
            "Number of\nRounds",
            xy=(0.8, -0.1),
            xytext=(0.8, -0.2),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", ec="black"),
            arrowprops=dict(arrowstyle="-[, widthB=11.2, lengthB=1.0", lw=1.0, color="black"),
        )

    # The below plot shows the regression coefficient values and their confidence intervals. The intercept (not shown) has a small positive value and is statistically signifant; it captures the impact of the dropped categories (the smallest value for each parameter).
    #
    # For 1-day forecasts, some but not all results are insightful: 21-day testing is better, and so is `min_samples_leaf` of 500 or 1,000. 100-200 trees seem to work best, but both shorter and longer training periods are better than intermediate values.

    # In[53]:

    with sns.axes_style("white"):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        axes = axes.flatten()
        for i, t in enumerate([1, 5]):
            visualize_lr_result(lin_reg[t], axes[i])
            axes[i].set_title(f"Lookahead: {t} Day(s)")
        fig.suptitle("OLS Coefficients & Confidence Intervals", fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)

    # ### Information Coefficient: Overall

    # We'll also take a look at the overall IC value, which is often reported but does not necessarily match the goal of a daily trading strategy that uses the model return predictions as well as the daily IC.

    # In[54]:

    ic.info()

    # #### Best Parameters

    # Directionally, and for shorter periods, similar hyperparameter settings work best (while the IC values are higher):

    # In[55]:

    ic.groupby("t").apply(lambda x: x.nlargest(3, "ic"))

    # #### Visualiztion

    # In[56]:

    g = sns.catplot(
        x="t",
        y="ic",
        col="train_length",
        row="test_length",
        data=ic[(ic.test_length == 21) & (ic.t < 21)],
        kind="box",
    )

    # In[57]:

    t = 1
    train_length = 756
    test_length = 21
    g = sns.catplot(
        x="rounds",
        y="ic",
        col="feature_fraction",
        hue="bagging_fraction",
        row="min_data_in_leaf",
        data=ic[(ic.t == t) & (ic.train_length == train_length) & (ic.test_length == test_length)],
        kind="swarm",
    )

    # ### Random Forest vs Linear Regression

    # Let's compare the best-performing (in-sample) random forest models to our linear regression baseline:

    # In[59]:

    lr_metrics = pd.read_csv(results_path / "lin_reg_performance.csv")
    lr_metrics.info()

    # In[60]:

    daily_ic_avg.info()

    # The results are mixed: for the shortest and longest horizons, the random forest outperforms (slightly for 1 day), while linear regression is competitive for the intermediate horizons:

    # In[61]:

    with sns.axes_style("white"):
        ax = (
            ic.groupby("t")
            .ic.max()
            .to_frame("Random Forest")
            .join(lr_metrics.groupby("lookahead").ic.max().to_frame("Linear Regression"))
            .plot.barh()
        )
        ax.set_ylabel("Lookahead")
        ax.set_xlabel("Information Coefficient")
        sns.despine()
        plt.tight_layout()

    # ## Generate predictions

    # To build and evaluate a trading strategy, we create predictions for the 2018-19 period using the 10 best models that we then ensemble:

    # In[62]:

    param_cols = [
        "train_length",
        "test_length",
        "bagging_fraction",
        "feature_fraction",
        "min_data_in_leaf",
        "rounds",
    ]

    # In[63]:

    def get_params(data, t=5, best=0):
        df = data[data.t == t].sort_values("ic", ascending=False).iloc[best]
        df = df.loc[param_cols]
        rounds = int(df.rounds)
        params = pd.to_numeric(df.drop("rounds"))
        return params, rounds

    # In[64]:

    base_params = dict(boosting_type="rf", objective="regression", bagging_freq=1, verbose=-1)

    store = Path(results_path / "predictions.h5")

    # In[81]:

    for lookahead in [1, 5, 10, 21]:
        if lookahead > 1:
            continue
        print(f"\nLookahead: {lookahead:02}")
        data = pd.read_hdf("data.h5", "stooq/japan/equities")
        labels = sorted(data.filter(like="fwd").columns)
        features = data.columns.difference(labels).tolist()
        label = f"fwd_ret_{lookahead:02}"
        data = data.loc[:, features + [label]].dropna()

        categoricals = ["year", "weekday", "month"]
        for feature in categoricals:
            data[feature] = pd.factorize(data[feature], sort=True)[0]

        lgb_data = lgb.Dataset(
            data=data[features],
            label=data[label],
            categorical_feature=categoricals,
            free_raw_data=False,
        )

        for position in range(10):
            params, num_boost_round = get_params(daily_ic_avg, t=lookahead, best=position)
            params = params.to_dict()
            params["min_data_in_leaf"] = int(params["min_data_in_leaf"])
            train_length = int(params.pop("train_length"))
            test_length = int(params.pop("test_length"))
            params.update(base_params)

            print(f"\tPosition: {position:02}")

            n_splits = int(2 * YEAR / test_length)
            cv = MultipleTimeSeriesCV(
                n_splits=n_splits,
                test_period_length=test_length,
                lookahead=lookahead,
                train_period_length=train_length,
            )

            predictions = []
            start = time()
            for i, (train_idx, test_idx) in enumerate(cv.split(X=data), 1):
                train_set = lgb_data.subset(
                    used_indices=train_idx.tolist(), params=params
                ).construct()

                model = lgb.train(
                    params=params,
                    train_set=train_set,
                    num_boost_round=num_boost_round,
                    verbose_eval=False,
                )

                test_set = data.iloc[test_idx, :]
                y_test = test_set.loc[:, label].to_frame("y_test")
                y_pred = model.predict(test_set.loc[:, model.feature_name()])
                predictions.append(y_test.assign(prediction=y_pred))

            if position == 0:
                test_predictions = pd.concat(predictions).rename(columns={"prediction": position})
            else:
                test_predictions[position] = pd.concat(predictions).prediction

        by_day = test_predictions.groupby(level="date")
        for position in range(10):
            if position == 0:
                ic_by_day = by_day.apply(lambda x: spearmanr(x.y_test, x[position])[0]).to_frame()
            else:
                ic_by_day[position] = by_day.apply(lambda x: spearmanr(x.y_test, x[position])[0])

        test_predictions.to_hdf(store, f"test/{lookahead:02}")

    # In[ ]:
