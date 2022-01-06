import pandas as pd
import numpy as np
import sys, os

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from time import time

import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
pd.options.display.float_format = "{:,.2f}".format


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


YEAR = 252
sys.path.insert(1, os.path.join(sys.path[0], ".."))

if __name__ == "__main__":
    with pd.HDFStore("../data/data.h5") as store:
        data = store["model_data"].dropna().drop(["open", "close", "low", "high"], axis=1)
    data = data.drop([c for c in data.columns if "year" in c or "lag" in c], axis=1)

    ### Select Investment Universe
    data = data[data.dollar_vol_rank < 100]

    ### Create Model Data
    y = data.filter(like="target").copy()
    X = data.drop(y.columns, axis=1)
    X = X.drop(["dollar_vol", "dollar_vol_rank", "volume", "consumer_durables"], axis=1).copy()

    ## Logistic Regression
    ### Define cross-validation parameters
    # train_period_length = 63
    # test_period_length = 10
    # lookahead = 1
    # n_splits = int(3 * YEAR / test_period_length)
    #
    # cv = MultipleTimeSeriesCV(
    #     n_splits=n_splits,
    #     test_period_length=test_period_length,
    #     lookahead=lookahead,
    #     train_period_length=train_period_length,
    # )
    #
    # target = f"target_{lookahead}d"
    # y.loc[:, "label"] = (y[target] > 0).astype(int)
    # print(y.label.value_counts())
    #
    # Cs = np.logspace(-5, 5, 11)
    # cols = ["C", "date", "auc", "ic", "pval"]

    ### Run cross-validation
    # log_coeffs, log_scores, log_predictions = {}, [], []
    # for C in Cs:
    #     print(C)
    #     model = LogisticRegression(C=C, fit_intercept=True, random_state=42, n_jobs=-1)
    #     pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    #     ics = aucs = 0
    #     start = time()
    #     coeffs = []
    #     for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    #         X_train, y_train, = (
    #             X.iloc[train_idx],
    #             y.label.iloc[train_idx],
    #         )
    #         pipe.fit(X=X_train, y=y_train)
    #         X_test, y_test = X.iloc[test_idx], y.label.iloc[test_idx]
    #         actuals = y[target].iloc[test_idx]
    #         if len(y_test) < 10 or len(np.unique(y_test)) < 2:
    #             continue
    #         y_score = pipe.predict_proba(X_test)[:, 1]
    #
    #         auc = roc_auc_score(y_score=y_score, y_true=y_test)
    #         actuals = y[target].iloc[test_idx]
    #         ic, pval = spearmanr(y_score, actuals)
    #
    #         log_predictions.append(
    #             y_test.to_frame("labels").assign(predicted=y_score, C=C, actuals=actuals)
    #         )
    #         date = y_test.index.get_level_values("date").min()
    #         log_scores.append([C, date, auc, ic * 100, pval])
    #         coeffs.append(pipe.named_steps["model"].coef_)
    #         ics += ic
    #         aucs += auc
    #         if i % 10 == 0:
    #             print(f"\t{time() - start:5.1f} | {i:03} | {ics / i:>7.2%} | {aucs / i:>7.2%}")
    #
    #     log_coeffs[C] = np.mean(coeffs, axis=0).squeeze()

    ### Evaluate Results
    # log_scores = pd.DataFrame(log_scores, columns=cols)
    # log_scores.to_hdf("../data/data.h5", "logistic/scores")
    #
    # log_coeffs = pd.DataFrame(log_coeffs, index=X.columns).T
    # log_coeffs.to_hdf("../data/data.h5", "logistic/coeffs")
    #
    # log_predictions = pd.concat(log_predictions)
    # log_predictions.to_hdf("../data/data.h5", "logistic/predictions")

    log_scores = pd.read_hdf("../data/data.h5", "logistic/scores")
    log_scores.info()
    print(log_scores.groupby("C").auc.describe())

    ### Plot Validation Scores
    def plot_ic_distribution(df, ax=None):
        if ax is not None:
            sns.histplot(data=df, x="ic", ax=ax)
        else:
            ax = sns.distplot(data=df, x="ic")
        mean, median = df.ic.mean(), df.ic.median()
        ax.axvline(0, lw=1, ls="--", c="k")
        ax.text(
            x=0.05,
            y=0.9,
            s=f"Mean: {mean:8.2f}\nMedian: {median:5.2f}",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Information Coefficient")
        plt.tight_layout()

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    sns.lineplot(x="C", y="auc", data=log_scores, estimator=np.mean, label="Mean", ax=axes[0])
    by_alpha = log_scores.groupby("C").auc.agg(["mean", "median"])
    best_auc = by_alpha["mean"].idxmax()
    by_alpha["median"].plot(logx=True, ax=axes[0], label="Median", xlim=(10e-6, 10e5))
    axes[0].axvline(best_auc, ls="--", c="k", lw=1, label="Max. Mean")
    axes[0].axvline(by_alpha["median"].idxmax(), ls="-.", c="k", lw=1, label="Max. Median")
    axes[0].legend()
    axes[0].set_ylabel("AUC")
    axes[0].set_xscale("log")
    axes[0].set_title("Area Under the Curve")

    plot_ic_distribution(log_scores[log_scores.C == best_auc], ax=axes[1])
    axes[1].set_title("Information Coefficient")

    fig.suptitle("Logistic Regression", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig("images/08-01.png", bboxinches="tight")
