# Long-Short Strategy, Part 5: Generating out-of-sample predictions

from time import time
import sys, os
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import lightgbm as lgb
from catboost import Pool, CatBoostRegressor

import matplotlib.pyplot as plt
import seaborn as sns


YEAR = 252
np.random.seed(42)

idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.2f}".format

store = Path("../data/ch12/predictions.h5")
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

    ## Generate LightGBM predictions
    ### Model Configuration
    base_params = dict(boosting="gbdt", objective="regression", verbose=-1)
    categoricals = ["year", "month", "sector", "weekday"]

    lookahead = 1

    ### Get Data
    data = pd.read_hdf("../data/12_data.h5", "model_data").sort_index()
    labels = sorted(data.filter(like="_fwd").columns)
    features = data.columns.difference(labels).tolist()
    label = f"r{lookahead:02}_fwd"
    data = data.loc[idx[:, "2010":], features + [label]].dropna()

    for feature in categoricals:
        data[feature] = pd.factorize(data[feature], sort=True)[0]

    lgb_data = lgb.Dataset(
        data=data[features],
        label=data[label],
        categorical_feature=categoricals,
        free_raw_data=False,
    )

    ### Generate predictions
    lgb_ic = pd.read_hdf(results_path / "model_tuning.h5", "lgb/ic")
    lgb_daily_ic = pd.read_hdf(results_path / "model_tuning.h5", "lgb/daily_ic")

    def get_lgb_params(data, t=5, best=0):
        param_cols = scope_params[1:] + lgb_train_params + ["boost_rounds"]
        df = data[data.lookahead == t].sort_values("ic", ascending=False).iloc[best]
        return df.loc[param_cols]

    for position in range(10):
        params = get_lgb_params(lgb_daily_ic, t=lookahead, best=position)
        params = params.to_dict()

        for p in ["min_data_in_leaf", "num_leaves"]:
            params[p] = int(params[p])
        train_length = int(params.pop("train_length"))
        test_length = int(params.pop("test_length"))
        num_boost_round = int(params.pop("boost_rounds"))
        params.update(base_params)
        print(f"\nPosition: {position:02}")

        # 1-year out-of-sample period
        n_splits = int(YEAR / test_length)
        cv = MultipleTimeSeriesCV(
            n_splits=n_splits,
            test_period_length=test_length,
            lookahead=lookahead,
            train_period_length=train_length,
        )

        predictions = []
        start = time()
        for i, (train_idx, test_idx) in enumerate(cv.split(X=data), 1):
            print(i, end=" ", flush=True)
            lgb_train = lgb_data.subset(used_indices=train_idx.tolist(), params=params).construct()

            model = lgb.train(
                params=params,
                train_set=lgb_train,
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
    print(ic_by_day.describe())
    test_predictions.to_hdf(store, f"lgb/test/{lookahead:02}")

    ## Generate CatBoost predictions
    lookaheads = [1, 5, 21]
    label_dict = dict(zip(lookaheads, labels))

    ### Model Configuration
    data = pd.read_hdf("../data/12_data.h5", "model_data").sort_index()
    labels = sorted(data.filter(like="_fwd").columns)
    features = data.columns.difference(labels).tolist()
    label = f"r{lookahead:02}_fwd"
    data = data.loc[idx[:, "2010":], features + [label]].dropna()

    for feature in categoricals:
        data[feature] = pd.factorize(data[feature], sort=True)[0]

    cat_cols_idx = [data.columns.get_loc(c) for c in categoricals]
    catboost_data = Pool(
        label=data[label], data=data.drop(label, axis=1), cat_features=cat_cols_idx
    )

    ### Generate predictions
    catboost_ic = pd.read_hdf(results_path / "model_tuning.h5", "catboost/ic")
    catboost_ic_avg = pd.read_hdf(results_path / "model_tuning.h5", "catboost/daily_ic")

    def get_cb_params(data, t=5, best=0):
        param_cols = scope_params[1:] + catboost_train_params + ["boost_rounds"]
        df = data[data.lookahead == t].sort_values("ic", ascending=False).iloc[best]
        return df.loc[param_cols]

    for position in range(10):
        params = get_cb_params(catboost_ic_avg, t=lookahead, best=position)
        params = params.to_dict()

        for p in ["max_depth", "min_child_samples"]:
            params[p] = int(params[p])
        train_length = int(params.pop("train_length"))
        test_length = int(params.pop("test_length"))
        num_boost_round = int(params.pop("boost_rounds"))
        params["task_type"] = "GPU"
        print(f"\nPosition: {position:02}")

        # 1-year out-of-sample period
        n_splits = int(YEAR / test_length)
        cv = MultipleTimeSeriesCV(
            n_splits=n_splits,
            test_period_length=test_length,
            lookahead=lookahead,
            train_period_length=train_length,
        )

        predictions = []
        start = time()
        for i, (train_idx, test_idx) in enumerate(cv.split(X=data), 1):
            print(i, end=" ", flush=True)
            train_set = catboost_data.slice(train_idx.tolist())

            model = CatBoostRegressor(**params)
            model.fit(X=train_set, verbose_eval=False)

            test_set = data.iloc[test_idx, :]
            y_test = test_set.loc[:, label].to_frame("y_test")
            y_pred = model.predict(test_set.loc[:, model.feature_names_])
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
    print(ic_by_day.describe())
    test_predictions.to_hdf(store, f"catboost/test/{lookahead:02}")
