from pathlib import Path
import os, sys
import numpy as np

from numpy.random import choice
import pandas as pd
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
pd.options.display.float_format = "{:,.2f}".format

np.random.seed(seed=42)

results_path = Path("../data/ch11", "random_forest")
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
    with pd.HDFStore("../data/11_data.h5") as store:
        data = store["us/equities/monthly"]
    data.info()

    y = data.target
    y_binary = (y > 0).astype(int)
    X = pd.get_dummies(data.drop("target", axis=1))

    ## Random Forests
    # ### Cross-validation parameters
    n_splits = 10
    train_period_length = 60
    test_period_length = 6
    lookahead = 1

    cv = MultipleTimeSeriesCV(
        n_splits=n_splits,
        train_period_length=train_period_length,
        test_period_length=test_period_length,
        lookahead=lookahead,
    )

    ### Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,  # default changed from 10 to 100 in version 0.22
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    #### Cross-Validation with default settings
    cv_score = cross_val_score(
        estimator=rf_clf, X=X, y=y_binary, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1
    )
    print(np.mean(cv_score))

    ### Regression RF
    def rank_correl(y, y_pred):
        return spearmanr(y, y_pred)[0]

    ic = make_scorer(rank_correl)

    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=None,
        verbose=0,
        warm_start=False,
    )

    cv_score = cross_val_score(estimator=rf_reg, X=X, y=y, scoring=ic, cv=cv, n_jobs=-1, verbose=1)
    print(np.mean(cv_score))

    ## Parameter Tuning
    # The key configuration parameters include the various hyperparameters for the individual decision trees introduced
    # in the notebook [decision_trees](01_decision_trees.ipynb).
    # The following tables lists additional options for the two `RandomForest` classes:

    # | Keyword      | Default | Description                                                                                                                |
    # |--------------|---------|----------------------------------------------------------------------------------------------------------------------------|
    # | bootstrap    | True    | Bootstrap samples during training                                                                                          |
    # | n_estimators | 10      | # trees in the forest.                                                                                                     |
    # | oob_score    | False   | Use out-of-bag samples to estimate the R2 on unseen data                                                                   |
    # | warm_start   | False   | Reuse result of previous call to continue training and add more trees to the ensemble, otherwise, train a whole new forest |

    # - The `bootstrap` parameter activates in the preceding bagging algorithm outline, which in turn enables the
    #   computation of the out-of-bag score (oob_score) that estimates the generalization accuracy using samples not
    #   included in the bootstrap sample used to train a given tree (see next section for detail).
    # - The `n_estimators` parameter defines the number of trees to be grown as part of the forest. Larger forests
    #   perform better, but also take more time to build. It is important to monitor the cross-validation error as a
    #   function of the number of base learners to identify when the marginal reduction of the prediction error declines
    #   and the cost of additional training begins to outweigh the benefits.
    # - The `max_features` parameter controls the size of the randomly selected feature subsets available when learning
    #   a new decision rule and split a node. A lower value reduces the correlation of the trees and, thus, the
    #   ensemble's variance, but may also increase the bias. Good starting values are `n_features` (the number of
    #   training features) for regression problems and `sqrt(n_features)` for classification problems, but will depend
    #   on the relationships among features and should be optimized using cross-validation.

    # Random forests are designed to contain deep fully-grown trees, which can be created using `max_depth=None` and
    # `min_samples_split=2`. However, these values are not necessarily optimal, especially for high-dimensional data
    # with many samples and, consequently, potentially very deep trees that can become very computationally-, and
    # memory-, intensive.
    # The `RandomForest` class provided by sklearn support parallel training and prediction by setting the n_jobs
    # parameter to the k number of jobs to run on different cores. The -1 value uses all available cores. The overhead
    # of interprocess communication may limit the speedup from being linear so that k jobs may take more than 1/k the
    # time of a single job. Nonetheless, the speedup is often quite significant for large forests or deep individual
    # trees that may take a meaningful amount of time to train when the data is large, and split evaluation becomes
    # costly. As always, the best parameter configuration should be identified using cross-validation. The following
    # steps illustrate the process:

    ### Define Parameter Grid
    param_grid = {
        "n_estimators": [50, 100, 250],
        "max_depth": [5, 15, None],
        "min_samples_leaf": [5, 25, 100],
    }

    ### Instantiate GridSearchCV
    # We will use 10-fold custom cross-validation and populate the parameter grid with values for the key configuration
    # settings:
    gridsearch_clf = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        refit=True,
        return_train_score=True,
        verbose=1,
    )

    ### Fit Classifier
    gridsearch_clf.fit(X=X, y=y_binary)

    #### Persist Result
    joblib.dump(gridsearch_clf, results_path / "gridsearch_clf.joblib")

    gridsearch_clf = joblib.load(results_path / "gridsearch_clf.joblib")
    print(gridsearch_clf.best_params_)
    print(gridsearch_clf.best_score_)

    #### Feature Importance
    # A random forest ensemble may contain hundreds of individual trees, but it is still possible to obtain an overall
    # summary measure of feature importance from bagged models.
    # For a given feature, the importance score is the total reduction in the objective function's value, which results
    # from splits based on this feature, averaged over all trees. Since the objective function takes into account how
    # many features are affected by a split, this measure is implicitly a weighted average so that features used near
    # the top of a tree will get higher scores due to the larger number of observations contained in the much smaller
    # number of available nodes. By averaging over many trees grown in a randomized fashion, the feature importance
    # estimate loses some variance and becomes more accurate.
    # The computation differs for classification and regression trees based on the different objectives used to learn
    # the decision rules and is measured in terms of the mean square error for regression trees and the Gini index or
    # entropy for classification trees.
    # `sklearn` further normalizes the feature-importance measure so that it sums up to 1. Feature importance thus
    # computed is also used for feature selection as an alternative to the mutual information measures we saw in
    # Chapter 6, The Machine Learning Process (see SelectFromModel in the sklearn.feature_selection module).
    # In our example, the importance values for the top-20 features are as shown here:

    fig, ax = plt.subplots(figsize=(12, 5))
    (
        pd.Series(gridsearch_clf.best_estimator_.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .iloc[:20]
        .sort_values()
        .plot.barh(ax=ax, title="RF Feature Importance")
    )
    fig.tight_layout()
    plt.savefig("images/03-01.png")

    ### Fit Regressor
    gridsearch_reg = GridSearchCV(
        estimator=rf_reg,
        param_grid=param_grid,
        scoring=ic,
        n_jobs=-1,
        cv=cv,
        refit=True,
        return_train_score=True,
        verbose=1,
    )

    gs_reg = gridsearch_reg
    gridsearch_reg.fit(X=X, y=y)
    joblib.dump(gridsearch_reg, results_path / "rf_reg_gridsearch.joblib")

    gridsearch_reg = joblib.load(results_path / "rf_reg_gridsearch.joblib")
    print(gridsearch_reg.best_params_)
    print(f"{gridsearch_reg.best_score_*100:.2f}")

    ### Compare Results
    #### Best Parameters
    print(
        pd.DataFrame(
            {
                "Regression": pd.Series(gridsearch_reg.best_params_),
                "Classification": pd.Series(gridsearch_clf.best_params_),
            }
        )
    )

    #### Feature Importance
    fi_clf = gridsearch_clf.best_estimator_.feature_importances_
    fi_reg = gridsearch_reg.best_estimator_.feature_importances_

    idx = [c.replace("_", " ").upper() for c in X.columns]

    fig, axes = plt.subplots(figsize=(14, 4), ncols=2)
    (
        pd.Series(fi_clf, index=idx)
        .sort_values(ascending=False)
        .iloc[:15]
        .sort_values()
        .plot.barh(ax=axes[1], title="Classifier")
    )
    (
        pd.Series(fi_reg, index=idx)
        .sort_values(ascending=False)
        .iloc[:15]
        .sort_values()
        .plot.barh(ax=axes[0], title="Regression")
    )
    fig.tight_layout()
    plt.savefig("images/03-02.png")
