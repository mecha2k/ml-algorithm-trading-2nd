# CNN for Trading - Part 2: From Time-Series Features to Clustered Images
# To exploit the grid-like structure of time-series data, we can use CNN architectures for univariate and multivariate
# time series. In the latter case, we consider different time series as channels, similar to the different color signals.
#
# An alternative approach converts a time series of alpha factors into a two-dimensional format to leverage the ability
# of CNNs to detect local patterns. [Sezer and Ozbayoglu (2018)
# (https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural
# _Networks_Time_Series_to_Image_Conversion_Approach) propose CNN-TA, which computes 15 technical indicators for
# different intervals and uses hierarchical clustering (see Chapter 13, Data-Driven Risk Factors and Asset Allocation
# with Unsupervised Learning) to locate indicators that behave similarly close to each other in a two-dimensional grid.
#
# The authors train a CNN similar to the CIFAR-10 example we used earlier to predict whether to buy, hold, or sell an
# asset on a given day. They compare the CNN performance to "buy-and-hold" and other models and find that it outperforms
# all alternatives using daily price series for Dow 30 stocks and the nine most-traded ETFs over the 2007-2017 time period.
#
# The section on *CNN for Trading* consists of three notebooks that experiment with this approach using daily US equity
# price data. They demonstrate
# 1. How to compute relevant financial features
# 2. How to convert a similar set of indicators into image format and cluster them by similarity
# 3. How to train a CNN to predict daily returns and evaluate a simple long-short strategy based on the resulting signals.

# ## Selecting and Clustering Features

# The next steps that we will tackle in this notebook are
# 1. Select the 15 most relevant features from the 20 candidates to fill the 15×15 input grid.
# 2. Apply hierarchical clustering to identify features that behave similarly and order the columns and the rows of
#    the grid accordingly.

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

import matplotlib.pyplot as plt
import seaborn as sns
import warnings


idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")

DATA_STORE = "../data/assets.h5"

results_path = Path("../data/ch18", "cnn_for_trading")
mnist_path = results_path / "mnist"
if not mnist_path.exists():
    mnist_path.mkdir(parents=True)

if __name__ == "__main__":
    MONTH = 21
    YEAR = 12 * MONTH

    START = "2000-01-01"
    END = "2017-12-31"

    ## Load Model Data
    with pd.HDFStore("../data/18_data.h5") as store:
        features = store.get("features")
        targets = store.get("targets")
    features.info()
    targets.info()

    ## Select Features using Mutual Information
    # To this end, we estimate the mutual information for each indicator and the 15 intervals with respect to our target,
    # the one-day forward returns. As discussed in Chapter 4, Financial Feature Engineering – How to Research Alpha
    # Factors, scikit-learn provides the `mutual_info_regression()` function that makes this straightforward, albeit
    # time-consuming and memory-intensive.
    # To accelerate the process, we randomly sample 100,000 observations:

    mi = {}
    for t in tqdm([1, 5]):
        target = f"r{t:02}_fwd"
        # sample a smaller number to speed up the computation
        df = features.join(targets[target]).dropna().sample(n=100000)
        X = df.drop(target, axis=1)
        y = df[target]
        mi[t] = pd.Series(mutual_info_regression(X=X, y=y), index=X.columns).sort_values(
            ascending=False
        )

    mutual_info = pd.DataFrame(mi)
    mutual_info.to_hdf("../data/18_data.h5", "mutual_info")
    mutual_info = pd.read_hdf("../data/18_data.h5", "mutual_info")

    mi_by_indicator = (
        mutual_info.groupby(mutual_info.index.to_series().str.split("_").str[-1])
        .mean()
        .rank(ascending=False)
        .sort_values(by=1)
    )
    mutual_info.boxplot()
    sns.despine()
    plt.savefig("images/06-01.png")

    # The below figure shows the mutual information, averaged across the 15 intervals for each indicator. NATR, PPO,
    # and Bollinger Bands are most important from this metric's perspective:
    (
        mutual_info.groupby(mutual_info.index.to_series().str.split("_").str[-1])[1]
        .mean()
        .sort_values()
        .plot.barh(title="Mutual Information with 1-Day Forward Returns")
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig("images/06_mutual_info_cnn_features.png", dpi=300)

    best_features = mi_by_indicator.head(15).index
    size = len(best_features)

    ## Hierarchical Feature Clustering
    features = pd.concat([features.filter(like=f"_{f}") for f in best_features], axis=1)

    new_cols = {}
    for feature in best_features:
        fnames = sorted(features.filter(like=f"_{feature}").columns.tolist())
        renamed = [f"{i:02}_{feature}" for i in range(1, len(fnames) + 1)]
        new_cols.update(dict(zip(fnames, renamed)))
    features = features.rename(columns=new_cols).sort_index(1)
    features.info()

    ## Hierarchical Clustering
    # As discussed in the first section of this chapter, CNNs rely on the locality of relevant patterns that is
    # typically found in images where nearby pixels are closely related and changes from one pixel to the next are
    # often gradual.
    #
    # To organize our indicators in a similar fashion, we will follow Sezer and Ozbayoglu's approach of applying
    # hierarchical clustering. The goal is to identify features that behave similarly and order the columns and the
    # rows of the grid accordingly.
    #
    # We can build on SciPy's `pairwise_distance()`, `linkage()`, and `dendrogram()` functions that we introduced
    # in *Chapter 13, Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning* alongside other forms
    # of clustering.
    #
    # We create a helper function that standardizes the input column-wise to avoid distorting distances among features
    # due to differences in scale, and use the Ward criterion that merges clusters to minimize variance. The function
    # returns the order of the leaf nodes in the dendrogram that in turn displays the successive formation of larger clusters:

    def cluster_features(data, labels, ax, title):
        data = StandardScaler().fit_transform(data)
        pairwise_distance = pdist(data)
        Z = linkage(data, "ward")
        c, coph_dists = cophenet(Z, pairwise_distance)
        dend = dendrogram(
            Z, labels=labels, orientation="top", leaf_rotation=0.0, leaf_font_size=8.0, ax=ax
        )
        ax.set_title(title)
        return dend["ivl"]

    # To obtain the optimized order of technical indicators in the columns and the different intervals in the rows,
    # we use NumPy's `.reshape()` method to ensure that the dimension we would like to cluster appears in the columns
    # of the two-dimensional array we pass to `cluster_features()`.
    fig, axes = plt.subplots(figsize=(15, 4), ncols=2)

    labels = sorted(best_features)
    title = "Column Features: Indicators"
    col_order = cluster_features(features.dropna().values.reshape(-1, 15).T, labels, axes[0], title)

    labels = list(range(1, 16))
    title = "Row Features: Indicator Parameters"
    row_order = cluster_features(
        features.dropna().values.reshape(-1, 15, 15).transpose((0, 2, 1)).reshape(-1, 15).T,
        labels,
        axes[1],
        title,
    )
    axes[0].set_xlabel("Indicators")
    axes[1].set_xlabel("Parameters")
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/06_cnn_clustering.png", dpi=300)

    # We reorder the features accordingly and store the result as inputs for the CNN that we will create in the next step.
    feature_order = [f"{i:02}_{j}" for i in row_order for j in col_order]
    features = features.loc[:, feature_order]
    features = features.apply(pd.to_numeric, downcast="float")
    features.info()
    features.to_hdf("../data/18_data.h5", "img_data")
