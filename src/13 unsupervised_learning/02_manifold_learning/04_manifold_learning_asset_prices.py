import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

sns.set_style("white")
np.random.seed(42)

## Load equity returns
# This example uses daily returns; remove the comment symbols to use weekly returns instead.
idx = pd.IndexSlice
with pd.HDFStore("../../data/assets.h5") as store:
    returns = (
        store["quandl/wiki/prices"]
        .loc[idx["2010":"2018", :], "adj_close"]
        .unstack(level="ticker")
        .pct_change()
    )
returns = returns.dropna(thresh=int(returns.shape[0] * 0.95), axis=1)
returns = returns.dropna(thresh=int(returns.shape[1] * 0.95)).clip(lower=-0.5, upper=0.5)
returns.info()

returns = returns.sample(n=250)
daily_avg = returns.mean(axis=1)
returns = returns.apply(lambda x: x.fillna(daily_avg))

pca = PCA(n_components=2)

## T-Stochastic Neighbor Embedding (TSNE): Parameter Settings
# Perplexity: emphasis on local vs global structure
n_iter = 5000

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()
axes[0].scatter(*pca.fit_transform(returns).T, s=10)
axes[0].set_title("PCA", fontsize=18)
axes[0].axes.get_xaxis().set_visible(False)
axes[0].axes.get_yaxis().set_visible(False)
for i, p in enumerate([2, 5, 7, 10, 15], 1):
    embedding = TSNE(n_components=2, perplexity=p, n_iter=n_iter).fit_transform(returns)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10)
    axes[i].set_title("Perplexity: {:.0f}".format(p), fontsize=14)
    axes[i].axes.get_xaxis().set_visible(False)
    axes[i].axes.get_yaxis().set_visible(False)
fig.suptitle(f"TSNE | Iterations: {n_iter:,.0f}", fontsize=18)
fig.subplots_adjust(top=0.9)
plt.savefig("images/04-01.png", bboxinches="tight")

### Convergence with `n_iter`
perplexity = 5
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()
axes[0].scatter(*pca.fit_transform(returns).T, s=10)
axes[0].set_title("PCA")
axes[0].axes.get_xaxis().set_visible(False)
axes[0].axes.get_yaxis().set_visible(False)

for i, n in enumerate([250, 500, 1000, 2500, 5000], 1):
    embedding = TSNE(perplexity=perplexity, n_iter=n).fit_transform(returns)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10)
    axes[i].set_title("Iterations: {:,.0f}".format(n), fontsize=14)
    axes[i].axes.get_xaxis().set_visible(False)
    axes[i].axes.get_yaxis().set_visible(False)

fig.suptitle(f"TSNE | Perpexity: {perplexity:,.0f}", fontsize=16)
fig.subplots_adjust(top=0.9)
plt.savefig("images/04-02.png", bboxinches="tight")

## Uniform Manifold Approximation and Projection (UMAP): Parameter Settings
## Neighbors
min_dist = 0.1

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()

axes[0].scatter(*pca.fit_transform(returns).T, s=10)
axes[0].set_title("PCA")
axes[0].axes.get_xaxis().set_visible(False)
axes[0].axes.get_yaxis().set_visible(False)

for i, n in enumerate([2, 3, 4, 5, 7], 1):
    embedding = umap.UMAP(n_neighbors=n, min_dist=min_dist).fit_transform(returns)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10)
    axes[i].set_title("Neighbors: {:.0f}".format(n), fontsize=14)
    axes[i].axes.get_xaxis().set_visible(False)
    axes[i].axes.get_yaxis().set_visible(False)

fig.suptitle(f"UMAP | Min. Distance: {min_dist:,.2f}", fontsize=16)
fig.subplots_adjust(top=0.9)
plt.savefig("images/04-03.png", bboxinches="tight")

### Minimum Distance
n_neighbors = 2

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()

axes[0].scatter(*pca.fit_transform(returns).T, s=10)
axes[0].set_title("PCA")
axes[0].axes.get_xaxis().set_visible(False)
axes[0].axes.get_yaxis().set_visible(False)

for i, d in enumerate([0.001, 0.01, 0.1, 0.2, 0.5], 1):
    embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=d).fit_transform(returns)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10)
    axes[i].set_title("Min. Distance: {:.3f}".format(d), fontsize=14)
    axes[i].axes.get_xaxis().set_visible(False)
    axes[i].axes.get_yaxis().set_visible(False)

fig.suptitle(f"UMAP | # Neighbors: {n_neighbors:,.0f}", fontsize=16)
fig.subplots_adjust(top=0.9)
plt.savefig("images/04-04.png", bboxinches="tight")
