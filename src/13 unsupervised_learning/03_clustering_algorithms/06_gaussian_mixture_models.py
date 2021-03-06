# Gaussian Mixture Models
# Gaussian mixture models (GMM) are a generative model that assumes the data has been generated by a mix of various
# multivariate normal distributions. The algorithm aims to estimate the mean & covariance matrices of these distributions.
#
# It generalizes the k-Means algorithm: it adds covariance among features so that clusters can be ellipsoids rather than
# spheres, while the centroids are represented by the means of each distribution. The GMM algorithm performs soft
# assignments because each point has a probability to be a member of any cluster.

## The Expectation-Maximization Algorithm
# Expectation-Maximization Algorithm
#
# GMM uses the expectation-maximization algorithm to identify the components of the mixture of Gaussian distributions.
# The goal is to learn the probability distribution parameters from unlabeled data.
#
# The algorithm proceeds iteratively as follows:
# 1. Initialization: Assume random centroids (e.g. from K-Means)
# 2. Repeat until convergence (changes in assignments drop below threshold):
#     1. Expectation Step: Soft assignment - compute probabilities for each point from each distribution
#     2. Maximization Step: Adjust normal-distribution parameters to make data points most likely

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from numpy import atleast_2d
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap

cmap = ListedColormap(sns.xkcd_palette(["denim blue", "medium green", "pale red"]))
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 18
warnings.filterwarnings("ignore")

## Load Iris Data
iris = load_iris()
iris.keys()
features = iris.feature_names
data = pd.DataFrame(data=np.column_stack([iris.data, iris.target]), columns=features + ["label"])
data.label = data.label.astype(int)
data.info()
scaler = StandardScaler()
features_standardized = scaler.fit_transform(data[features])
n = len(data)

pca = PCA(n_components=2)
features_2D = pca.fit_transform(features_standardized)

ev1, ev2 = pca.explained_variance_ratio_
ax = plt.figure(figsize=(10, 6)).gca(
    title="2D Projection",
    xlabel=f"Explained Variance: {ev1:.2%}",
    ylabel=f"Explained Variance: {ev2:.2%}",
)
ax.scatter(*features_2D.T, c=data.label, s=15, cmap=cmap)
ax.set_xticklabels([])
ax.set_xticks([])
plt.savefig("images/06-01.png")

## Perform GMM clustering
n_components = 3
gmm = GaussianMixture(n_components=n_components)
gmm.fit(features_standardized)

data["clusters"] = gmm.predict(features_standardized)
labels, clusters = data.label, data.clusters
mi = adjusted_mutual_info_score(labels, clusters)

fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
axes[0].scatter(*features_2D.T, c=data.label, s=25, cmap=cmap)
axes[0].set_title("Original Data")
axes[1].scatter(*features_2D.T, c=data.clusters, s=25, cmap=cmap)
axes[1].set_title("Clusters | MI={:.2f}".format(mi))
for ax in axes:
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
plt.savefig("images/06-02.png")

### Visualize Gaussian Distributions
# The following figures show the GMM cluster membership probabilities for the iris dataset as contour lines:
xmin, ymin = features_2D.min(axis=0)
xmax, ymax = features_2D.max(axis=0)

x = np.linspace(xmin, xmax, 500)
y = np.linspace(ymin, ymax, 500)
X, Y = np.meshgrid(x, y)

simulated_2D = np.column_stack([np.ravel(X), np.ravel(Y)])
simulated_4D = pca.inverse_transform(simulated_2D)
Z = atleast_2d(np.clip(np.exp(gmm.score_samples(simulated_4D)), a_min=0, a_max=1)).reshape(X.shape)

fig, ax = plt.subplots(figsize=(10, 6))
CS = ax.contour(X, Y, Z, cmap="RdBu_r", alpha=0.8)
CB = plt.colorbar(CS, shrink=0.8)
ax.scatter(*features_2D.T, c=data.label, s=25, cmap=cmap)
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig("images/06-03.png")

fig = plt.figure(figsize=(10, 8))
ax = fig.gca(projection="3d")
CS = ax.contourf3D(X, Y, Z, cmap="RdBu_r", alpha=0.5)
CB = plt.colorbar(CS, shrink=0.8)
ax.scatter(*features_2D.T, c=data.label, s=25, cmap=cmap)
plt.tight_layout()
plt.savefig("images/06-04.png")

### Bayesian Information Criterion
# We are looking for the minimum value, so two clusters would be the preferred solution; with three as the close
# runner-up (varies depending on random sample).

bic = {}
for n_components in range(2, 8):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(features_standardized)
    bic[n_components] = gmm.bic(features_standardized)
print(pd.Series(bic))
