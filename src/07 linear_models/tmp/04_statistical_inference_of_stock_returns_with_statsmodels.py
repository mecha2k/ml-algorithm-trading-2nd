import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from statsmodels.api import OLS, add_constant, graphics
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from icecream import ic

sns.set_style("whitegrid")
idx = pd.IndexSlice
warnings.simplefilter(action="ignore", category=FutureWarning)

with pd.HDFStore("../data/data.h5") as store:
    data = store["model_data"].dropna().drop(["open", "close", "low", "high"], axis=1)

data = data[data.dollar_vol_rank < 100]
data.info(show_counts=True)

y = data.filter(like="target")
X = data.drop(y.columns, axis=1)
X = X.drop(["dollar_vol", "dollar_vol_rank", "volume", "consumer_durables"], axis=1)
ic(X.head())
ic(y.head())

sns.clustermap(
    y.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0, annot=True, fmt=".2%"
)
plt.savefig("../images/ch07_im12.png", dpi=300, bboxinches="tight")

sns.clustermap(X.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0)
plt.gcf().set_size_inches((14, 14))
plt.savefig("../images/ch07_im13.png", dpi=300, bboxinches="tight")


corr_mat = X.corr().stack().reset_index()
corr_mat.columns = ["var1", "var2", "corr"]
corr_mat = corr_mat[corr_mat.var1 != corr_mat.var2].sort_values(by="corr", ascending=False)
ic(corr_mat.head().append(corr_mat.tail()))

# y.boxplot()
# plt.savefig("../images/ch07_im14.png", dpi=300, bboxinches="tight")

sectors = X.iloc[:, -10:]
X = (
    X.drop(sectors.columns, axis=1)
    .groupby(level="ticker")
    .transform(lambda x: (x - x.mean()) / x.std())
    .join(sectors)
    .fillna(0)
)
ic(X.head())

target = "target_1d"
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()
print(trained_model.summary())

target = "target_5d"
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()
print(trained_model.summary())

preds = trained_model.predict(add_constant(X))
residuals = y[target] - preds
ic(residuals)

fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
sns.histplot(
    data=residuals,
    # color="red",
    label="Residuals",
    kde=False,
    stat="density",
    # linewidth=1,
    ax=axes[0],
)
x0, x1 = axes[0].get_xlim()
x_pdf = np.linspace(x0, x1, 100)
axes[0].plot(x_pdf, norm.pdf(x_pdf), "r", lw=2, label="pdf")
axes[0].set_title("Residual Distribution")
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title="Residual Autocorrelation")
axes[1].set_xlabel("Lags")
sns.despine()
plt.savefig("../images/ch07_im15.png", dpi=300, bboxinches="tight")
plt.show()

target = "target_10d"
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()
print(trained_model.summary())

target = "target_21d"
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()
print(trained_model.summary())
