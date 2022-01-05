import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from statsmodels.api import OLS, add_constant, graphics
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from icecream import ic


idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.2f}".format


if __name__ == "__main__":
    with pd.HDFStore("../data/data.h5") as store:
        data = store["model_data"].dropna().drop(["open", "close", "low", "high"], axis=1)

    ### Select Investment Universe
    data = data[data.dollar_vol_rank < 100]
    data.info(show_counts=True)

    ### Create Model Data
    y = data.filter(like="target")
    X = data.drop(y.columns, axis=1)
    X = X.drop(["dollar_vol", "dollar_vol_rank", "volume", "consumer_durables"], axis=1)

    ## Explore Data
    plt.figure(figsize=(10, 6))
    sns.clustermap(
        y.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0, annot=True, fmt=".2%"
    )
    plt.tight_layout()
    plt.savefig("images/04-01.png", bboxinches="tight")

    plt.figure(figsize=(10, 6))
    sns.clustermap(X.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0)
    plt.gcf().set_size_inches((14, 14))
    plt.tight_layout()
    plt.savefig("images/04-02.png", bboxinches="tight")

    corr_mat = X.corr().stack().reset_index()
    corr_mat.columns = ["var1", "var2", "corr"]
    corr_mat = corr_mat[corr_mat.var1 != corr_mat.var2].sort_values(by="corr", ascending=False)
    ic(corr_mat.head().append(corr_mat.tail()))

    plt.figure(figsize=(10, 6))
    y.boxplot()
    plt.tight_layout()
    plt.savefig("images/04-03.png", bboxinches="tight")

    ## Linear Regression for Statistical Inference: OLS with statsmodels
    ### Ticker-wise standardization
    # `statsmodels` warns of high design matrix condition numbers. This can arise when the variables are not
    # standardized and the Eigenvalues differ due to scaling. The following step avoids this warning.

    sectors = X.iloc[:, -10:]
    X = (
        X.drop(sectors.columns, axis=1)
        .groupby(level="ticker")
        .transform(lambda x: (x - x.mean()) / x.std())
        .join(sectors)
        .fillna(0)
    )

    ### 1-Day Returns
    target = "target_1d"
    model = OLS(endog=y[target], exog=add_constant(X))
    trained_model = model.fit()
    print(trained_model.summary())

    ### 5-Day Returns
    target = "target_5d"
    model = OLS(endog=y[target], exog=add_constant(X))
    trained_model = model.fit()
    print(trained_model.summary())

    #### Obtain the residuals
    preds = trained_model.predict(add_constant(X))
    residuals = y[target] - preds

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    sns.distplot(residuals, fit=norm, ax=axes[0], axlabel="Residuals", label="Residuals")
    axes[0].set_title("Residual Distribution")
    axes[0].legend()
    plot_acf(residuals, lags=10, zero=False, ax=axes[1], title="Residual Autocorrelation")
    axes[1].set_xlabel("Lags")
    fig.tight_layout()
    plt.savefig("images/04-04.png", bboxinches="tight")

    ### 10-Day Returns
    target = "target_10d"
    model = OLS(endog=y[target], exog=add_constant(X))
    trained_model = model.fit()
    print(trained_model.summary())

    ### Monthly Returns
    target = "target_21d"
    model = OLS(endog=y[target], exog=add_constant(X))
    trained_model = model.fit()
    print(trained_model.summary())
