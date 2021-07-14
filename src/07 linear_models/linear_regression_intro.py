import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
pd.options.display.float_format = "{:,.2f}".format

x = np.linspace(-5, 50, 100)
y = 50 + 2 * x + np.random.normal(0, 20, size=len(x))
data = pd.DataFrame({"X": x, "Y": y})
ax = data.plot.scatter(x="X", y="Y", figsize=(14, 6))
sns.despine()
plt.tight_layout()
plt.show()

X = sm.add_constant(data["X"])
model = sm.OLS(data["Y"], X).fit()
print(model.summary())

beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
pd.Series(beta, index=X.columns)

data["y-hat"] = model.predict()
data["residuals"] = model.resid
ax = data.plot.scatter(x="X", y="Y", c="darkgrey", figsize=(14, 6))
data.plot.line(x="X", y="y-hat", ax=ax)
for _, row in data.iterrows():
    plt.plot((row.X, row.X), (row.Y, row["y-hat"]), "k-")
sns.despine()
plt.tight_layout()
plt.show()

size = 25
X_1, X_2 = np.meshgrid(np.linspace(-50, 50, size), np.linspace(-50, 50, size), indexing="ij")
data = pd.DataFrame({"X_1": X_1.ravel(), "X_2": X_2.ravel()})
data["Y"] = 50 + data.X_1 + 3 * data.X_2 + np.random.normal(0, 50, size=size ** 2)

three_dee = plt.figure(figsize=(15, 5)).gca(projection="3d")
three_dee.scatter(data.X_1, data.X_2, data.Y, c="g")
sns.despine()
plt.tight_layout()
plt.show()

X = data[["X_1", "X_2"]]
y = data["Y"]

X_ols = sm.add_constant(X)
model = sm.OLS(y, X_ols).fit()
print(model.summary())

beta = np.linalg.inv(X_ols.T.dot(X_ols)).dot(X_ols.T.dot(y))
pd.Series(beta, index=X_ols.columns)

plt.rc("figure", figsize=(12, 7))
plt.text(0.01, 0.05, str(model.summary()), {"fontsize": 14}, fontproperties="monospace")
plt.axis("off")
plt.tight_layout()
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.1)
plt.savefig("../images/multiple_regression_summary.png", bbox_inches="tight", dpi=300)
plt.show()

three_dee = plt.figure(figsize=(15, 5)).gca(projection="3d")
three_dee.scatter(data.X_1, data.X_2, data.Y, c="g")
data["y-hat"] = model.predict()
to_plot = data.set_index(["X_1", "X_2"]).unstack().loc[:, "y-hat"]
three_dee.plot_surface(
    X_1, X_2, to_plot.values, color="black", alpha=0.2, linewidth=1, antialiased=True
)
for _, row in data.iterrows():
    plt.plot((row.X_1, row.X_1), (row.X_2, row.X_2), (row.Y, row["y-hat"]), "k-")
three_dee.set_xlabel("$X_1$")
three_dee.set_ylabel("$X_2$")
three_dee.set_zlabel("$Y, \hat{Y}$")
sns.despine()
plt.tight_layout()
plt.show()

scaler = StandardScaler()
X_ = scaler.fit_transform(X)

sgd = SGDRegressor(
    loss="squared_loss",
    fit_intercept=True,
    shuffle=True,
    random_state=42,
    learning_rate="invscaling",
    eta0=0.01,
    power_t=0.25,
)

sgd.fit(X=X_, y=y)
coeffs = (sgd.coef_ * scaler.scale_) + scaler.mean_
pd.Series(coeffs, index=X.columns)
resids = pd.DataFrame({"sgd": y - sgd.predict(X_), "ols": y - model.predict(sm.add_constant(X))})
resids.pow(2).sum().div(len(y)).pow(0.5)
resids.plot.scatter(x="sgd", y="ols")
sns.despine()
plt.tight_layout()
plt.show()
