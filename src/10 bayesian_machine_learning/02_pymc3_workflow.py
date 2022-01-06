# Logistic regression with PyMC3
# As discussed in chapter 6, logistic regression estimates a linear relationship between a set of
# features and a binary outcome, mediated by a sigmoid function to ensure the model produces probabilities.
# The frequentist approach resulted in point estimates for the parameters that measure the influence of
# each feature on the probability that a data point belongs to the positive class, with confidence intervals
# based on assumptions about the parameter distribution.
# In contrast, Bayesian logistic regression estimates the posterior distribution over the parameters itself.
# The posterior allows for more robust estimates of what is called a Bayesian credible interval for each
# parameter with the benefit of more transparency about the model’s uncertainty.

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scipy import stats

import pandas_datareader.data as web

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

import theano
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from IPython.display import HTML

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
pd.options.display.float_format = "{:,.2f}".format

data_path = Path("../data")
fig_path = Path("images")
model_path = Path("../data/models")
for p in [data_path, fig_path, model_path]:
    if not p.exists():
        p.mkdir()


## The Data: Recessions & Leading Indicators
# We will use a small and simple dataset so we can focus on the workflow. We use the Federal Reserve’s Economic Data
# (FRED) service (see Chapter 2) to download the US recession dates as defined by the National Bureau of Economic
# Research. We also source four variables that are commonly used to predict the onset of a recession (Kelley 2019)
# and available via FRED, namely:
# - The long-term spread of the treasury yield curve, defined as the difference between the ten-year and
#   the three-month Treasury yield.
# - The University of Michigan’s consumer sentiment indicator
# - The National Financial Conditions Index (NFCI), and
# - The NFCI nonfinancial leverage subindex.

if __name__ == "__main__":
    ### Download from FRED
    indicators = ["JHDUSRGDPBR", "T10Y3M", "NFCI", "NFCILEVERAGE", "UMCSENT"]
    var_names = ["recession", "yield_curve", "financial_conditions", "leverage", "sentiment"]

    features = var_names[1:]
    label = var_names[0]

    var_display = ["Recession", "Yield Curve", "Financial Conditions", "Leverage", "Sentiment"]
    col_dict = dict(zip(var_names, var_display))

    data = web.DataReader(indicators, "fred", 1980, 2020).ffill().resample("M").last().dropna()
    data.columns = var_names

    # We standardize the features so all have mean 0 standard deviation of 1:
    data.loc[:, features] = scale(data.loc[:, features])
    data.info()

    ### Quick exploration
    mi = []
    months = list(range(1, 25))
    for month in months:
        df_ = data.copy()
        df_[label] = df_[label].shift(-month)
        df_ = df_.dropna()
        mi.append(mutual_info_classif(df_.loc[:, features], df_[label]))
    mi = pd.DataFrame(mi, columns=features, index=months)

    mi.sum(1).mul(100).iloc[:12].sort_index(ascending=False).plot.barh(
        figsize=(12, 4), xlim=(15, 40)
    )
    plt.savefig("images/02_01.png", bboxinches="tight")

    fig, ax = plt.subplots(figsize=(20, 3))
    sns.heatmap(
        mi.rename(columns=col_dict).T * 100, cmap="Greys", ax=ax, annot=True, fmt=".1f", cbar=False
    )
    ax.set_xlabel("Lead Time (Months)")
    ax.set_title("Mutual Information between Indicators and Recession by Lead Time")
    fig.tight_layout()
    plt.savefig("images/02_02.png", bboxinches="tight")

    data[label] = data[label].shift(-12)
    data = data.dropna()
    data_ = pd.melt(data.rename(columns=col_dict), id_vars="Recession").rename(
        columns=str.capitalize
    )
    fig = plt.figure(figsize=(10, 6))
    g = sns.catplot(x="Recession", y="Value", col="Variable", data=data_, kind="box")
    plt.savefig("images/02_03.png", bboxinches="tight")

    X = data.loc[:, features]
    y = data[label]
    print(y.value_counts())

    data.to_csv("../data/recessions.csv")
    data = pd.read_csv("../data/recessions.csv", index_col=0)
    data.info()

    ### Models
    simple_model = "recession ~ yield_curve + leverage"
    full_model = simple_model + " + financial_conditions + sentiment"

    ## MAP Inference
    # A probabilistic program consists of observed and unobserved random variables (RVs). As discussed, we define
    # the observed RVs via likelihood distributions and unobserved RVs via prior distributions. PyMC3 includes numerous
    # probability distributions for this purpose.
    # The PyMC3 library makes it very straightforward to perform approximate Bayesian inference for logistic regression.
    # Logistic regression models the probability that individual i earns a high income based on k features as outlined
    # in the below figure that uses plate notation:

    ### Manual Model Specification
    # We will use the context manager with to define a manual_logistic_model that we can refer to later as a
    # probabilistic model:
    # - The random variables for the unobserved parameters for intercept and two features are expressed using
    #   uninformative priors that assume normal distributions with mean 0 and standard deviation of 100.
    # - The likelihood combines the parameters with the data according to the specification of the logistic regression
    # - The outcome is modeled as a Bernoulli RV with success probability given by the likelihood.

    with pm.Model() as manual_logistic_model:
        # random variables for coefficients with
        # uninformative priors for each parameter
        intercept = pm.Normal("intercept", 0, sd=100)
        beta_1 = pm.Normal("beta_1", 0, sd=100)
        beta_2 = pm.Normal("beta_2", 0, sd=100)

        # Transform random variables into vector of probabilities p(y_i=1)
        # according to logistic regression model specification.
        likelihood = pm.invlogit(intercept + beta_1 * data.yield_curve + beta_2 * data.leverage)

        # Bernoulli random vector with probability of success
        # given by sigmoid function and actual data as observed
        pm.Bernoulli(name="logit", p=likelihood, observed=data.recession)

    print(manual_logistic_model.model)

    # The command `pm.model_to_graphviz(manual_logistic_model)` produces the plate notation displayed below.
    # It shows the unobserved parameters as light and the observed elements as dark circles. The rectangle indicates
    # the number of repetitions of the observed model element implied by the data included in the model definition.

    pm.model_to_graphviz(manual_logistic_model)

    # opionally: persist
    # graph = pm.model_to_graphviz(manual_logistic_model)
    # graph.save('log_reg.dot')

    ### Run Inference
    with manual_logistic_model:
        # compute maximum a-posteriori estimate
        # for logistic regression weights
        manual_map_estimate = pm.find_MAP()

    def print_map(result):
        return pd.Series({k: np.asscalar(v) for k, v in result.items()})

    print_map(manual_map_estimate)

    ### GLM Model
    # PyMC3 includes numerous common models so that we can usually leave the manual specification for custom
    # applications. The following code defines the same logistic regression as a member of the Generalized Linear
    # Models (GLM) family using the formula format inspired by the statistical language R and ported to python by the
    # `patsy` library:

    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula(simple_model, data, family=pm.glm.families.Binomial())

    pm.model_to_graphviz(logistic_model)

    ### MAP Estimate
    # We obtain point MAP estimates for the three parameters using the just defined model’s .find_MAP() method:

    with logistic_model:
        map_estimate = pm.find_MAP()

    # PyMC3 solves the optimization problem of finding the posterior point with the highest density using the
    # quasi-Newton Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm but offers several alternatives provided by the
    # scipy library. The result is virtually identically to the corresponding statsmodels estimate:

    model = smf.logit(formula=simple_model, data=data)
    result = model.fit()
    print(result.summary())
    print_map(map_estimate)
    print(result.params)

    ## Markov Chain Monte Carlo
    def plot_traces(traces, burnin=2000):
        summary = pm.summary(traces[burnin:])["mean"].to_dict()
        ax = pm.traceplot(traces[burnin:], figsize=(15, len(traces.varnames) * 1.5), lines=summary)

        for i, mn in enumerate(summary.values()):
            ax[i, 0].annotate(
                f"{mn:.2f}",
                xy=(mn, 0),
                xycoords="data",
                xytext=(5, 10),
                textcoords="offset points",
                rotation=90,
                va="bottom",
                fontsize="large",
                color="#AA0022",
            )

    ### Define the Model
    # We will use the full model to illustrate Markov Chain Monte Carlo inference:
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula(formula=full_model, data=data, family=pm.glm.families.Binomial())
    print(logistic_model.basic_RVs)

    ### Metropolis-Hastings
    # We will use the Metropolis-Hastings algorithm to sample from the posterior distribution.
    # Explore the hyperparameters of Metropolis-Hastings such as the proposal distribution variance to speed up the
    # convergence. Use `plot_traces` function to visually inspect the convergence. You may also use MAP-estimate to
    # initialize the sampling scheme to speed things up. This will make the warmup (burnin) period shorter since you
    # will start from a probable point.

    with logistic_model:
        trace_mh = pm.sample(tune=1000, draws=5000, step=pm.Metropolis(), cores=4)

    ### Inspect Trace
    plot_traces(trace_mh, burnin=0)

    pm.trace_to_dataframe(trace_mh).info()

    ### Continue Training
    with logistic_model:
        trace_mh = pm.sample(draws=100000, step=pm.Metropolis(), trace=trace_mh)
    plot_traces(trace_mh, burnin=0)

    # optionally: persist
    with open("../data/logistic_model_mh.pkl", "wb") as buff:
        pickle.dump({"model": logistic_model, "trace": trace_mh}, buff)

    # optionally: restore persisted model
    with open("../data/logistic_model_mh.pkl", "rb") as buff:
        pickled_data = pickle.load(buff)

    logistic_model, trace_mh = pickled_data["model"], pickled_data["trace"]
    pm.summary(trace_mh)

    ### NUTS sampler
    # Using pm.sample without specifying a sampling method defaults to the No U-Turn Sampler, a form of Hamiltonian
    # Monte Carlo that automatically tunes its parameters. It usually converges faster and gives less correlated
    # samples compared to vanilla Metropolis-Hastings.
    # Note that variables measured on very different scales can slow down the sampling process. Hence, we first apply
    # sklearn’s the scale() function to standardize the variables age, hours and educ.

    #### Draw small number of samples
    # Once we have defined our model as above with the new formula, we are ready to perform inference to approximate
    # the posterior distribution. MCMC sampling algorithms are available through the pm.sample() function.
    # By default, PyMC3 automatically selects the most efficient sampler and initializes the sampling process for
    # efficient convergence. For a continuous model, PyMC3 chooses the NUTS sampler discussed in the previous section.
    # It also runs variational inference via ADVI to find good starting parameters for the sampler. One among several
    # alternatives is to use the MAP estimate.
    # To see convergence looks like, we first draw only 100 samples after tuning the sampler for 1,000 iterations that
    # will be discarded. The sampling process can be parallelized for multiple chains using the cores argument (except
    # when using GPU).

    draws = 100
    tune = 1000
    with logistic_model:
        trace_NUTS = pm.sample(
            draws=draws, tune=tune, init="adapt_diag", chains=4, cores=1, random_seed=42
        )

    trace_df = pm.trace_to_dataframe(trace_NUTS).assign(chain=lambda x: x.index // draws)
    trace_df.info()

    plot_traces(trace_NUTS, burnin=0)

    #### Continue Training
    # The resulting trace contains the sampled values for each random variable. We can continue sampling by providing
    # the trace of a prior run as input:

    draws = 50000
    chains = 4
    with logistic_model:
        trace_NUTS = pm.sample(
            draws=draws,
            tune=tune,
            init="adapt_diag",
            trace=trace_NUTS,
            chains=chains,
            cores=1,
            random_seed=42,
        )

    plot_traces(trace_NUTS, burnin=1000)

    #### Persist Results
    with open("../data/logistic_model_nuts.pkl", "wb") as buff:
        pickle.dump({"model": logistic_model, "trace": trace_NUTS}, buff)

    with open("../data/logistic_model_nuts.pkl", "rb") as buff:
        pickled_data = pickle.load(buff)

    logistic_model, trace_NUTS = pickled_data["model"], pickled_data["trace"]

    #### Combine Traces
    df = (
        pm.trace_to_dataframe(trace_NUTS)
        .iloc[200:]
        .reset_index(drop=True)
        .assign(chain=lambda x: x.index // draws)
    )
    trace_df = pd.concat([trace_df.assign(samples=100), df.assign(samples=len(df) + len(trace_df))])
    trace_df.info()

    #### Visualize both traces
    trace_df_long = pd.melt(trace_df, id_vars=["samples", "chain"])
    trace_df_long.info()

    g = sns.FacetGrid(
        trace_df_long, col="variable", row="samples", hue="chain", sharex="col", sharey=False
    )
    g = g.map(sns.distplot, "value", hist=False, rug=False)
    plt.savefig("images/02_04.png", bboxinches="tight")

    model = smf.logit(formula=full_model, data=data)
    result = model.fit()
    print(result.summary())

    pm.summary(trace_NUTS).assign(statsmodels=result.params).to_csv(model_path / "trace_nuts.csv")
    print(pm.summary(trace_NUTS).assign(statsmodels=result.params))

    ### Computing Credible Intervals
    # We can compute the credible intervals, the Bayesian counterpart of confidence intervals, as percentiles of the
    # trace. The resulting boundaries reflect our confidence about the range of the parameter value for a given
    # probability threshold, as opposed to the number of times the parameter will be within this range for a large
    # number of trials.
    def get_credible_int(trace, param):
        b = trace[param]
        lb, ub = np.percentile(b, 2.5), np.percentile(b, 97.5)
        lb, ub = np.exp(lb), np.exp(ub)
        return b, lb, ub

    b = trace_NUTS["yield_curve"]
    lb, ub = np.percentile(b, 2.5), np.percentile(b, 97.5)
    lb, ub = np.exp(lb), np.exp(ub)
    print(f"P({lb:.3f} < Odds Ratio < {ub:.3f}) = 0.95")

    b, lb, ub = get_credible_int(trace_NUTS, "yield_curve")
    print(f"P({lb:.3f} < Odds Ratio < {ub:.3f}) = 0.95")

    fig, axes = plt.subplots(figsize=(14, 4), ncols=2)
    b, lb, ub = get_credible_int(trace_NUTS, "yield_curve")
    sns.distplot(np.exp(b), axlabel="Odds Ratio", ax=axes[0])
    axes[0].set_title(f"Yield Curve: P({lb:.3f} < Odds Ratio < {ub:.3f}) = 0.95")
    axes[0].axvspan(lb, ub, alpha=0.5, color="gray")

    b, lb, ub = get_credible_int(trace_NUTS, "leverage")
    sns.distplot(np.exp(b), axlabel="Odds Ratio", ax=axes[1])
    axes[1].set_title(f"Leverage: P({lb:.3f} < Odds Ratio < {ub:.3f}) = 0.95")
    axes[1].axvspan(lb, ub, alpha=0.5, color="gray")

    fig.suptitle("Credible Intervals", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig("images/02_05.png", bboxinches="tight")

    ## Variational Inference
    # ### Run Automatic Differentation Variational Inference (ADVI)
    # The interface for variational inference is very similar to the MCMC implementation. We just use the fit()
    # instead of the sample() function, with the option to include an early stopping CheckParametersConvergence
    # callback if the distribution-fitting process converged up to a given tolerance:
    with logistic_model:
        callback = CheckParametersConvergence(diff="absolute")
        approx = pm.fit(n=100000, callbacks=[callback])

    ### Persist Result
    with open("../data/logistic_model_advi.pkl", "wb") as buff:
        pickle.dump({"model": logistic_model, "approx": approx}, buff)

    ### Sample from approximated distribution
    # We can draw samples from the approximated distribution to obtain a trace object as above for the MCMC sampler:
    trace_advi = approx.sample(10000)
    pm.summary(trace_advi)
    pm.summary(trace_advi).to_csv(model_path / "trace_advi.csv")

    ## Model Diagnostics
    # Bayesian model diagnostics includes validating that the sampling process has converged and consistently samples
    # from high-probability areas of the posterior, and confirming that the model represents the data well.
    # For high-dimensional models with many variables, it becomes cumbersome to inspect numerous at traces. When using
    # NUTS, the energy plot helps to assess problems of convergence. It summarizes how efficiently the random process
    # explores the posterior. The plot shows the energy and the energy transition matrix that should be well matched
    # as in the below example (see references for conceptual detail).

    ### Energy Plot
    pm.energyplot(trace_NUTS)

    ### Forest Plot
    pm.forestplot(trace_NUTS)

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    pm.forestplot(trace_NUTS, ax=axes[0])
    axes[0].set_title("Forest Plot")

    pm.energyplot(trace_NUTS, ax=axes[1])
    axes[1].set_title("Energy Plot")
    fig.tight_layout()
    plt.savefig("images/02_06.png", bboxinches="tight")

    ### Posterior Predictive Checks
    # PPCs are very useful for examining how well a model fits the data. They do so by generating data from the model
    # using parameters from draws from the posterior. We use the function pm.sample_ppc for this purpose and obtain
    # n samples for each observation (the GLM module automatically names the outcome ‘y’):
    ppc = pm.sample_posterior_predictive(trace_NUTS, samples=500, model=logistic_model)
    print(ppc["y"].shape)

    y_score = np.mean(ppc["y"], axis=0)
    roc_auc_score(y_score=np.mean(ppc["y"], axis=0), y_true=data.recession)

    ## Prediction
    # Follows PyMC3 [docs](https://docs.pymc.io/notebooks/posterior_predictive.html)
    # Predictions use theano’s shared variables to replace the training data with test data before running posterior
    # predictive checks. To facilitate visualization, we create a variable with a single predictor hours, create the
    # train and test datasets, and convert the former to a shared variable. Note that we need to use numpy arrays and
    # provide a list of column labels:

    ### Train-test split
    X = data[["yield_curve"]]
    labels = X.columns
    y = data.recession
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ### Create shared theano variable
    X_shared = theano.shared(X_train.values)

    ### Define logistic model
    with pm.Model() as logistic_model_pred:
        pm.glm.GLM(x=X_shared, labels=labels, y=y_train, family=pm.glm.families.Binomial())

    ### Run NUTS sampler
    with logistic_model_pred:
        pred_trace = pm.sample(draws=10000, tune=1000, chains=2, cores=1, init="adapt_diag")

    ### Replace shared variable with test set
    # We then run the sampler as before, and apply the pm.sample_ppc function to the resulting trace after replacing
    # the train with test data:
    X_shared.set_value(X_test)

    ppc = pm.sample_posterior_predictive(pred_trace, model=logistic_model_pred, samples=100)

    #### Check AUC Score
    y_score = np.mean(ppc["y"], axis=0)
    roc_auc_score(y_score=np.mean(ppc["y"], axis=0), y_true=y_test)

    def invlogit(x):
        return np.exp(x) / (1 + np.exp(x))

    x = X_test.yield_curve

    fig, ax = plt.subplots(figsize=(14, 5))
    β = stats.beta((ppc["y"] == 1).sum(axis=0), (ppc["y"] == 0).sum(axis=0))

    # estimated probability
    ax.scatter(x=x, y=β.mean())

    # error bars on the estimate
    plt.vlines(x, *β.interval(0.95))

    # actual outcomes
    ax.scatter(x=x, y=y_test, marker="x")

    # True probabilities
    x_ = np.linspace(x.min() * 1.05, x.max() * 1.05, num=100)
    ax.plot(-x_, invlogit(x_), linestyle="-")

    ax.set_xlabel("Yield Curve")
    ax.set_ylabel("Recession")
    ax.invert_xaxis()
    fig.tight_layout()
    plt.savefig("images/02_07.png", bboxinches="tight")

    # ## MCMC Sampler Animation
    # The code is based on [MCMC visualization tutorial](https://twiecki.github.io/blog/2014/01/02/visualizing-mcmc/).

    ### Setup
    # Number of MCMC iteration to animate.
    burnin = 1000
    samples = 1000

    var1 = "yield_curve"
    var1_range = (trace_df[var1].min() * 0.95, trace_df[var1].max() * 1.05)

    var2 = "sentiment"
    var2_range = (trace_df[var2].min() * 0.95, trace_df[var2].max() * 1.05)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)

    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula(formula=full_model, data=data, family=pm.glm.families.Binomial())

    ### NUTS samples
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        trace = trace_df.iloc[: i + 1]
        idx = list(range(len(trace)))
        line1.set_data(trace[var1].iloc[::-1], idx)
        line2.set_data(idx, trace[var2].iloc[::-1])
        line3.set_data(trace[var1], trace[var2])
        line4.set_data(trace[var1], trace[var2])
        line5.set_data(
            [trace[var1].iloc[-1], trace[var1].iloc[-1]], [trace[var2].iloc[-1], var2_range[1]]
        )
        line6.set_data(
            [trace[var1].iloc[-1], var1_range[1]], [trace[var2].iloc[-1], trace[var2].iloc[-1]]
        )
        return lines

    # In[90]:

    with logistic_model:
        nuts_trace = pm.sample(draws=samples, tune=burnin, init="adapt_diag", chains=1)
        trace_df = pm.trace_to_dataframe(nuts_trace)
    trace_df.to_csv("trace.csv", index=False)
    trace_df = pd.read_csv("trace.csv")
    print(trace_df.info())

    # In[98]:

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221, xlim=var1_range, ylim=(0, samples))
    ax2 = fig.add_subplot(224, xlim=(0, samples), ylim=var2_range)
    ax3 = fig.add_subplot(223, xlim=var1_range, ylim=var2_range, xlabel=var1, ylabel=var2)

    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    (line1,) = ax1.plot([], [], lw=1)
    (line2,) = ax2.plot([], [], lw=1)
    (line3,) = ax3.plot([], [], "o", lw=2, alpha=0.1)
    (line4,) = ax3.plot([], [], lw=1, alpha=0.3)
    (line5,) = ax3.plot([], [], "k", lw=1)
    (line6,) = ax3.plot([], [], "k", lw=1)
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    lines = [line1, line2, line3, line4, line5, line6]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=samples, interval=5, blit=True
    )
    anim.save("images/nuts.mp4", writer=writer)
    # or display; either requres ffmpeg installation
    # HTML(anim.to_html5_video())

    ### Metropolis-Hastings samples
    with logistic_model:
        step = pm.Metropolis()
        mh_trace = pm.sample(draws=samples, tune=burnin, step=step, chains=1)
        trace_df = pm.trace_to_dataframe(mh_trace)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221, xlim=var1_range, ylim=(0, samples))
    ax2 = fig.add_subplot(224, xlim=(0, samples), ylim=var2_range)
    ax3 = fig.add_subplot(223, xlim=var1_range, ylim=var2_range, xlabel=var1, ylabel=var2)

    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    (line1,) = ax1.plot([], [], lw=1)
    (line2,) = ax2.plot([], [], lw=1)
    (line3,) = ax3.plot([], [], "o", lw=2, alpha=0.1)
    (line4,) = ax3.plot([], [], lw=1, alpha=0.3)
    (line5,) = ax3.plot([], [], "k", lw=1)
    (line6,) = ax3.plot([], [], "k", lw=1)
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    lines = [line1, line2, line3, line4, line5, line6]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        trace = trace_df.iloc[: i + 1]
        idx = list(range(len(trace)))
        line1.set_data(trace[var1].iloc[::-1], idx)
        line2.set_data(idx, trace[var2].iloc[::-1])
        line3.set_data(trace[var1], trace[var2])
        line4.set_data(trace[var1], trace[var2])
        line5.set_data(
            [trace[var1].iloc[-1], trace[var1].iloc[-1]], [trace[var2].iloc[-1], var2_range[1]]
        )
        line6.set_data(
            [trace[var1].iloc[-1], var1_range[1]], [trace[var2].iloc[-1], trace[var2].iloc[-1]]
        )
        return lines

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=samples, interval=5, blit=True
    )
    anim.save("images/nuts-01.mp4", writer=writer)

    # HTML(anim.to_html5_video())
