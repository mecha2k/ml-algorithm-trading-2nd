import numpy as np
import pandas as pd
import scipy.stats as ss
from itertools import product
from icecream import ic


def get_analytical_max_sr(mu, sigma, num_trials):
    """Compute the expected maximum Sharpe ratio (Analytically)"""
    # Euler-Mascheroni constant
    emc = 0.5772156649
    maxZ = (1 - emc) * ss.norm.ppf(1 - 1.0 / num_trials) + emc * ss.norm.ppf(
        1 - 1 / (num_trials * np.e)
    )
    return mu + sigma * maxZ


def get_numerical_max_sr(mu, sigma, num_trials, n_iter):
    """Compute the expected maximum Sharpe ratio (Numerically)"""
    max_sr, count = [], 0
    while count < n_iter:
        count += 1
        series = np.random.normal(mu, sigma, num_trials)
        max_sr.append(max(series))
    return np.mean(max_sr), np.std(max_sr)


def simulate(mu, sigma, num_trials, n_iter):
    """Get analytical and numerical solutions"""
    expected_max_sr = get_analytical_max_sr(mu, sigma, num_trials)
    mean_max_sr, stdmean_max_sr = get_numerical_max_sr(mu, sigma, num_trials, n_iter)
    return expected_max_sr, mean_max_sr, stdmean_max_sr


def deflated_sr():
    n_iter, sigma, output, count = 1e4, 1, [], 0
    mu_trials = list(product(np.linspace(-100, 100, 101), range(100, 1001, 100)))
    for i, prod_ in enumerate(mu_trials, 1):
        if i % 100 == 0:
            print(i, end=" ", flush=True)
        mu, num_trials = prod_[0], prod_[1]
        expected_max_sr, mean_max_sr, std_max_sr = simulate(mu, sigma, num_trials, n_iter)
        err = expected_max_sr - mean_max_sr
        output.append(
            [mu, sigma, num_trials, n_iter, expected_max_sr, mean_max_sr, std_max_sr, err]
        )
    output = pd.DataFrame(
        output,
        columns=[
            "mu",
            "sigma",
            "num_trials",
            "n_iter",
            "expected_max_sr",
            "mean_max_sr",
            "std_max_sr",
            "err",
        ],
    )
    print(output.info())
    output.to_csv("../data/DSR.csv")


if __name__ == "__main__":
    deflated_sr()

    output = pd.read_csv("../data/DSR.csv")
    ic(output.head())
