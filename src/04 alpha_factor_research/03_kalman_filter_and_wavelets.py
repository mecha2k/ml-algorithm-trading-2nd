import itertools
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pykalman import KalmanFilter


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
idx = pd.IndexSlice


if __name__ == "__main__":
    DATA_STORE = "../data/assets.h5"
    with pd.HDFStore(DATA_STORE) as store:
        sp500 = store["sp500/stooq"].loc["2009":"2010", "close"]
    print(sp500.head())

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01,
    )
    state_means, _ = kf.filter(sp500)

    sp500_smoothed = sp500.to_frame("close")
    sp500_smoothed["Kalman Filter"] = state_means
    for months in [1, 2, 3]:
        sp500_smoothed[f"MA ({months}m)"] = sp500.rolling(window=months * 21).mean()

    ax = sp500_smoothed.plot(title="Kalman Filter vs Moving Average", figsize=(14, 6), lw=1, rot=0)
    ax.set_xlabel("")
    ax.set_ylabel("S&P 500")
    sns.despine()
    plt.savefig("../images/ch04_im01.png", dpi=300, bboxinches="tight")

    wavelet = pywt.Wavelet("db6")
    phi, psi, x = wavelet.wavefun(level=5)
    df = pd.DataFrame({"$\phi$": phi, "$\psi$": psi}, index=x)
    df.plot(title="Daubechies", subplots=True, layout=(1, 2), figsize=(14, 4), lw=2, rot=0)
    plt.savefig("../images/ch04_im02.png", dpi=300, bboxinches="tight")

    plot_data = [("db", (4, 3)), ("sym", (4, 3)), ("coif", (3, 2))]
    for family, (rows, cols) in plot_data:
        fig = plt.figure(figsize=(24, 12))
        fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.02, left=0.06, right=0.97, top=0.94)
        colors = itertools.cycle("bgrcmyk")
        wnames = pywt.wavelist(family)
        i = iter(wnames)
        for col in range(cols):
            for row in range(rows):
                try:
                    wavelet = pywt.Wavelet(next(i))
                except StopIteration:
                    break
                phi, psi, x = wavelet.wavefun(level=5)

                color = next(colors)
                ax = fig.add_subplot(rows, 2 * cols, 1 + 2 * (col + row * cols))
                ax.set_title(wavelet.name + " phi")
                ax.plot(x, phi, color, lw=1)
                ax.set_xlim(min(x), max(x))

                ax = fig.add_subplot(rows, 2 * cols, 1 + 2 * (col + row * cols) + 1)
                ax.set_title(wavelet.name + " psi")
                ax.plot(x, psi, color, lw=1)
                ax.set_xlim(min(x), max(x))
        sns.despine()
    plt.savefig("../images/ch04_im03.png", dpi=300, bboxinches="tight")

    for family, (rows, cols) in [("bior", (4, 3)), ("rbio", (4, 3))]:
        fig = plt.figure(figsize=(24, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.2, bottom=0.02, left=0.06, right=0.97, top=0.94)

        colors = itertools.cycle("bgrcmyk")
        wnames = pywt.wavelist(family)
        i = iter(wnames)
        for col in range(cols):
            for row in range(rows):
                try:
                    wavelet = pywt.Wavelet(next(i))
                except StopIteration:
                    break
                phi, psi, phi_r, psi_r, x = wavelet.wavefun(level=5)
                row *= 2

                color = next(colors)
                ax = fig.add_subplot(2 * rows, 2 * cols, 1 + 2 * (col + row * cols))
                ax.set_title(wavelet.name + " phi")
                ax.plot(x, phi, color, lw=1)
                ax.set_xlim(min(x), max(x))

                ax = fig.add_subplot(2 * rows, 2 * cols, 2 * (1 + col + row * cols))
                ax.set_title(wavelet.name + " psi")
                ax.plot(x, psi, color, lw=1)
                ax.set_xlim(min(x), max(x))

                row += 1
                ax = fig.add_subplot(2 * rows, 2 * cols, 1 + 2 * (col + row * cols))
                ax.set_title(wavelet.name + " phi_r")
                ax.plot(x, phi_r, color, lw=1)
                ax.set_xlim(min(x), max(x))

                ax = fig.add_subplot(2 * rows, 2 * cols, 1 + 2 * (col + row * cols) + 1)
                ax.set_title(wavelet.name + " psi_r")
                ax.plot(x, psi_r, color, lw=1)
                ax.set_xlim(min(x), max(x))
        sns.despine()
    plt.savefig("../images/ch04_im04.png", dpi=300, bboxinches="tight")

    pywt.families(short=False)
    signal = pd.read_hdf(DATA_STORE, "sp500/stooq").loc["2008":"2009"].close.pct_change().dropna()

    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
    wavelet = "db6"
    for i, scale in enumerate([0.1, 0.5]):
        coefficients = pywt.wavedec(signal, wavelet, mode="per")
        coefficients[1:] = [
            pywt.threshold(i, value=scale * signal.max(), mode="soft") for i in coefficients[1:]
        ]
        reconstructed_signal = pywt.waverec(coefficients, wavelet, mode="per")
        signal.plot(
            color="b",
            alpha=0.5,
            label="original signal",
            lw=2,
            title=f"Threshold Scale: {scale:.1f}",
            ax=axes[i],
        )
        pd.Series(reconstructed_signal, index=signal.index).plot(
            c="k", label="DWT smoothing}", linewidth=1, ax=axes[i]
        )
        axes[i].legend()
    fig.tight_layout()
    sns.despine()
    plt.savefig("../images/ch04_im05.png", dpi=300, bboxinches="tight")
