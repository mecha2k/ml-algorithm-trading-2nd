# RNN & Word Embeddings for SEC Filings to Predict Returns
# RNNs are commonly applied to various natural language processing tasks. We've already encountered sentiment analysis
# using text data in part three of this book.
# We are now going to apply an RNN model to SEC filings to learn custom word embeddings (see Chapter 16) and predict
# the returns over the week after the filing date.

from pathlib import Path
from time import time
from collections import Counter
from datetime import datetime, timedelta
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yfinance as yf

from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    GRU,
    Bidirectional,
    Embedding,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

idx = pd.IndexSlice
np.random.seed(seed=42)
tf.random.set_seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")

data_path = Path("..", "data", "sec-filings")
results_path = Path("../data/ch19", "sec-filings")
if not results_path.exists():
    results_path.mkdir(parents=True)


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"


if __name__ == "__main__":
    deciles = np.arange(0.1, 1, 0.1).round(1)

    ## Get stock price data
    selected_section_path = results_path / "ngrams_1"
    ngram_path = results_path / "ngrams"
    vector_path = results_path / "vectors"

    for path in [vector_path, selected_section_path, ngram_path]:
        if not path.exists():
            path.mkdir(parents=True)

    filing_index = pd.read_csv(data_path / "filing_index.csv", parse_dates=["DATE_FILED"]).rename(
        columns=str.lower
    )
    filing_index.index += 1
    filing_index.info()
    print(filing_index.head())
    print(filing_index.ticker.nunique())
    print(filing_index.date_filed.describe())

    ### Download stock price data using Yfinance
    # `yfinance` can be unstable so that connections drop; if you experience this you may want to store intermediate
    # results so you don't have to start over.
    yf_data, missing = [], []
    for i, (symbol, dates) in enumerate(filing_index.groupby("ticker").date_filed, 1):
        if i % 250 == 0:
            print(i, len(yf_data), len(set(missing)), flush=True)
        ticker = yf.Ticker(symbol)
        for filing, date in dates.to_dict().items():
            start = date - timedelta(days=93)
            end = date + timedelta(days=31)
            df = ticker.history(start=start, end=end)
            if df.empty:
                missing.append(symbol)
            else:
                yf_data.append(df.assign(ticker=symbol, filing=filing))

    yf_data = pd.concat(yf_data).rename(columns=str.lower)
    yf_data.to_hdf(results_path / "sec_returns.h5", "data/yfinance")

    yf_data = pd.read_hdf(results_path / "sec_returns.h5", "data/yfinance")
    print(yf_data.ticker.nunique())
    yf_data.info()

    ### Get (some) missing prices from Quandl
    to_do = filing_index.loc[
        ~filing_index.ticker.isin(yf_data.ticker.unique()), ["ticker", "date_filed"]
    ]
    print(to_do.date_filed.min())

    quandl_tickers = (
        pd.read_hdf("../data/assets.h5", "quandl/wiki/prices")
        .loc[idx["2012":, :], :]
        .index.unique("ticker")
    )
    quandl_tickers = list(set(quandl_tickers).intersection(set(to_do.ticker)))
    print(len(quandl_tickers))

    to_do = filing_index.loc[filing_index.ticker.isin(quandl_tickers), ["ticker", "date_filed"]]
    to_do.info()

    ohlcv = ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
    quandl = (
        pd.read_hdf("../data/assets.h5", "quandl/wiki/prices")
        .loc[idx["2012":, quandl_tickers], ohlcv]
        .rename(columns=lambda x: x.replace("adj_", ""))
    )
    quandl.info()

    quandl_data = []
    for i, (symbol, dates) in enumerate(to_do.groupby("ticker").date_filed, 1):
        if i % 100 == 0:
            print(i, end=" ", flush=True)
        for filing, date in dates.to_dict().items():
            start = date - timedelta(days=93)
            end = date + timedelta(days=31)
            quandl_data.append(
                quandl.loc[idx[start:end, symbol], :].reset_index("ticker").assign(filing=filing)
            )
    quandl_data = pd.concat(quandl_data)
    quandl_data.to_hdf(results_path / "sec_returns.h5", "data/quandl")

    ### Combine, clean and persist
    data = (
        pd.read_hdf(results_path / "sec_returns.h5", "data/yfinance")
        .drop(["dividends", "stock splits"], axis=1)
        .append(pd.read_hdf(results_path / "sec_returns.h5", "data/quandl"))
    )
    data = data.loc[:, ["filing", "ticker", "open", "high", "low", "close", "volume"]]
    data.info()
    print(data[["filing", "ticker"]].nunique())
    data.to_hdf(results_path / "sec_returns.h5", "prices")

    ## Copy filings with stock price data
    data = pd.read_hdf(results_path / "sec_returns.h5", "prices")
    filings_with_data = data.filing.unique()
    print(len(filings_with_data))

    ### Remove short and long sentences
    min_sentence_length = 5
    max_sentence_length = 50

    sent_length = Counter()
    for i, idx in enumerate(filings_with_data, 1):
        if i % 500 == 0:
            print(i, end=" ", flush=True)
        text = pd.read_csv(data_path / "selected_sections" / f"{idx}.csv").text
        sent_length.update(text.str.split().str.len().tolist())
        text = text[text.str.split().str.len().between(min_sentence_length, max_sentence_length)]
        text = "\n".join(text.tolist())
        with (selected_section_path / f"{idx}.txt").open("w") as f:
            f.write(text)

    sent_length = pd.Series(dict(sent_length.most_common()))

    with sns.axes_style("white"):
        sent_length.sort_index().cumsum().div(sent_length.sum()).loc[5:51].plot.bar(
            figsize=(12, 4), rot=0
        )
        plt.savefig("images/07-01.png")

    with sns.axes_style("white"):
        sent_length.sort_index().loc[:50].plot.bar(figsize=(14, 4))
        plt.savefig("images/07-02.png")

    ### Create bi- and trigrams
    # Combine all filings
    files = selected_section_path.glob("*.txt")
    texts = [f.read_text() for f in files]
    unigrams = ngram_path / "ngrams_1.txt"
    unigrams.write_text("\n".join(texts))
    texts = unigrams.read_text()

    # This takes quite some time; last attempt was 30 min per iteration.
    n_grams = []
    start = time()
    for i, n in enumerate([2, 3]):
        sentences = LineSentence(ngram_path / f"ngrams_{n-1}.txt")
        phrases = Phrases(
            sentences=sentences,
            min_count=25,  # ignore terms with a lower count
            threshold=0.5,  # accept phrases with higher score
            max_vocab_size=4000000,  # prune of less common words to limit memory use
            delimiter=b"_",  # how to join ngram tokens
            scoring="npmi",
        )

        s = pd.DataFrame(
            [[k.decode("utf-8"), v] for k, v in phrases.export_phrases(sentences)],
            columns=["phrase", "score"],
        ).assign(length=n)

        n_grams.append(s.groupby("phrase").score.agg(["mean", "size"]))
        print(n_grams[-1].nlargest(5, columns="size"))

        grams = Phraser(phrases)
        sentences = grams[sentences]
        (ngram_path / f"ngrams_{n}.txt").write_text("\n".join([" ".join(s) for s in sentences]))

        src_dir = results_path / f"ngrams_{n-1}"
        target_dir = results_path / f"ngrams_{n}"
        if not target_dir.exists():
            target_dir.mkdir()

        for f in src_dir.glob("*.txt"):
            text = LineSentence(f)
            text = grams[text]
            (target_dir / f"{f.stem}.txt").write_text("\n".join([" ".join(s) for s in text]))
        print("\n\tDuration: ", format_time(time() - start))

    n_grams = pd.concat(n_grams).sort_values("size", ascending=False)
    n_grams.to_parquet(results_path / "ngrams.parquet")
    print(n_grams.groupby(n_grams.index.str.replace("_", " ").str.count(" ")).size())

    ### Convert filings to integer sequences based on token count
    sentences = (ngram_path / "ngrams_3.txt").read_text().split("\n")
    n = len(sentences)

    token_cnt = Counter()
    for i, sentence in enumerate(sentences, 1):
        if i % 500000 == 0:
            print(f"{i/n:.1%}", end=" ", flush=True)
        token_cnt.update(sentence.split())
    token_cnt = pd.Series(dict(token_cnt.most_common()))
    token_cnt = token_cnt.reset_index()
    token_cnt.columns = ["token", "n"]
    token_cnt.to_parquet(results_path / "token_cnt")
    print(token_cnt.n.describe(deciles).apply(lambda x: f"{x:,.0f}"))
    token_cnt.info()
    print(token_cnt.nlargest(10, columns="n"))
    print(token_cnt.sort_values(by=["n", "token"], ascending=[False, True]).head())

    token_by_freq = token_cnt.sort_values(by=["n", "token"], ascending=[False, True]).token
    token2id = {token: i for i, token in enumerate(token_by_freq, 3)}
    print(len(token2id))

    for token, i in token2id.items():
        print(token, i)
        break

    def generate_sequences(min_len=100, max_len=20000, num_words=25000, oov_char=2):
        if not vector_path.exists():
            vector_path.mkdir()
        seq_length = {}
        skipped = 0
        for i, f in tqdm(enumerate((results_path / "ngrams_3").glob("*.txt"), 1)):
            file_id = f.stem
            text = f.read_text().split("\n")
            vector = [
                token2id[token] if token2id[token] + 2 < num_words else oov_char
                for line in text
                for token in line.split()
            ]
            vector = vector[:max_len]
            if len(vector) < min_len:
                skipped += 1
                continue
            seq_length[int(file_id)] = len(vector)
            np.save(vector_path / f"{file_id}.npy", np.array(vector))
        seq_length = pd.Series(seq_length)
        return seq_length

    seq_length = generate_sequences()
    pd.Series(seq_length).to_csv(results_path / "seq_length.csv")
    print(seq_length.describe(deciles))
    print(seq_length.sum())

    fig, axes = plt.subplots(ncols=3, figsize=(18, 5))
    token_cnt.n.plot(logy=True, logx=True, ax=axes[0], title="Token Frequency (log-log scale)")
    sent_length.sort_index().loc[:50].plot.bar(ax=axes[1], rot=0, title="Sentence Length")

    n = 5
    ticks = axes[1].xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in axes[1].xaxis.get_ticklabels()]
    axes[1].xaxis.set_ticks(ticks[n - 1 :: n])
    axes[1].xaxis.set_ticklabels(ticklabels[n - 1 :: n])
    axes[1].set_xlabel("Sentence Length")

    sns.distplot(seq_length, ax=axes[2], bins=50)
    axes[0].set_ylabel("Token Frequency")
    axes[0].set_xlabel("Token ID")

    axes[2].set_xlabel("# Words per Filing")
    axes[2].set_title("Filing Length Distribution")

    fig.suptitle("Corpus Stats", fontsize=13)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig("images/07_sec_seq_len.png", dpi=300)

    files = vector_path.glob("*.npy")
    filings = sorted([int(f.stem) for f in files])

    ## Prepare Model Data
    ### Create weekly forward returns
    prices = pd.read_hdf(results_path / "sec_returns.h5", "prices")
    prices.info()

    fwd_return = {}
    for filing in filings:
        date_filed = filing_index.at[filing, "date_filed"]
        price_data = prices[prices.filing == filing].close.sort_index()

        try:
            r = price_data.pct_change(periods=5).shift(-5).loc[:date_filed].iloc[-1]
        except:
            continue
        if not np.isnan(r) and -0.5 < r < 1:
            fwd_return[filing] = r
    print(len(fwd_return))

    ### Combine returns with filing data
    y, X = [], []
    for filing_id, fwd_ret in fwd_return.items():
        X.append(np.load(vector_path / f"{filing_id}.npy") + 2)
        y.append(fwd_ret)
    y = np.array(y)
    print(len(y), len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ### Pad sequences
    # In the second step, we convert the lists of integers into fixed-size arrays that we can stack and provide as
    # input to our RNN. The pad_sequence function produces arrays of equal length, truncated, and padded to conform
    # to maxlen, as follows:
    maxlen = 20000

    X_train = pad_sequences(X_train, truncating="pre", padding="pre", maxlen=maxlen)
    X_test = pad_sequences(X_test, truncating="pre", padding="pre", maxlen=maxlen)
    print(X_train.shape, X_test.shape)

    ## Define Model Architecture
    K.clear_session()

    # Now we can define our RNN architecture. The first layer learns the word embeddings. We define the embedding
    # dimension as previously using the input_dim keyword to set the number of tokens that we need to embed, the
    # output_dim keyword, which defines the size of each embedding, and how long each input sequence is going to be.
    embedding_size = 100

    # Note that we are using GRUs this time, which train faster and perform better on smaller data. We are also using
    # dropout for regularization, as follows:
    input_dim = X_train.max() + 1

    rnn = Sequential(
        [
            Embedding(
                input_dim=input_dim, output_dim=embedding_size, input_length=maxlen, name="EMB"
            ),
            BatchNormalization(name="BN1"),
            Bidirectional(GRU(32), name="BD1"),
            BatchNormalization(name="BN2"),
            Dropout(0.1, name="DO1"),
            Dense(5, name="D"),
            Dense(1, activation="linear", name="OUT"),
        ]
    )

    # The resulting model has over 2 million parameters.
    rnn.summary()

    rnn.compile(
        loss="mse",
        optimizer="Adam",
        metrics=[RootMeanSquaredError(name="RMSE"), MeanAbsoluteError(name="MAE")],
    )

    ## Train model
    early_stopping = EarlyStopping(monitor="val_MAE", patience=5, restore_best_weights=True)

    # Training stops after eight epochs and we recover the weights for the best models to find a high test AUC of 0.9346:
    training = rnn.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1,
    )

    ## Evaluate the Results
    df = pd.DataFrame(training.history)
    df.to_csv(results_path / "rnn_sec.csv", index=False)

    df.index += 1

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)
    plot_data = df[["RMSE", "val_RMSE"]].rename(
        columns={"RMSE": "Training", "val_RMSE": "Validation"}
    )
    plot_data.plot(ax=axes[0], title="Root Mean Squared Error")

    plot_data = df[["MAE", "val_MAE"]].rename(columns={"MAE": "Training", "val_MAE": "Validation"})
    plot_data.plot(ax=axes[1], title="Mean Absolute Error")

    for i in [0, 1]:
        axes[i].set_xlim(1, 10)
        axes[i].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig("images/07_sec_cv_performance.png", dpi=300)

    y_score = rnn.predict(X_test)
    rho, p = spearmanr(y_score.squeeze(), y_test)
    print(f"Information Coefficient: {rho*100:.2f} ({p:.2%})")

    g = sns.jointplot(y_score.squeeze(), y_test, kind="reg")
    plt.savefig("images/07-03.png")
