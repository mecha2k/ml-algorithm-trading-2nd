# How to train your own word vector embeddings using Gensim
# Many tasks require embeddings or domain-specific vocabulary that pre-trained models based on a generic corpus may not
# represent well or at all. Standard word2vec models are not able to assign vectors to out-of-vocabulary words and
# instead use a default vector that reduces their predictive value.
# E.g., when working with industry-specific documents, the vocabulary or its usage may change over time as new
# technologies or products emerge. As a result, the embeddings need to evolve as well. In addition, corporate earnings
# releases use nuanced language not fully reflected in Glove vectors pre-trained on Wikipedia articles.
# In this notebook we illustrate the more performant gensim adaptation of the code provided by the word2vec authors.
# To illustrate the word2vec network architecture, we use the Financial News data that we first introduced
# in chapter 14 on Topic Modeling.

import warnings
from time import time
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist, cosine

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from sklearn.decomposition import IncrementalPCA


np.random.seed(42)
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.set_option("float_format", "{:,.2f}".format)
warnings.filterwarnings("ignore")

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"


news_path = Path("../data/ch16", "fin_news")
data_path = news_path / "data"
analogy_path = Path("../data/ch16", "analogies-en.txt")

gensim_path = news_path / "gensim"
if not gensim_path.exists():
    gensim_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ## Model Configuration
    NGRAMS = 3  # Longest ngram in text
    MIN_FREQ = 100
    WINDOW_SIZE = 5
    EMBEDDING_SIZE = 300
    NEGATIVE_SAMPLES = 20
    EPOCHS = 1

    FILE_NAME = f"articles_{NGRAMS}_grams.txt"

    ## Sentence Generator
    sentence_path = data_path / FILE_NAME
    sentences = LineSentence(str(sentence_path))

    ## Train word2vec Model
    start = time()
    model = Word2Vec(
        sentences,
        sg=1,  # set to 1 for skipgram; CBOW otherwise
        size=EMBEDDING_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_FREQ,
        negative=NEGATIVE_SAMPLES,
        workers=8,
        iter=EPOCHS,
        alpha=0.05,
    )

    # persist model
    model.save(str(gensim_path / "word2vec.model"))

    # persist word vectors
    model.wv.save(str(gensim_path / "word_vectors.bin"))
    print("Duration:", format_time(time() - start))

    ## Evaluate results
    cat_dict = {
        "capital-common-countries": "Capitals",
        "capital-world": "Capitals RoW",
        "city-in-state": "City-State",
        "currency": "Currency",
        "family": "Famliy",
        "gram1-adjective-to-adverb": "Adj-Adverb",
        "gram2-opposite": "Opposite",
        "gram3-comparative": "Comparative",
        "gram4-superlative": "Superlative",
        "gram5-present-participle": "Pres. Part.",
        "gram6-nationality-adjective": "Nationality",
        "gram7-past-tense": "Past Tense",
        "gram8-plural": "Plural",
        "gram9-plural-verbs": "Plural Verbs",
        "total": "Total",
    }

    def accuracy_by_category(acc, detail=True):
        results = [[c["section"], len(c["correct"]), len(c["incorrect"])] for c in acc]
        results = pd.DataFrame(results, columns=["category", "correct", "incorrect"])
        results["average"] = results.correct.div(results[["correct", "incorrect"]].sum(1))
        if detail:
            print(results.sort_values("average", ascending=False))
        return (
            results.loc[results.category == "total", ["correct", "incorrect", "average"]]
            .squeeze()
            .tolist()
        )

    # gensim computes accuracy based on source text files
    detailed_accuracy = model.wv.accuracy(analogy_path.as_posix(), case_insensitive=True)

    # get accuracy per category
    summary = accuracy_by_category(detailed_accuracy)
    print("Base Accuracy: Correct {:,.0f} | Wrong {:,.0f} | Avg {:,.2%}\n".format(*summary))

    most_sim = model.wv.most_similar(positive=["woman", "king"], negative=["man"], topn=20)
    pd.DataFrame(most_sim, columns=["token", "similarity"])

    counter = Counter(sentence_path.read_text().split())

    most_common = pd.DataFrame(counter.most_common(), columns=["token", "count"])
    most_common = most_common[most_common["count"] > MIN_FREQ]
    most_common["p"] = np.log(most_common["count"]) / np.log(most_common["count"]).sum()

    similars = pd.DataFrame()
    for token in np.random.choice(most_common.token, size=10, p=most_common.p):
        similars[token] = [s[0] for s in model.wv.most_similar(token)]
    print(similars.T)

    ## Continue Training
    accuracies = [summary]
    best_accuracy = summary[-1]
    for i in range(1, 10):
        start = time()
        model.train(sentences, epochs=1, total_examples=model.corpus_count)
        detailed_accuracy = model.wv.accuracy(analogy_path)
        accuracies.append(accuracy_by_category(detailed_accuracy, detail=False))
        print(
            f"{i:02} | Duration: {format_time(time() - start)} | Accuracy: {accuracies[-1][-1]:.2%} "
        )
        if accuracies[-1][-1] > best_accuracy:
            model.save(str(gensim_path / f"word2vec_{i:02}.model"))
            model.wv.save(str(gensim_path / f"word_vectors_{i:02}.bin"))
            best_accuracy = accuracies[-1][-1]
        (
            pd.DataFrame(accuracies, columns=["correct", "wrong", "average"]).to_csv(
                gensim_path / "accuracies.csv", index=False
            )
        )
    model.wv.save(str(gensim_path / "word_vectors_final.bin"))

    ## Evaluate Best Model
    pd.DataFrame(
        accuracies,
        columns=["correct", "wrong", "average"],
        index=list(range(1, len(accuracies) + 1)),
    ).average.plot()
    plt.savefig("images/05-01")

    best_model = Word2Vec.load((gensim_path / "word2vec_06.model").as_posix())

    # gensim computes accuracy based on source text files
    detailed_accuracy = best_model.wv.accuracy(analogy_path.as_posix(), case_insensitive=True)

    # get accuracy per category
    summary = accuracy_by_category(detailed_accuracy)
    print("Base Accuracy: Correct {:,.0f} | Wrong {:,.0f} | Avg {:,.2%}\n".format(*summary))

    results = [[c["section"], len(c["correct"]), len(c["incorrect"])] for c in detailed_accuracy]
    results = pd.DataFrame(results, columns=["category", "correct", "incorrect"])
    results["category"] = results.category.map(cat_dict)
    results["average"] = results.correct.div(results[["correct", "incorrect"]].sum(1))
    results = results.rename(columns=str.capitalize).set_index("Category")
    total = results.loc["Total"]
    results = results.drop("Total")

    most_sim = best_model.wv.most_similar(positive=["woman", "king"], negative=["man"], topn=20)
    print(pd.DataFrame(most_sim, columns=["token", "similarity"]))

    fig, axes = plt.subplots(ncols=2, figsize=(16, 5))
    axes[0] = results.loc[:, ["Correct", "Incorrect"]].plot.bar(
        stacked=True, ax=axes[0], title="Analogy Accuracy"
    )
    ax1 = results.loc[:, ["Average"]].plot(ax=axes[0], secondary_y=True, lw=1, c="k", rot=35)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    (
        pd.DataFrame(most_sim, columns=["token", "similarity"])
        .set_index("token")
        .similarity.sort_values()
        .tail(10)
        .plot.barh(xlim=(0.3, 0.37), ax=axes[1], title="Closest matches for Woman + King - Man")
    )
    fig.tight_layout()
    fig.savefig("images/05-02")

    counter = Counter(sentence_path.read_text().split())
    most_common = pd.DataFrame(counter.most_common(), columns=["token", "count"])
    most_common = most_common[most_common["count"] > MIN_FREQ]
    most_common["p"] = np.log(most_common["count"]) / np.log(most_common["count"]).sum()

    similars = pd.DataFrame()
    for token in np.random.choice(most_common.token, size=10, p=most_common.p):
        similars[token] = [s[0] for s in best_model.wv.most_similar(token)]
    print(similars.T)

    similars.T.iloc[:5, :5].to_csv("images/most_similar.csv")

    ## Resources
    # - [Distributed representations of words and phrases and their compositionality]
    #   (http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
    # - [Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf?)
    # - [Sebastian Ruder's Blog](http://ruder.io/word-embeddings-1/)
