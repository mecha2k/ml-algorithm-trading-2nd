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
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import IncrementalPCA

sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format

## Convert GloVE Vectors to gensim format
# The various GloVE vectors are available [here](https://nlp.stanford.edu/projects/glove/). Download link
# for the [wikipedia](http://nlp.stanford.edu/data/glove.6B.zip) version. Unzip and store in `data/glove`.
glove_path = Path("..", "data", "glove")

analogies_path = Path("../data/ch16", "analogies-en.txt")

results_path = Path("../data/ch16/results", "glove")
if not results_path.exists():
    results_path.mkdir(parents=True)

if __name__ == "__main__":
    ### WikiPedia
    glove_wiki_file = glove_path / "glove.6B.300d.txt"
    word2vec_wiki_file = glove_path / "glove.wiki.gensim.txt"
    glove2word2vec(glove_input_file=glove_wiki_file, word2vec_output_file=word2vec_wiki_file)

    # ### Twitter Data
    # glove_twitter_file = glove_path / "glove.twitter.27B.200d.txt"
    # word2vec_twitter_file = glove_path / "glove.twitter.gensim.txt"
    # glove2word2vec(glove_input_file=glove_twitter_file, word2vec_output_file=word2vec_twitter_file)
    #
    # ### Common Crawl
    # glove_crawl_file = glove_path / "glove.840B.300d.txt"
    # word2vec_crawl_file = glove_path / "glove.crawl.gensim.txt"
    # glove2word2vec(glove_input_file=glove_crawl_file, word2vec_output_file=word2vec_crawl_file)

    ## Evaluate embeddings
    def eval_analogies(file_name, vocab=30000):
        model = KeyedVectors.load_word2vec_format(file_name, binary=False)
        accuracy = model.accuracy(analogies_path, restrict_vocab=vocab, case_insensitive=True)
        return (
            pd.DataFrame(
                [[c["section"], len(c["correct"]), len(c["incorrect"])] for c in accuracy],
                columns=["category", "correct", "incorrect"],
            )
            .assign(samples=lambda x: x.correct.add(x.incorrect))
            .assign(average=lambda x: x.correct.div(x.samples))
            .drop(["correct", "incorrect"], axis=1)
        )

    # ### twitter result
    # twitter_result = eval_analogies(word2vec_twitter_file, vocab=100000)
    # twitter_result.to_csv(glove_path / "accuracy_twitter.csv", index=False)
    # print(twitter_result)

    ### wiki result
    wiki_result = eval_analogies(word2vec_wiki_file, vocab=100000)
    wiki_result.to_csv(glove_path / "accuracy_wiki.csv", index=False)
    print(wiki_result)

    # ### Common Crawl result
    # crawl_result = eval_analogies(word2vec_crawl_file, vocab=100000)
    # crawl_result.to_csv(glove_path / "accuracy_crawl.csv", index=False)
    # print(crawl_result)
    #
    # ### Combine & compare results
    # cat_dict = {
    #     "capital-common-countries": "Capitals",
    #     "capital-world": "Capitals RoW",
    #     "city-in-state": "City-State",
    #     "currency": "Currency",
    #     "family": "Famliy",
    #     "gram1-adjective-to-adverb": "Adj-Adverb",
    #     "gram2-opposite": "Opposite",
    #     "gram3-comparative": "Comparative",
    #     "gram4-superlative": "Superlative",
    #     "gram5-present-participle": "Pres. Part.",
    #     "gram6-nationality-adjective": "Nationality",
    #     "gram7-past-tense": "Past Tense",
    #     "gram8-plural": "Plural",
    #     "gram9-plural-verbs": "Plural Verbs",
    #     "total": "Total",
    # }
    #
    # accuracy = (
    #     twitter_result.assign(glove="Twitter")
    #     .append(wiki_result.assign(glove="Wiki"))
    #     .append(crawl_result.assign(glove="Crawl"))
    # )
    #
    # accuracy.category = accuracy.category.replace(cat_dict)
    # accuracy = accuracy.rename(columns=str.capitalize)
    # accuracy.to_csv(results_path / "accuracy.csv", index=False)
    # accuracy = pd.read_csv(results_path / "accuracy.csv")
    #
    # fig, ax = plt.subplots(figsize=(16, 4))
    # sns.barplot(x="Category", y="Average", hue="Glove", data=accuracy, ax=ax)
    # ax.set_title(
    #     f"Word Vector Accuracy by Glove Source: Twitter: {0.564228:.2%}, Wiki: {0.75444:.2%}, Crawl: {0.779347:.2%}"
    # )
    # ax.set_ylim(0, 1)
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    # sns.despine()
    # fig.tight_layout()
    # fig.savefig("images/16_glove_accuracy", dpi=300)
    #
    # ## Visualize Embeddings
    # ### Load GloVe Wiki Vectors
    # model = KeyedVectors.load_word2vec_format(word2vec_wiki_file, binary=False)
    # accuracy = model.accuracy(questions=str(analogies_path), restrict_vocab=100000)
    #
    # vectors = model.vectors[:100000]
    # vectors /= norm(vectors, axis=1).reshape(-1, 1)
    # print(vectors.shape)
    #
    # words = model.index2word[:100000]
    # word2id = {w: i for i, w in enumerate(words)}
    #
    # ### Project Embedding into 2D
    # pca = IncrementalPCA(n_components=2)
    #
    # vectors2D = pca.fit_transform(vectors)
    # pd.Series(pca.explained_variance_ratio_).mul(100)
    #
    # ### Plot Analogy Examples
    # results = pd.DataFrame()
    # correct = incorrect = 0
    # for section in accuracy:
    #     correct += len(section["correct"])
    #     incorrect += len(section["incorrect"])
    #     df = (
    #         pd.DataFrame(section["correct"])
    #         .apply(lambda x: x.str.lower())
    #         .assign(section=section["section"])
    #     )
    #     results = pd.concat([results, df])
    #
    # def find_most_similar_analogy(v):
    #     """Find analogy that most similar in 2D"""
    #     v1 = vectors2D[v[1]] - vectors2D[v[0]]
    #     v2 = vectors2D[v[3]] - vectors2D[v[2]]
    #     idx, most_similar = None, np.inf
    #
    #     for i in range(len(v1)):
    #         similarity = cosine(v1[i], v2[i])
    #         if similarity < most_similar:
    #             idx = i
    #             most_similar = similarity
    #     return idx
    #
    # def get_plot_lims(coordinates):
    #     xlim, ylim = coordinates.agg(["min", "max"]).T.values
    #     xrange, yrange = (xlim[1] - xlim[0]) * 0.1, (ylim[1] - ylim[0]) * 0.1
    #     xlim[0], xlim[1] = xlim[0] - xrange, xlim[1] + xrange
    #     ylim[0], ylim[1] = ylim[0] - yrange, ylim[1] + yrange
    #     return xlim, ylim
    #
    # fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
    # axes = axes.flatten()
    # fc = ec = "darkgrey"
    # for s, (section, result) in enumerate(results.groupby("section")):
    #     if s > 11:
    #         break
    #
    #     df = result.drop("section", axis=1).apply(lambda x: x.map(word2id))
    #     most_similar_idx = find_most_similar_analogy(df)
    #
    #     best_analogy = result.iloc[most_similar_idx, :4].tolist()
    #     analogy_idx = [words.index(word) for word in best_analogy]
    #     best_analogy = [a.capitalize() for a in best_analogy]
    #     coords = pd.DataFrame(vectors2D[analogy_idx])  # xy array
    #
    #     xlim, ylim = get_plot_lims(coords)
    #     axes[s].set_xlim(xlim)
    #     axes[s].set_ylim(ylim)
    #
    #     for i in [0, 2]:
    #         axes[s].annotate(
    #             s=best_analogy[i],
    #             xy=coords.iloc[i + 1],
    #             xytext=coords.iloc[i],
    #             arrowprops=dict(width=1, headwidth=5, headlength=5, fc=fc, ec=ec, shrink=0.1),
    #             fontsize=12,
    #         )
    #
    #         axes[s].annotate(
    #             best_analogy[i + 1],
    #             xy=coords.iloc[i + 1],
    #             xytext=coords.iloc[i + 1],
    #             va="center",
    #             ha="center",
    #             fontsize=12,
    #             color="darkred" if i == 2 else "k",
    #         )
    #
    #     axes[s].axis("off")
    #     title = " ".join([s.capitalize() for s in section.split("-") if not s.startswith("gram")])
    #     axes[s].set_title(title, fontsize=16)
    #
    # fig.suptitle("word2vec Embeddings | Analogy Examples", fontsize=18)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    # fig.savefig("images/01-01.png")
