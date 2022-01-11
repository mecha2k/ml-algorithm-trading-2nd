# Financial News - Preprocessing for word2vec
# See notebook [lda_financial_news](../15_topic_modeling/07_financial_news/lda_financial_news.ipynb)
# for download instructions.

import os, tarfile, sys, json
from pathlib import Path
from time import time
from pprint import pprint
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice
import pandas as pd
import seaborn as sns

import spacy
from spacy.lang.en import English
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser


np.random.seed(42)
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.set_option("float_format", "{:,.2f}".format)


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"


data_path = Path("..", "data", "us-financial-news")

results_path = Path("../data/ch16", "financial_news")
if not results_path.exists():
    results_path.mkdir(exist_ok=True)


if __name__ == "__main__":
    stop_words = set(
        pd.read_csv(
            "http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words",
            header=None,
            squeeze=True,
        ).tolist()
    )

    article_path = results_path / "articles.txt"
    clean_article_path = results_path / "articles_clean.txt"

    ## Load Data
    section_titles = [
        "Press Releases - CNBC",
        "Reuters: Company News",
        "Reuters: World News",
        "Reuters: Business News",
        "Reuters: Financial Services and Real Estate",
        "Top News and Analysis (pro)",
        "Reuters: Top News",
        "The Wall Street Journal &amp; Breaking News, Business, Financial and Economic News, World News and Video",
        "Business &amp; Financial News, U.S &amp; International Breaking News | Reuters",
        "Reuters: Money News",
        "Reuters: Technology News",
    ]

    articles = []
    counter = Counter()
    start = time()
    for f in fin_news_path.glob("*/**/*.json"):
        article = json.load(f.open())
        if article["thread"]["section_title"] in set(section_titles):
            text = article["text"].lower().split()
            counter.update(text)
            articles.append(" ".join([t for t in text if t not in stop_words]))
    print(f"Done loading {len(articles):,.0f} articles in {format_time(time()-start)}")
    article_path.write_text("\n".join(articles))

    ## Clean Financial News Articles
    articles = article_path.read_text().split("\n")
    print(len(articles))

    ### Sentence Boundary Detection
    def clean_doc(d):
        doc = []
        for sent in d.sents:
            s = [
                t.text.lower()
                for t in sent
                if not any([t.is_digit, not t.is_alpha, t.is_punct, t.is_space])
            ]
            if len(s) > 5 or len(sent) < 100:
                doc.append(" ".join(s))
        return doc

    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    clean_articles = []
    iter_articles = (article for article in articles)
    for i, doc in enumerate(nlp.pipe(iter_articles, batch_size=100, n_process=8), 1):
        if i % int(len(articles) / 100) + 1 == 0:
            print(f"{i / len(articles):.1%}", end=" ", flush=True)
        clean_articles.extend(clean_doc(doc))
    clean_article_path.write_text("\n".join(clean_articles))

    ### Corpus Stats
    print(len(clean_articles))

    vocab = Counter()
    sent_length = []
    for sentence in clean_articles:
        tokens = sentence.lower().split()
        sent_length.append(len(tokens))
        vocab.update(tokens)
    print(len(sent_length))
    print(pd.Series(sent_length).describe(percentiles=np.arange(0.1, 1, 0.1).round(1)))

    most_common = pd.DataFrame(vocab.most_common(), columns=["token", "count"]).pipe(
        lambda x: x[~x.token.str.lower().isin(stop_words)]
    )

    most_common.head(25).set_index("token")["count"].sort_values().plot.barh(
        title="Most Common Tokens", figsize=(14, 6)
    )
    plt.savefig("images/03-01")

    ### Inspect Result
    print(clean_articles[:3])

    ## Create n-grams
    max_length = 3

    n_grams = pd.DataFrame()
    start = time()
    sentences = LineSentence(clean_article_path.as_posix())
    for n in range(2, max_length + 1):
        print(n, end=" ")
        if n > 2:
            sentences = LineSentence((results_path / f"articles_{n-1}_grams.txt").as_posix())
        phrases = Phrases(sentences, threshold=100, min_count=10)

        s = pd.Series({k.decode("utf-8"): v for k, v in phrases.export_phrases(sentences)})
        s = s.to_frame("score").reset_index().rename(columns={"index": "phrase"}).assign(length=n)

        n_grams = pd.concat([n_grams, s])
        grams = Phraser(phrases)
        sentences = grams[sentences]

        with (results_path / f"articles_{n}_grams.txt").open("w") as f:
            for sentence in sentences:
                f.write(" ".join(sentence) + "\n")

    n_grams = n_grams.sort_values("score", ascending=False)
    n_grams.phrase = n_grams.phrase.str.replace("_", " ")
    n_grams["ngram"] = n_grams.phrase.str.replace(" ", "_")

    print("\n\tDuration: ", format_time(time() - start))
    print(f"\tngrams: {len(n_grams):,d}\n")
    print(n_grams.groupby("length").size())
    print(n_grams.groupby("length").apply(lambda x: x.nlargest(10, "score")))
