# Topic Modeling: Financial News
# This notebook contains an example of LDA applied to financial news articles.

from collections import Counter
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

import pyLDAvis
from pyLDAvis.gensim_models import prepare


stop_words = set(
    pd.read_csv(
        "http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words", header=None, squeeze=True
    ).tolist()
)


## Helper Viz Functions
def show_word_list(model, corpus, top=10, save=False):
    top_topics = model.top_topics(corpus=corpus, coherence="u_mass", topn=20)
    words, probs = [], []
    for top_topic, _ in top_topics:
        words.append([t[1] for t in top_topic[:top]])
        probs.append([t[0] for t in top_topic[:top]])

    fig, ax = plt.subplots(figsize=(model.num_topics * 1.2, 5))
    sns.heatmap(
        pd.DataFrame(probs).T, annot=pd.DataFrame(words).T, fmt="", ax=ax, cmap="Blues", cbar=False
    )
    fig.tight_layout()
    if save:
        fig.savefig(f"images/07_fin_news_wordlist_{top}.png", dpi=300)


def show_coherence(model, corpus, tokens, top=10, cutoff=0.01):
    top_topics = model.top_topics(corpus=corpus, coherence="u_mass", topn=20)
    word_lists = pd.DataFrame(model.get_topics().T, index=tokens)
    order = []
    for w, word_list in word_lists.items():
        target = set(word_list.nlargest(top).index)
        for t, (top_topic, _) in enumerate(top_topics):
            if target == set([t[1] for t in top_topic[:top]]):
                order.append(t)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    title = f"# Words with Probability > {cutoff:.2%}"
    (word_lists.loc[:, order] > cutoff).sum().reset_index(drop=True).plot.bar(
        title=title, ax=axes[1]
    )

    umass = model.top_topics(corpus=corpus, coherence="u_mass", topn=20)
    pd.Series([c[1] for c in umass]).plot.bar(title="Topic Coherence", ax=axes[0])
    fig.tight_layout()
    fig.savefig(f"images/07_fin_news_coherence_{top}", dpi=300)


def show_top_docs(model, corpus, docs):
    doc_topics = model.get_document_topics(corpus)
    df = pd.concat(
        [
            pd.DataFrame(doc_topic, columns=["topicid", "weight"]).assign(doc=i)
            for i, doc_topic in enumerate(doc_topics)
        ]
    )

    for topicid, data in df.groupby("topicid"):
        print(topicid, docs[int(data.sort_values("weight", ascending=False).iloc[0].doc)])
        print(pd.DataFrame(lda.show_topic(topicid=topicid)))


## Load Financial News
# The data is avaialble from [Kaggle](https://www.kaggle.com/jeet2016/us-financial-news-articles).
# Download and unzip into data directory in repository root folder, then rename the enclosing folder to
# `us-financial-news` and the subfolders, so you get the following directory structure:
# data
#   |-us-financial-news
#      |-2018_01
#      |-2018_02
#      |-2018_03
#      |-2018_04
#      |-2018_05

data_path = Path("..", "data", "us-financial-news")

results_path = Path("../data/ch15/results", "financial_news")
if not results_path.exists():
    results_path.mkdir(parents=True)

if __name__ == "__main__":
    # We limit the article selection to the following sections in the dataset:
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

    def read_articles():
        articles = []
        counter = Counter()
        for f in data_path.glob("*/**/*.json"):
            article = json.load(f.open())
            if article["thread"]["section_title"] in set(section_titles):
                text = article["text"].lower().split()
                counter.update(text)
                articles.append(" ".join([t for t in text if t not in stop_words]))
        return articles, counter

    articles, counter = read_articles()
    print(f"Done loading {len(articles):,.0f} articles")

    most_common = pd.DataFrame(counter.most_common(), columns=["token", "count"]).pipe(
        lambda x: x[~x.token.str.lower().isin(stop_words)]
    )
    most_common.head(10)

    ## Preprocessing with SpaCy
    def clean_doc(d):
        doc = []
        for t in d:
            if not any(
                [
                    t.is_stop,
                    t.is_digit,
                    not t.is_alpha,
                    t.is_punct,
                    t.is_space,
                    t.lemma_ == "-PRON-",
                ]
            ):
                doc.append(t.lemma_)
        return " ".join(doc)

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 6000000
    nlp.select_pipes(disable="ner")
    print(nlp.pipe_names)

    def preprocess(articles):
        iter_articles = (article for article in articles)
        clean_articles = []
        for i, doc in enumerate(nlp.pipe(iter_articles, batch_size=100, n_process=8), 1):
            if i % 1000 == 0:
                print(f"{i / len(articles):.2%}", end=" ", flush=True)
            clean_articles.append(clean_doc(doc))
        return clean_articles

    clean_articles = preprocess(articles)
    clean_path = results_path / "clean_text"
    clean_path.write_text("\n".join(clean_articles))

    ## Vectorize data
    docs = clean_path.read_text().split("\n")
    print(len(docs))

    ### Explore cleaned data
    article_length, token_count = [], Counter()
    for i, doc in enumerate(docs, 1):
        if i % 1e6 == 0:
            print(i, end=" ", flush=True)
        d = doc.lower().split()
        article_length.append(len(d))
        token_count.update(d)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    (
        pd.DataFrame(token_count.most_common(), columns=["token", "count"])
        .pipe(lambda x: x[~x.token.str.lower().isin(stop_words)])
        .set_index("token")
        .squeeze()
        .iloc[:25]
        .sort_values()
        .plot.barh(ax=axes[0], title="Most frequent tokens")
    )
    sns.boxenplot(x=pd.Series(article_length), ax=axes[1])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Word Count (log scale)")
    axes[1].set_title("Article Length Distribution")
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/15_fn_explore", dpi=300)

    print(pd.Series(article_length).describe(percentiles=np.arange(0.1, 1.0, 0.1)))

    docs = [x.lower() for x in docs]
    print(docs[3])

    ### Set vocab parameters
    min_df = 0.005
    max_df = 0.1
    ngram_range = (1, 1)
    binary = False

    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=min_df, max_df=max_df, ngram_range=ngram_range, binary=binary
    )
    dtm = vectorizer.fit_transform(docs)
    tokens = vectorizer.get_feature_names()
    print(dtm.shape)

    corpus = Sparse2Corpus(dtm, documents_columns=False)
    id2word = pd.Series(tokens).to_dict()
    dictionary = Dictionary.from_corpus(corpus, id2word)

    ## Train & Evaluate LDA Model
    logging.basicConfig(
        filename="gensim.log", format="%(asctime)s:%(levelname)s:%(message)s", level=logging.DEBUG
    )
    logging.root.level = logging.DEBUG

    ### Train models with 5-25 topics
    num_topics = [5, 10, 15, 20]
    for topics in num_topics:
        print(topics)
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=topics,
            chunksize=len(docs),
            update_every=1,
            alpha="auto",  # a-priori belief for the topics' probability
            eta="auto",  # a-priori belief on word probability
            decay=0.5,  # percentage of previous lambda value forgotten
            offset=1.0,
            eval_every=1,
            passes=10,
            iterations=50,
            gamma_threshold=0.001,
            minimum_probability=0.01,  # filter topics with lower probability
            minimum_phi_value=0.01,  # lower bound on term probabilities
            random_state=42,
        )
        lda_model.save((results_path / f"model_{topics}").as_posix())

    ### Evaluate results
    # We show results for one model using a vocabulary of 3,800 tokens based on min_df=0.1% and max_df=25% with a
    # single pass to avoid length training time for 20 topics. We can use pyldavis topic_info attribute to compute
    # relevance values for lambda=0.6 that produces the following word list
    def eval_lda_model(ntopics, model, corpus=corpus, tokens=tokens):
        show_word_list(model=model, corpus=corpus, top=ntopics, save=True)
        show_coherence(model=model, corpus=corpus, tokens=tokens, top=ntopics)
        vis = prepare(model, corpus, dictionary, mds="tsne")
        pyLDAvis.save_html(vis, f"lda_{ntopics}.html")
        return 2 ** (-model.log_perplexity(corpus))

    lda_models = {}
    perplexity = {}
    for ntopics in num_topics:
        print(ntopics)
        lda_models[ntopics] = LdaModel.load((results_path / f"model_{ntopics}").as_posix())
        perplexity[ntopics] = eval_lda_model(ntopics=ntopics, model=lda_models[ntopics])

    ### Perplexity
    pd.Series(perplexity).plot.bar()
    sns.despine()
    plt.savefig("images/15-01.png")

    ### PyLDAVis for 15 Topics
    vis = prepare(lda_models[15], corpus, dictionary, mds="tsne")
    pyLDAvis.display(vis)

    ## LDAMultiCore Timing
    df = pd.read_csv(results_path / "lda_multicore_test_results.csv")
    print(df.head())

    df[df.num_topics == 10].set_index("workers")[["duration", "test_perplexity"]].plot.bar(
        subplots=True, layout=(1, 2), figsize=(14, 5), legend=False
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig("images/15-02.png")
