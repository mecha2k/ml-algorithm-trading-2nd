## Topic Modeling: Latent Dirichlet Allocation with sklearn
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import pyLDAvis
from pyLDAvis.sklearn import prepare

from wordcloud import WordCloud
from termcolor import colored

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import joblib

sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = (14, 8)
pd.options.display.float_format = "{:,.2f}".format

# pyLDAvis.enable_notebook()

# change to your data path if necessary
DATA_DIR = Path("../data")
data_path = DATA_DIR / "bbc"

results_path = Path("../data/ch15/results")
model_path = Path("../data/ch15/results", "bbc")
if not model_path.exists():
    model_path.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    ## Load BBC data
    # Using the BBC data as before, we use [sklearn.decomposition.LatentDirichletAllocation]
    # (http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) to train
    # an LDA model with five topics.
    files = sorted(list(data_path.glob("**/*.txt")))
    doc_list = []
    for i, file in enumerate(files):
        with open(str(file), encoding="latin1") as f:
            topic = file.parts[-2]
            lines = f.readlines()
            heading = lines[0].strip()
            body = " ".join([l.strip() for l in lines[1:]])
            doc_list.append([topic.capitalize(), heading, body])

    ### Convert to DataFrame
    docs = pd.DataFrame(doc_list, columns=["topic", "heading", "article"])
    docs.info()

    ## Create Train & Test Sets
    train_docs, test_docs = train_test_split(
        docs, stratify=docs.topic, test_size=125, random_state=42
    )
    print(train_docs.shape, test_docs.shape)
    print(pd.Series(test_docs.topic).value_counts())

    ### Vectorize train & test sets
    # experiments with different settings results yields the following hyperparameters (see issue 50)
    vectorizer = TfidfVectorizer(max_df=0.11, min_df=0.026, stop_words="english")

    train_dtm = vectorizer.fit_transform(train_docs.article)
    words = vectorizer.get_feature_names_out()
    print(train_dtm)

    test_dtm = vectorizer.transform(test_docs.article)
    print(test_dtm)

    ## LDA with sklearn
    n_components = 5
    topic_labels = [f"Topic {i}" for i in range(1, n_components + 1)]

    lda_base = LatentDirichletAllocation(
        n_components=n_components, n_jobs=-1, learning_method="batch", max_iter=10
    )
    lda_base.fit(train_dtm)

    ### Persist model
    # The model tracks the in-sample perplexity during training and stops iterating once this measure stops improving.
    # We can persist and load the result as usual with sklearn objects:
    joblib.dump(lda_base, model_path / "lda_10_iter.pkl")

    lda_base = joblib.load(model_path / "lda_10_iter.pkl")
    print(lda_base)

    ## Explore topics & word distributions
    # pseudo counts
    topics_count = lda_base.components_
    print(topics_count.shape)
    print(topics_count[:5])

    topics_prob = topics_count / topics_count.sum(axis=1).reshape(-1, 1)
    topics = pd.DataFrame(topics_prob.T, index=words, columns=topic_labels)
    print(topics.head())

    # all words have positive probability for all topics
    print(topics[topics.gt(0).all(1)].shape[0] == topics.shape[0])

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(
        topics.sort_values(topic_labels, ascending=False),
        cmap="Blues",
        ax=ax,
        cbar_kws={"shrink": 0.6},
    )
    fig.tight_layout()
    fig.savefig("images/04-01.png")

    top_words = {}
    for topic, words_ in topics.items():
        top_words[topic] = words_.nlargest(10).index.tolist()
    print(pd.DataFrame(top_words))

    fig, axes = plt.subplots(nrows=5, sharey=True, sharex=True, figsize=(10, 15))
    for i, (topic, prob) in enumerate(topics.items()):
        sns.histplot(data=prob, ax=axes[i], bins=100, kde=False)
        axes[i].set_yscale("log")
        axes[i].xaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:.1%}".format(x)))
    fig.suptitle("Topic Distributions")
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/04-02.png")

    ## Evaluate Fit on Train Set
    train_preds = lda_base.transform(train_dtm)
    print(train_preds.shape)

    train_eval = pd.DataFrame(train_preds, columns=topic_labels, index=train_docs.topic)
    print(train_eval.head())
    train_eval.groupby(level="topic").mean().plot.bar(title="Avg. Topic Probabilities")

    df = train_eval.groupby(level="topic").idxmax(axis=1).reset_index(-1, drop=True)
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(
        df.groupby(level="topic").value_counts(normalize=True).unstack(-1),
        annot=True,
        fmt=".1%",
        cmap="Blues",
        square=True,
    )
    plt.title("Train Data: Topic Assignments")
    plt.savefig("images/04-03.png")

    ## Evaluate Fit on Test Set
    test_preds = lda_base.transform(test_dtm)
    test_eval = pd.DataFrame(test_preds, columns=topic_labels, index=test_docs.topic)
    print(test_eval.head())

    fig = plt.figure(figsize=(10, 6))
    test_eval.groupby(level="topic").mean().plot.bar(
        title="Avg. Topic Probabilities", figsize=(12, 4), rot=0
    )
    plt.xlabel("")
    sns.despine()
    plt.tight_layout()
    plt.savefig("images/04-04.png")

    fig = plt.figure(figsize=(10, 6))
    df = test_eval.groupby(level="topic").idxmax(axis=1).reset_index(-1, drop=True)
    sns.heatmap(
        df.groupby(level="topic").value_counts(normalize=True).unstack(-1),
        annot=True,
        fmt=".1%",
        cmap="Blues",
        square=True,
    )
    plt.title("Topic Assignments")
    plt.savefig("images/04-05.png")

    ## Retrain until perplexity no longer decreases
    lda_opt = LatentDirichletAllocation(
        n_components=5,
        n_jobs=-1,
        max_iter=500,
        learning_method="batch",
        evaluate_every=5,
        verbose=1,
        random_state=42,
    )
    lda_opt.fit(train_dtm)

    joblib.dump(lda_opt, model_path / "lda_opt.pkl")
    lda_opt = joblib.load(model_path / "lda_opt.pkl")

    train_opt_eval = pd.DataFrame(
        data=lda_opt.transform(train_dtm), columns=topic_labels, index=train_docs.topic
    )

    test_opt_eval = pd.DataFrame(
        data=lda_opt.transform(test_dtm), columns=topic_labels, index=test_docs.topic
    )

    ## Compare Train & Test Topic Assignments
    fig, axes = plt.subplots(ncols=2, figsize=(18, 8))
    source = ["Train", "Test"]
    for i, df in enumerate([train_opt_eval, test_opt_eval]):
        df = df.groupby(level="topic").idxmax(axis=1).reset_index(-1, drop=True)
        sns.heatmap(
            df.groupby(level="topic").value_counts(normalize=True).unstack(-1),
            annot=True,
            fmt=".1%",
            cmap="Blues",
            square=True,
            ax=axes[i],
        )
        axes[i].set_title("{} Data: Topic Assignments".format(source[i]))
    plt.savefig("images/04-06.png")

    ## Explore misclassified articles
    test_assignments = test_docs.assign(predicted=test_opt_eval.idxmax(axis=1).values)
    print(test_assignments.head())

    misclassified = test_assignments[
        (test_assignments.topic == "Entertainment") & (test_assignments.predicted == "Topic 4")
    ]
    print(misclassified.heading)
    print(misclassified.article.tolist())

    ## PyLDAVis
    # LDAvis helps you interpret LDA results by answer 3 questions:
    #
    # 1. What is the meaning of each topic?
    # 2. How prevalent is each topic?
    # 3. How do topics relate to each other?

    # Topic visualization facilitates the evaluation of topic quality using human judgment. pyLDAvis is a python port
    # of LDAvis, developed in R and D3.js. We will introduce the key concepts; each LDA implementation notebook contains
    # examples.
    # pyLDAvis displays the global relationships among topics while also facilitating their semantic evaluation by
    # inspecting the terms most closely associated with each individual topic and, inversely, the topics associated
    # with each term. It also addresses the challenge that terms that are frequent in a corpus tend to dominate the
    # multinomial distribution over words that define a topic. LDAVis introduces the relevance r of term w to topic t
    # to produce a flexible ranking of key terms using a weight parameter 0<=ƛ<=1.
    #
    # With $\phi_{wt}$  as the model’s probability estimate of observing the term w for topic t, and   as the marginal
    # probability of w in the corpus:
    # $$r(w, k | \lambda) = \lambda \log(\phi_{kw}) + (1 − \lambda) \log \frac{\phi_{kw}}{p_w}$$
    #
    # The first term measures the degree of association of term t with topic w, and the second term measures the lift
    # or saliency, i.e., how much more likely the term is for the topic than in the corpus.
    # The tool allows the user to interactively change ƛ to adjust the relevance, which updates the ranking of terms.
    # User studies have found that ƛ=0.6 produces the most plausible results.

    ## Refit using all data
    vectorizer = CountVectorizer(max_df=0.5, min_df=5, stop_words="english", max_features=2000)
    dtm = vectorizer.fit_transform(docs.article)

    lda_all = LatentDirichletAllocation(
        n_components=5,
        max_iter=500,
        learning_method="batch",
        evaluate_every=10,
        random_state=42,
        verbose=1,
    )
    lda_all.fit(dtm)
    joblib.dump(lda_all, model_path / "lda_all.pkl")

    lda_all = joblib.load(model_path / "lda_all.pkl")

    #### Lambda
    # - **$\lambda$ = 0**: how probable is a word to appear in a topic - words are ranked on lift
    # P(word | topic) / P(word)
    # - **$\lambda$ = 1**: how exclusive is a word to a topic -  words are purely ranked on P(word | topic)
    # The ranking formula is $\lambda * P(\text{word} \vert \text{topic}) + (1 - \lambda) * \text{lift}$
    # User studies suggest $\lambda = 0.6$ works for most people.

    prepare(lda_all, dtm, vectorizer)

    ## Topics as WordClouds
    topics_prob = lda_all.components_ / lda_all.components_.sum(axis=1).reshape(-1, 1)
    topics = pd.DataFrame(
        topics_prob.T, index=vectorizer.get_feature_names_out(), columns=topic_labels
    )

    w = WordCloud()
    fig, axes = plt.subplots(nrows=5, figsize=(15, 30))
    axes = axes.flatten()
    for t, (topic, freq) in enumerate(topics.items()):
        w.generate_from_frequencies(freq.to_dict())
        axes[t].imshow(w, interpolation="bilinear")
        axes[t].set_title(topic, fontsize=18)
        axes[t].axis("off")
    plt.savefig("images/04-07.png")

    ### Visualize topic-word assocations per document
    dtm_ = pd.DataFrame(data=lda_all.transform(dtm), columns=topic_labels, index=docs.topic)
    print(dtm_.head())

    color_dict = OrderedDict()
    color_dict["Topic 1"] = {"color": "white", "on_color": "on_blue"}
    color_dict["Topic 2"] = {"color": "white", "on_color": "on_green"}
    color_dict["Topic 3"] = {"color": "white", "on_color": "on_red"}
    color_dict["Topic 4"] = {"color": "white", "on_color": "on_magenta"}
    color_dict["Topic 5"] = {"color": "blue", "on_color": "on_yellow"}

    dtm_["article"] = docs.article.values
    dtm_["heading"] = docs.heading.values
    sample = dtm_[dtm_[topic_labels].gt(0.05).all(1)]
    print(sample)

    colored_text = []
    for word in sample.iloc[0, 5].split():
        try:
            topic = topics.loc[word.strip().lower()].idxmax()
            colored_text.append(colored(word, **color_dict[topic]))
        except:
            colored_text.append(word)

    print(" ".join([colored(k, **v) for k, v in color_dict.items()]))
    print("\n", sample.iloc[0, 6], "\n")
    text = " ".join(colored_text)
    print(text)
