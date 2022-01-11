# Topic Modeling: Latent Dirichlet Allocation with gensim is a specialized NLP library with a fast LDA implementation and many additional features. We will also use it
# in the next chapter on word vectors (see the notebook lda_with_gensim for details.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyLDAvis

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

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
    # print(train_dtm)

    test_dtm = vectorizer.transform(test_docs.article)
    # print(test_dtm)

    ## LDA with gensim
    ### Using `CountVectorizer` Input
    max_df = 0.2
    min_df = 3
    max_features = 2000

    # used by sklearn: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
    # stop_words = pd.read_csv(
    #     "http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words", header=None, squeeze=True
    # )
    # stop_words.to_csv(results_path / "stop_words.csv")
    stop_words = pd.read_csv(results_path / "stop_words.csv").values

    vectorizer = CountVectorizer(
        max_df=max_df, min_df=min_df, stop_words="english", max_features=max_features
    )

    train_dtm = vectorizer.fit_transform(train_docs.article)
    test_dtm = vectorizer.transform(test_docs.article)

    ### Convert sklearn DTM to gensim data structures
    # It faciltiates the conversion of DTM produced by sklearn to gensim data structures as follows:
    train_corpus = Sparse2Corpus(train_dtm, documents_columns=False)
    test_corpus = Sparse2Corpus(test_dtm, documents_columns=False)
    id2word = pd.Series(vectorizer.get_feature_names_out()).to_dict()

    ### Train Model & Review Results
    LdaModel(
        corpus=train_corpus,
        num_topics=100,
        id2word=None,
        distributed=False,
        chunksize=2000,  # Number of documents to be used in each training chunk.
        passes=1,  # Number of passes through the corpus during training
        update_every=1,  # Number of docs to be iterated through for each update
        alpha="symmetric",
        eta=None,  # a-priori belief on word probability
        decay=0.5,  # percentage of previous lambda forgotten when new document is examined
        offset=1.0,  # controls slow down of the first steps the first few iterations.
        eval_every=10,  # estimate log perplexity
        iterations=50,  # Maximum number of iterations through the corpus
        gamma_threshold=0.001,  # Minimum change in the value of the gamma parameters to continue iterating
        minimum_probability=0.01,  # Topics with a probability lower than this threshold will be filtered out
        random_state=None,
        ns_conf=None,
        minimum_phi_value=0.01,  # if `per_word_topics` is True, represents lower bound on term probabilities
        per_word_topics=False,  #  If True, compute a list of most likely topics for each word with phi values multiplied by word count
        callbacks=None,
    )

    num_topics = 5
    topic_labels = ["Topic {}".format(i) for i in range(1, num_topics + 1)]

    lda_gensim = LdaModel(corpus=train_corpus, num_topics=num_topics, id2word=id2word)

    topics = lda_gensim.print_topics()
    print(topics[0])

    ### Evaluate Topic Coherence measures whether the words in a topic tend to co-occur together.
    # - It adds up a score for each distinct pair of top ranked words.
    # - The score is the log of the probability that a document containing at least one instance of the higher-ranked
    #   word also contains at least one instance of the lower-ranked word.
    # Large negative values indicate words that don't co-occur often; values closer to zero indicate that words tend
    # to co-occur more often.
    coherence = lda_gensim.top_topics(corpus=train_corpus, coherence="u_mass")

    # Gensim permits topic coherence evaluation that produces the topic coherence and shows the most important
    # words per topic:
    topic_coherence = []
    topic_words = pd.DataFrame()
    for t in range(len(coherence)):
        label = topic_labels[t]
        topic_coherence.append(coherence[t][1])
        df = pd.DataFrame(coherence[t][0], columns=[(label, "prob"), (label, "term")])
        df[(label, "prob")] = df[(label, "prob")].apply(lambda x: "{:.2%}".format(x))
        topic_words = pd.concat([topic_words, df], axis=1)

    topic_words.columns = pd.MultiIndex.from_tuples(topic_words.columns)
    pd.set_option("expand_frame_repr", False)
    topic_words.head().to_csv("../data/ch15/results/topic_words.csv", index=False)
    print(topic_words.head())

    pd.Series(topic_coherence, index=topic_labels).plot.bar(figsize=(12, 4))
    plt.savefig("images/05-01.png")

    ### Using `gensim` `Dictionary`
    docs = [d.split() for d in train_docs.article.tolist()]
    docs = [[t for t in doc if t not in stop_words] for doc in docs]

    dictionary = Dictionary(docs)
    print(dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=max_features))
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print("Number of unique tokens: %d" % len(dictionary))
    print("Number of documents: %d" % len(corpus))

    num_topics = 5
    chunksize = 500
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha="auto",
        eta="auto",
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every,
    )
    model.show_topics()

    ### Evaluating Topic Assignments on the Test Set
    docs_test = [d.split() for d in test_docs.article.tolist()]
    docs_test = [[t for t in doc if t not in stop_words] for doc in docs_test]

    test_dictionary = Dictionary(docs_test)
    test_dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=max_features)
    test_corpus = [dictionary.doc2bow(doc) for doc in docs_test]

    gamma, _ = model.inference(test_corpus)
    topic_scores = pd.DataFrame(gamma)
    print(topic_scores.head(10))

    topic_probabilities = topic_scores.div(topic_scores.sum(axis=1), axis=0)
    print(topic_probabilities.head())
    print(topic_probabilities.idxmax(axis=1).head())

    predictions = test_docs.topic.to_frame("topic").assign(
        predicted=topic_probabilities.idxmax(axis=1).values
    )
    heatmap_data = predictions.groupby("topic").predicted.value_counts().unstack()
    sns.heatmap(heatmap_data, annot=True, cmap="Blues")
    plt.savefig("images/05-02.png")

    ## Resources
    # - pyLDAvis:
    #     - [Talk by the Author](https://speakerdeck.com/bmabey/visualizing-topic-models) and [Paper by (original)
    #        Author](http://www.aclweb.org/anthology/W14-3110)
    #     - [Documentation](http://pyldavis.readthedocs.io/en/latest/index.html)
    # - LDA:
    #     - [David Blei Homepage @ Columbia](http://www.cs.columbia.edu/~blei/)
    #     - [Introductory Paper](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf) and
    #       [more technical review paper] (http://www.cs.columbia.edu/~blei/papers/BleiLafferty2009.pdf)
    #     - [Blei Lab @ GitHub](https://github.com/Blei-Lab)
    # - Topic Coherence:
    #     - [Exploring Topic Coherence over many models and many topics]
    #       (https://www.aclweb.org/anthology/D/D12/D12-1087.pdf)
    #     - [Paper on various Methods](http://www.aclweb.org/anthology/N10-1012)
    #     - [Blog Post - Overview](http://qpleple.com/topic-coherence-to-evaluate-topic-models/)
