# From tokens to numbers: the document-term matrix
# The bag of words models represents a document based on the frequency of the terms or tokens it contains. Each document
# becomes a vector with one entry for each token in the vocabulary that reflects the token’s relevance to the document.
# The document-term matrix is straightforward to compute given the vocabulary. However, it is also a crude
# simplification because it abstracts from word order and grammatical relationships. Nonetheless, it often achieves
# good results in text classification quickly and, thus, a very useful starting point.
# There are several ways to weigh a token’s vector entry to capture its relevance to the document. We will illustrate
# below how to use sklearn to use binary flags that indicate presence or absence, counts, and weighted counts that
# account for differences in term frequencies across all documents, i.e., in the corpus.

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from ipywidgets import interact, FloatRangeSlider

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

idx = pd.IndexSlice
np.random.seed(seed=42)
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format

results_path = Path("../data/ch14", "bbc")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    # ## Load BBC data
    # path = Path("..", "data", "bbc")
    # files = sorted(list(path.glob("**/*.txt")))
    # doc_list = []
    # for i, file in enumerate(files):
    #     topic = file.parts[-2]
    #     article = file.read_text(encoding="latin1").split("\n")
    #     heading = article[0].strip()
    #     body = " ".join([l.strip() for l in article[1:]]).strip()
    #     doc_list.append([topic, heading, body])
    # ### Convert to DataFrame
    # docs = pd.DataFrame(doc_list, columns=["topic", "heading", "body"])
    # docs.info()

    ### Inspect results
    docs = pd.read_pickle(results_path / "bbc_v37.pkl")
    print(docs.sample(10))

    ### Data drawn from 5 different categories
    print(
        docs.topic.value_counts(normalize=True)
        .to_frame("count")
        .style.format({"count": "{:,.2%}".format})
    )

    # ## Explore Corpus
    # ### Token Count via Counter()
    # word_count = docs.body.str.split().str.len().sum()
    # print(f"Total word count: {word_count:,d} | per article: {word_count/len(docs):,.0f}")
    #
    # token_count = Counter()
    # for i, doc in enumerate(docs.body.tolist(), 1):
    #     if i % 500 == 0:
    #         print(i, end=" ", flush=True)
    #     token_count.update([t.strip() for t in doc.split()])
    #
    # tokens = (
    #     pd.DataFrame(token_count.most_common(), columns=["token", "count"])
    #     .set_index("token")
    #     .squeeze()
    # )
    #
    # n = 50
    # tokens.iloc[:50].plot.bar(
    #     figsize=(14, 4), title=f"Most frequent {n} of {len(tokens):,d} tokens"
    # )
    # sns.despine()
    # plt.tight_layout()
    # plt.savefig("images/03-01.png")
    #
    # ## Document-Term Matrix with `CountVectorizer`
    # # The scikit-learn preprocessing module offers two tools to create a document-term matrix. The [CountVectorizer]
    # # (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) uses
    # # binary or absolute counts to measure the term frequency tf(d, t) for each document d and token t.
    # #
    # # The [TfIDFVectorizer]
    # # (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
    # # in contrast, weighs the (absolute) term frequency by the inverse document frequency (idf). As a result, a term
    # # that appears in more documents will receive a lower weight than a token with the same frequency for a given
    # # document but lower frequency across all documents.
    # #
    # # The resulting tf-idf vectors for each document are normalized with respect to their absolute or squared totals
    # # (see the sklearn documentation for details). The tf-idf measure was originally used in information retrieval to
    # # rank search engine results and has subsequently proven useful for text classification or clustering.
    #
    # # Both tools use the same interface and perform tokenization and further optional preprocessing of a list of
    # # documents before vectorizing the text by generating token counts to populate the document-term matrix.
    # #
    # # Key parameters that affect the size of the vocabulary include:
    # #
    # # - `stop_words`: use a built-in or provide a list of (frequent) words to exclude
    # # - `ngram_range`: include n-grams in a range for n defined by a tuple of (nmin, nmax)
    # # - `lowercase`: convert characters accordingly (default is True)
    # # - `min_df `/ max_df: ignore words that appear in less / more (int) or a smaller / larger share of documents
    # #    (if float [0.0,1.0])
    # # - `max_features`: limit number of tokens in vocabulary accordingly
    # # - `binary`: set non-zero counts to 1 True
    #
    # ### Key parameters
    # print(CountVectorizer().__doc__)
    #
    # ### Document Frequency Distribution
    # binary_vectorizer = CountVectorizer(max_df=1.0, min_df=1, binary=True)
    # binary_dtm = binary_vectorizer.fit_transform(docs.body)
    # print(binary_dtm)
    #
    # n_docs, n_tokens = binary_dtm.shape
    # tokens_dtm = binary_vectorizer.get_feature_names_out()
    #
    # #### CountVectorizer skips certain tokens by default
    # print(tokens.index.difference(pd.Index(tokens_dtm)))
    #
    # #### Persist Result
    # dtm_path = results_path / "binary_dtm.npz"
    # if not dtm_path.exists():
    #     sparse.save_npz(dtm_path, binary_dtm)
    #
    # token_path = results_path / "tokens.csv"
    # if not token_path.exists():
    #     pd.Series(tokens_dtm).to_csv(token_path, index=False)
    # else:
    #     tokens = pd.read_csv(token_path, header=None, squeeze=True)
    #
    # doc_freq = pd.Series(np.array(binary_dtm.sum(axis=0)).squeeze()).div(n_docs)
    # max_unique_tokens = np.array(binary_dtm.sum(axis=1)).squeeze().max()
    #
    # ### `min_df` vs `max_df`: Interactive Visualization
    # # The notebook contains an interactive visualization that explores the impact of the min_df and max_df settings
    # # on the size of the vocabulary. We read the articles into a DataFrame, set the CountVectorizer to produce binary
    # # flags and use all tokens, and call its .fit_transform() method to produce a document-term matrix:
    #
    # # The visualization shows that requiring tokens to appear in at least 1%  and less than 50% of documents restricts
    # # the vocabulary to around 10% of the almost 30K tokens.
    # # This leaves a mode of slightly over 100 unique tokens per document (left panel), and the right panel shows the
    # # document frequency histogram for the remaining tokens.
    #
    # df_range = FloatRangeSlider(
    #     value=[0.0, 1.0],
    #     min=0,
    #     max=1,
    #     step=0.0001,
    #     description="Doc. Freq.",
    #     disabled=False,
    #     continuous_update=True,
    #     orientation="horizontal",
    #     readout=True,
    #     readout_format=".1%",
    #     layout={"width": "800px"},
    # )
    #
    # @interact(df_range=df_range)
    # def document_frequency_simulator(df_range):
    #     min_df, max_df = df_range
    #     keep = doc_freq.between(left=min_df, right=max_df)
    #     left = keep.sum()
    #
    #     fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
    #
    #     updated_dtm = binary_dtm.tocsc()[:, np.flatnonzero(keep)]
    #     unique_tokens_per_doc = np.array(updated_dtm.sum(axis=1)).squeeze()
    #     sns.distplot(unique_tokens_per_doc, ax=axes[0], kde=False, norm_hist=False)
    #     axes[0].set_title("Unique Tokens per Doc")
    #     axes[0].set_yscale("log")
    #     axes[0].set_xlabel("# Unique Tokens")
    #     axes[0].set_ylabel("# Documents (log scale)")
    #     axes[0].set_xlim(0, max_unique_tokens)
    #     axes[0].yaxis.set_major_formatter(ScalarFormatter())
    #
    #     term_freq = pd.Series(np.array(updated_dtm.sum(axis=0)).squeeze())
    #     sns.distplot(term_freq, ax=axes[1], kde=False, norm_hist=False)
    #     axes[1].set_title("Document Frequency")
    #     axes[1].set_ylabel("# Tokens")
    #     axes[1].set_xlabel("# Documents")
    #     axes[1].set_yscale("log")
    #     axes[1].set_xlim(0, n_docs)
    #     axes[1].yaxis.set_major_formatter(ScalarFormatter())
    #
    #     title = f"Document/Term Frequency Distribution | # Tokens: {left:,d} ({left/n_tokens:.2%})"
    #     fig.suptitle(title, fontsize=14)
    #     sns.despine()
    #     fig.tight_layout()
    #     fig.subplots_adjust(top=0.9)
    #
    # ### Most similar documents
    # # The CountVectorizer result lets us find the most similar documents using the `pdist()` function for pairwise
    # # distances provided by the `scipy.spatial.distance` module.
    # # It returns a  condensed distance matrix with entries corresponding to the upper triangle of a square matrix.
    # # We use `np.triu_indices()` to translate the index that minimizes the distance to the row and column indices that
    # # in turn correspond to the closest token vectors.
    # m = binary_dtm.todense()
    # pairwise_distances = pdist(m, metric="cosine")
    #
    # closest = np.argmin(pairwise_distances)
    # rows, cols = np.triu_indices(n_docs)
    # print(rows[closest], cols[closest])
    #
    # docs.iloc[6].to_frame(6).join(docs.iloc[245].to_frame(245)).to_csv(
    #     results_path / "most_similar.csv"
    # )
    # print(docs.iloc[6])
    # print(pd.DataFrame(binary_dtm[[6, 245], :].todense()).sum(0).value_counts())
    #
    # ### Baseline document-term matrix
    # # Baseline: number of unique tokens
    # vectorizer = CountVectorizer()  # default: binary=False
    # doc_term_matrix = vectorizer.fit_transform(docs.body)
    # print(doc_term_matrix)
    # print(doc_term_matrix.shape)
    #
    # ### Inspect tokens
    # # vectorizer keeps words
    # words = vectorizer.get_feature_names_out()
    # print(words[:10])
    #
    # ### Inspect doc-term matrix
    # # from scipy compressed sparse row matrix to sparse DataFrame
    # doc_term_matrix_df = pd.DataFrame.sparse.from_spmatrix(doc_term_matrix, columns=words)
    # print(doc_term_matrix_df.head())
    #
    # ### Most frequent terms
    # word_freq = doc_term_matrix_df.sum(axis=0).astype(int)
    # print(word_freq.sort_values(ascending=False).head())
    #
    # ### Compute relative term frequency
    # vectorizer = CountVectorizer(binary=True)
    # doc_term_matrix = vectorizer.fit_transform(docs.body)
    # print(doc_term_matrix.shape)
    #
    # words = vectorizer.get_feature_names_out()
    # word_freq = doc_term_matrix.sum(axis=0)
    #
    # # reduce to 1D array
    # word_freq_1d = np.squeeze(np.asarray(word_freq))
    # print(
    #     pd.Series(word_freq_1d, index=words)
    #     .div(docs.shape[0])
    #     .sort_values(ascending=False)
    #     .head(10)
    # )
    #
    # ### Visualize Doc-Term Matrix
    # sns.heatmap(pd.DataFrame(doc_term_matrix.todense(), columns=words), cmap="Blues")
    # plt.gcf().set_size_inches(14, 8)
    # plt.savefig("images/03-02.png")
    #
    # ### Using thresholds to reduce the number of tokens
    # vectorizer = CountVectorizer(max_df=0.2, min_df=3, stop_words="english")
    # doc_term_matrix = vectorizer.fit_transform(docs.body)
    # print(doc_term_matrix.shape)
    #
    # ### Use CountVectorizer with Lemmatization
    # #### Building a custom `tokenizer` for Lemmatization with `spacy`
    # nlp = spacy.load("en_core_web_sm")
    #
    # def tokenizer(doc):
    #     return [w.lemma_ for w in nlp(doc) if not w.is_punct | w.is_space]
    #
    # vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True)
    # doc_term_matrix = vectorizer.fit_transform(docs.body)
    # print(doc_term_matrix.shape)
    #
    # lemmatized_words = vectorizer.get_feature_names_out()
    # word_freq = doc_term_matrix.sum(axis=0)
    # word_freq_1d = np.squeeze(np.asarray(word_freq))
    # word_freq_1d = pd.Series(word_freq_1d, index=lemmatized_words).div(docs.shape[0])
    # print(word_freq_1d.sort_values().tail(20))
    #
    # # Unlike verbs and common nouns, there's no clear base form of a personal pronoun. Should the lemma of "me" be "I",
    # # or should we normalize person as well, giving "it" — or maybe "he"? spaCy's solution is to introduce a novel
    # # symbol, -PRON-, which is used as the lemma for all personal pronouns.

    ## Document-Term Matrix with `TfIDFVectorizer`
    # The TfIDFTransfomer computes the tf-idf weights from a document-term matrix of token counts like the one produced
    # by the CountVectorizer.
    # The TfIDFVectorizer performs both computations in a single step. It adds a few parameters to the CountVectorizer
    # API that controls the smoothing behavior.

    ### Key Parameters
    # The `TfIDFTransformer` builds on the `CountVectorizer` output; the `TfIDFVectorizer` integrates both
    print(TfidfTransformer().__doc__)

    ### How Term Frequency - Inverse Document Frequency works
    # The TFIDF computation works as follows for a small text sample
    sample_docs = ["call you tomorrow", "Call me a taxi", "please call me... PLEASE!"]

    #### Compute term frequency
    vectorizer = CountVectorizer()
    tf_dtm = vectorizer.fit_transform(sample_docs).todense()
    tokens = vectorizer.get_feature_names_out()
    term_frequency = pd.DataFrame(data=tf_dtm, columns=tokens)
    print(term_frequency)

    #### Compute document frequency
    vectorizer = CountVectorizer(binary=True)
    df_dtm = vectorizer.fit_transform(sample_docs).todense().sum(axis=0)
    document_frequency = pd.DataFrame(data=df_dtm, columns=tokens)
    print(document_frequency)

    #### Compute TfIDF
    tfidf = pd.DataFrame(data=tf_dtm / df_dtm, columns=tokens)
    print(tfidf)

    #### The effect of smoothing
    # The TfidfVectorizer uses smoothing for document and term frequencies:
    # - `smooth_idf`: add one to document frequency, as if an extra document contained every token in the vocabulary
    #      once to prevents zero divisions
    # - `sublinear_tf`: scale term Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)

    vect = TfidfVectorizer(
        smooth_idf=True,
        norm="l2",  # squared weights sum to 1 by document
        sublinear_tf=False,  # if True, use 1+log(tf)
        binary=False,
    )
    print(
        pd.DataFrame(
            vect.fit_transform(sample_docs).todense(), columns=vect.get_feature_names_out()
        )
    )

    ### TfIDF with new articles
    # Due to their ability to assign meaningful token weights, TFIDF vectors are also used to summarize text data.
    # E.g., reddit's autotldr function is based on a similar algorithm.
    tfidf = TfidfVectorizer(stop_words="english")
    dtm_tfidf = tfidf.fit_transform(docs.body)
    tokens = tfidf.get_feature_names_out()
    print(dtm_tfidf.shape)

    token_freq = pd.DataFrame({"tfidf": dtm_tfidf.sum(axis=0).A1, "token": tokens}).sort_values(
        "tfidf", ascending=False
    )
    print(token_freq.head(10).append(token_freq.tail(10)).set_index("token"))

    ### Summarizing news articles using TfIDF weights
    #### Select random article
    article = docs.sample(1).squeeze()
    article_id = article.name
    print(f"Topic:\t{article.topic.capitalize()}\n\n{article.heading}\n")
    print(article.body.strip())

    #### Select most relevant tokens by tfidf value
    article_tfidf = dtm_tfidf[article_id].todense().A1
    article_tokens = pd.Series(article_tfidf, index=tokens)
    print(article_tokens.sort_values(ascending=False).head(10))

    #### Compare to random selection
    print(pd.Series(article.body.split()).sample(10).tolist())

    ## Create Train & Test Sets
    ### Stratified `train_test_split`
    train_docs, test_docs = train_test_split(
        docs, stratify=docs.topic, test_size=50, random_state=42
    )
    print(train_docs.shape, test_docs.shape)
    print(pd.Series(test_docs.topic).value_counts())

    ### Vectorize train & test sets
    vectorizer = CountVectorizer(max_df=0.2, min_df=3, stop_words="english", max_features=2000)

    train_dtm = vectorizer.fit_transform(train_docs.body)
    words = vectorizer.get_feature_names_out()
    print(train_dtm)

    test_dtm = vectorizer.transform(test_docs.body)
    print(test_dtm)
