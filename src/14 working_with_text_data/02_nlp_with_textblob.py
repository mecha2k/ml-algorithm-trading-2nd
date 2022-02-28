# NLP with TextBlob is a python library that provides a simple API for common NLP tasks and builds on the Natural Language
# Toolkit (nltk) and the Pattern web mining libraries. TextBlob facilitates part-of-speech tagging, noun phrase
# extraction, sentiment analysis, classification, translation, and others.

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

import nltk
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

from icecream import ic

np.random.seed(42)
sns.set_style("white")

results_path = Path("../data/ch14", "bbc")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    # download NLTK resources
    nltk.download("punkt", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("wordnet", quiet=True)

    # Load BBC Data
    # To illustrate the use of TextBlob, we sample a BBC sports article with the headline ‘Robinson ready for difficult
    # task’. Similar to spaCy and other libraries, the first step is to pass the document through a pipeline represented
    # by the TextBlob object to assign annotations required for various tasks.
    # path = Path("..", "data", "bbc")
    # files = sorted(list(path.glob("**/*.txt")))
    # doc_list = []
    # for i, file in enumerate(files):
    #     topic = file.parts[-2]
    #     article = file.read_text(encoding="latin1").split("\n")
    #     heading = article[0].strip()
    #     body = " ".join([l.strip() for l in article[1:]]).strip()
    #     doc_list.append([topic, heading, body])
    # docs = pd.DataFrame(doc_list, columns=["topic", "heading", "body"])
    # docs.to_pickle(results_path / "bbc_v37.pkl")

    docs = pd.read_pickle(results_path / "bbc_v37.pkl")
    docs.info()
    ic(docs.shape)

    ## Introduction to TextBlob
    # You should already have downloaded TextBlob, a Python library used to explore common NLP tasks.

    ### Select random article
    article = docs.sample(n=1).squeeze()
    ic(f"Topic:\t{article.topic.capitalize()}\n{article.heading}")
    ic(article.body.strip())
    parsed_body = TextBlob(article.body)

    ### Tokenization
    ic(parsed_body.words)

    ### Sentence boundary detection
    ic(parsed_body.sentences)

    ### Stemming
    # To perform stemming, we instantiate the SnowballStemmer from the nltk library, call its .stem() method on each
    # token and display tokens that were modified as a result:

    # Initialize stemmer.
    stemmer = SnowballStemmer("english")

    # Stem each word.
    ic(
        [
            (word, stemmer.stem(word))
            for i, word in enumerate(parsed_body.words)
            if word.lower() != stemmer.stem(parsed_body.words[i])
        ]
    )

    ### Lemmatization
    ic(
        [
            (word, word.lemmatize())
            for i, word in enumerate(parsed_body.words)
            if word != parsed_body.words[i].lemmatize()
        ]
    )

    # Lemmatization relies on parts-of-speech (POS) tagging; `spaCy` performs POS tagging, here we make assumptions,
    # e.g. that each token is verb.
    ic(
        [
            (word, word.lemmatize(pos="v"))
            for i, word in enumerate(parsed_body.words)
            if word != parsed_body.words[i].lemmatize(pos="v")
        ]
    )

    ### Sentiment & Polarity
    # TextBlob provides polarity and subjectivity estimates for parsed documents using dictionaries provided by the
    # Pattern library. These dictionaries' lexicon map adjectives frequently found in product reviews to sentiment
    # polarity scores, ranging from -1 to +1 (negative ↔ positive) and a similar subjectivity score
    # (objective ↔ subjective).
    # The .sentiment attribute provides the average for each over the relevant tokens, whereas
    # the .sentiment_assessments attribute lists the underlying values for each token
    ic(parsed_body.sentiment)
    ic(parsed_body.sentiment_assessments)

    ### Combine Textblob Lemmatization with `CountVectorizer`
    def lemmatizer(text):
        words = TextBlob(text.lower()).words
        return [word.lemmatize() for word in words]

    vectorizer = CountVectorizer(analyzer=lemmatizer, decode_error="replace")
