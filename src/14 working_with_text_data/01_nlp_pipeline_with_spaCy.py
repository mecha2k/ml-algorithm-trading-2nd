# NLP Pipeline with spaCy
# [spaCy](https://spacy.io/) is a widely used python library with a comprehensive feature set for fast text processing
# in multiple languages.
# The usage of the tokenization and annotation engines requires the installation of language models. The features we
# will use in this chapter only require the small models, the larger models also include word vectors that we will
# cover in chapter 15.
# ![spaCy](assets/spacy.jpg)

import sys
from pathlib import Path
import pandas as pd

import spacy
from spacy import displacy
from textacy.extract import ngrams, entities
import warnings

warnings.filterwarnings("ignore")

### SpaCy Language Model Installation
# In addition to the `spaCy` library, we need [language models](https://spacy.io/usage/models).

#### English
# Only need to run once.
# %%bash
# python3 -m spacy download en_core_web_sm

#### Spanish
# [Spanish language models](https://spacy.io/models/es#es_core_news_sm) trained on [AnCora Corpus]
# (http://clic.ub.edu/corpus/) and [WikiNER](http://schwa.org/projects/resources/wiki/Wikiner)
# Only need to run once.
# python3 -m spacy download es_core_news_sm

# Create shortcut names (deprecated...)
# python3 -m spacy link en_core_web_sm en --force
# python3 -m spacy link es_core_news_sm es --force

#### Validate Installation
# python3 -m spacy validate


# - [BBC Articles](http://mlg.ucd.ie/datasets/bbc.html), use raw text files ([download]
#       (http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip))
#     - Data already included in [data](../data) directory, just unzip before first-time use.
# - [TED2013](http://opus.nlpl.eu/TED2013.php), a parallel corpus of TED talk subtitles in 15 langugages
#       (sample provided) in `results/TED` subfolder of this directory.

## SpaCy Pipeline & Architecture
### The Processing Pipeline
# When you call a spaCy model on a text, spaCy
# 1) tokenizes the text to produce a `Doc` object.
# 2) passes the `Doc` object through the processing pipeline that may be customized and for the default models
#    consists of
# - a tagger,
# - a parser and
# - an entity recognizer.
# Each pipeline component returns the processed Doc, which is then passed on to the next component.
# ![Architecture](assets/pipeline.svg)

### Key Data Structures
# The central data structures in spaCy are the **Doc** and the **Vocab**. Text annotations are also designed to allow
# a single source of truth:
# - The **`Doc`** object owns the sequence of tokens and all their annotations. `Span` and `Token` are views that point
#   into it. It is constructed by the `Tokenizer`, and then modified in place by the components of the pipeline.
# - The **`Vocab`** object owns a set of look-up tables that make common information available across documents.
# - The **`Language`** object coordinates these components. It takes raw text and sends it through the pipeline,
#   returning an annotated document. It also orchestrates training and serialization.
# ![Architecture](assets/spaCy-architecture.svg)

## SpaCy in Action
### Create & Explore the Language Object
# Once installed and linked, we can instantiate a spaCy language model and then call it on a document. As a result,
# spaCy produces a Doc object that tokenizes the text and processes it according to configurable pipeline components
# that by default consist of a tagger, a parser, and a named-entity recognizer.

DATA_DIR = Path("..", "data")

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    print(type(nlp))
    print(nlp.lang)
    spacy.info("en_core_web_sm")

    def get_attributes(f):
        print([a for a in dir(f) if not a.startswith("_")], end=" ")

    get_attributes(nlp)

    ### Explore the Pipeline
    # Let’s illustrate the pipeline using a simple sentence:
    sample_text = "Apple is looking at buying U.K. startup for $1 billion"
    doc = nlp(sample_text)
    get_attributes(doc)

    print(doc.is_parsed)
    print(doc.is_sentenced)
    print(doc.is_tagged)
    print(doc.text)

    get_attributes(doc.vocab)
    print(doc.vocab.length)

    #### Explore `Token` annotations
    # The parsed document content is iterable and each element has numerous attributes produced by the processing
    # pipeline. The below sample illustrates how to access the following attributes:
    print(pd.Series([token.text for token in doc]))
    print(
        pd.DataFrame(
            [
                [t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.shape_, t.is_alpha, t.is_stop]
                for t in doc
            ],
            columns=["text", "lemma", "pos", "tag", "dep", "shape", "is_alpha", "is_stop"],
        )
    )

    #### Visualize POS Dependencies
    # We can visualize the syntactic dependency in a browser or notebook
    options = {
        "compact": True,
        "bg": "white",
        "color": "black",
        "font": "Source Sans Pro",
        "notebook": True,
    }

    # #### Visualize Named Entities
    # displacy.render(doc, style="dep", options=options)
    # displacy.render(doc, style="ent", jupyter=True)

    ### Read BBC Data
    # We will now read a larger set of 2,225 BBC News articles (see GitHub for data source details) that belong to five
    # categories and are stored in individual text files. We
    # - call the .glob() method of the pathlib’s Path object,
    # - iterate over the resulting list of paths,
    # - read all lines of the news article excluding the heading in the first line, and
    # - append the cleaned result to a list

    files = (DATA_DIR / "bbc").glob("**/*.txt")
    bbc_articles = []
    for i, file in enumerate(sorted(list(files))):
        with file.open(encoding="latin1") as f:
            lines = f.readlines()
            body = " ".join([l.strip() for l in lines[1:]]).strip()
            bbc_articles.append(body)
    print(len(bbc_articles))
    print(bbc_articles[0])

    ### Parse first article through Pipeline
    print(nlp.pipe_names)

    doc = nlp(bbc_articles[0])
    print(type(doc))

    ### Detect sentence boundary
    # Sentence boundaries are calculated from the syntactic parse tree, so features such as punctuation and
    # capitalisation play an important but non-decisive role in determining the sentence boundaries.
    # Usually this means that the sentence boundaries will at least coincide with clause boundaries, even given poorly
    # punctuated text.
    # spaCy computes sentence boundaries from the syntactic parse tree so that punctuation and capitalisation play an
    # important but not decisive role. As a result, boundaries will coincide with clause boundaries, even for poorly
    # punctuated text.
    # We can access the parsed sentences using the .sents attribute:

    sentences = [s for s in doc.sents]
    print(sentences[:3])
    get_attributes(sentences[0])
    print(
        pd.DataFrame(
            [[t.text, t.pos_, spacy.explain(t.pos_)] for t in sentences[0]],
            columns=["Token", "POS Tag", "Meaning"],
        ).head(15)
    )

    options = {"compact": True, "bg": "#09a3d5", "color": "white", "font": "Source Sans Pro"}
    # displacy.render(sentences[0].as_doc(), style="dep", jupyter=True, options=options)

    for t in sentences[0]:
        if t.ent_type_:
            print("{} | {} | {}".format(t.text, t.ent_type_, spacy.explain(t.ent_type_)))
    # displacy.render(sentences[0].as_doc(), style="ent", jupyter=True)

    ### Named Entity-Recognition with textacy
    # spaCy enables named entity recognition using the .ent_type_ attribute:
    # Textacy makes access to the named entities that appear in the first article easy:

    entities = [e.text for e in entities(doc)]
    print(pd.Series(entities).value_counts().head())

    ### N-Grams with textacy
    # N-grams combine N consecutive tokens. This can be useful for the bag-of-words model because, depending on the
    # textual context, treating, e.g, ‘data scientist’ as a single token may be more meaningful than the two distinct
    # tokens ‘data’ and ‘scientist’.
    # Textacy makes it easy to view the ngrams of a given length n occurring with at least min_freq times:
    print(pd.Series([n.text for n in ngrams(doc, n=2, min_freq=2)]).value_counts())

    ### The spaCy streaming Pipeline API
    # To pass a larger number of documents through the processing pipeline, we can use spaCy’s streaming API as follows:
    iter_texts = (bbc_articles[i] for i in range(len(bbc_articles)))
    for i, doc in enumerate(nlp.pipe(iter_texts, batch_size=50, n_process=-1)):
        if i % 100 == 0:
            print(i, end=" ")
        assert doc.is_parsed

    ### Multi-language Features
    # spaCy includes trained language models for English, German, Spanish, Portuguese, French, Italian and Dutch,
    # as well as a multi-language model for named-entity recognition. Cross-language usage is straightforward since
    # the API does not change.
    # We will illustrate the Spanish language model using a parallel corpus of TED talk subtitles. For this purpose,
    # we instantiate both language models

    #### Create a Spanish Language Object
    model = {}
    for language in ["en_core_web_sm", "es_core_news_sm"]:
        model[language] = spacy.load(language)

    #### Read bilingual TED2013 samples
    text = {}
    path = Path("data", "TED")
    filemap = {"en_core_web_sm": "en", "es_core_news_sm": "es"}
    for language in ["en_core_web_sm", "es_core_news_sm"]:
        file_name = path / "TED2013_sample.{}".format(filemap[language])
        text[language] = file_name.read_text()

    #### Sentence Boundaries English vs Spanish
    parsed, sentences = {}, {}
    for language in ["en_core_web_sm", "es_core_news_sm"]:
        parsed[language] = model[language](text[language])
        sentences[language] = list(parsed[language].sents)
        print("Sentences:", language, len(sentences[language]))

    for i, (en, es) in enumerate(zip(sentences["en_core_web_sm"], sentences["es_core_news_sm"]), 1):
        print("\n", i)
        print("English:\t", en)
        print("Spanish:\t", es)
        if i > 5:
            break

    #### POS Tagging English vs Spanish
    pos = {}
    for language in ["en_core_web_sm", "es_core_news_sm"]:
        pos[language] = pd.DataFrame(
            [[t.text, t.pos_, spacy.explain(t.pos_)] for t in sentences[language][0]],
            columns=["Token", "POS Tag", "Meaning"],
        )

    bilingual_parsed = pd.concat([pos["en_core_web_sm"], pos["es_core_news_sm"]], axis=1)
    print(bilingual_parsed.head(15))
    # displacy.render(sentences["es_core_news_sm"][0].as_doc(), style="dep", jupyter=True, options=options)
