# Yelp Sentiment Analysis with doc2vec Document Vectors

import nltk

from pathlib import Path
import logging
from random import shuffle

import numpy as np
import pandas as pd

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from nltk import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import class_weight

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
idx = pd.IndexSlice
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.expand_frame_repr", False)

nltk.download("stopwords")

data_path = Path("..", "data", "yelp")

results_path = Path("../data/ch16", "yelp")
if not results_path.exists():
    results_path.mkdir(parents=True)

if __name__ == "__main__":
    logging.basicConfig(
        filename=results_path / "doc2vec.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    ## Load Data
    # Refer to download information [here](../data/create_yelp_review_data.ipynb).
    # We'll create a smaller sample of 100,000 reviews per star rating.
    df = pd.read_parquet(data_path / "user_reviews.parquet").loc[:, ["stars", "text"]]
    df.info()
    df.stars.value_counts()

    stars = range(1, 6)
    sample = pd.concat([df[df.stars == s].sample(n=100000) for s in stars])
    sample.info()
    sample.stars.value_counts()
    sample.to_parquet(results_path / "review_sample.parquet")

    sample = pd.read_parquet(results_path / "review_sample.parquet").reset_index(drop=True)
    sample.head()

    ax = sns.histplot(data=sample.text.str.split().str.len())
    ax.set_xlabel("# Tokens")
    sns.despine()
    plt.savefig("images/08-01")

    ## Doc2Vec
    ### Basic text cleaning
    tokenizer = RegexpTokenizer(r"\w+")
    stopword_set = set(stopwords.words("english"))

    def clean(review):
        tokens = tokenizer.tokenize(review)
        return " ".join([t for t in tokens if t not in stopword_set])

    sample.text = sample.text.str.lower().apply(clean)
    sample.sample(n=10)
    sample = sample[sample.text.str.split().str.len() > 10]
    sample.info()

    ### Create sentence stream
    sentences = []
    for i, (_, text) in enumerate(sample.values):
        sentences.append(TaggedDocument(words=text.split(), tags=[i]))

    ### Formulate the model
    model = Doc2Vec(
        documents=sentences,
        dm=1,  # 1=distributed memory, 0=dist.BOW
        epochs=5,
        vector_size=300,  # vector size
        window=5,  # max. distance betw. target and context
        min_count=50,  # ignore tokens w. lower frequency
        negative=5,  # negative training samples
        dm_concat=0,  # 1=concatenate vectors, 0=sum
        dbow_words=0,  # 1=train word vectors as well
        workers=4,
    )
    pd.DataFrame(model.most_similar("good"), columns=["token", "similarity"])

    ### Continue training
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    most_similar = pd.DataFrame(model.most_similar("good"), columns=["token", "similarity"])
    print(most_similar)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    sns.histplot(data=sample.text.str.split().str.len(), ax=axes[0])
    axes[0].set_title("# Tokens per Review")

    most_similar.set_index("token").similarity.sort_values().plot.barh(
        ax=axes[1], title="Terms Most Similar to 'good'", xlim=(0.5, 0.8)
    )
    axes[1].set_xlabel("Similarity")
    axes[1].set_ylabel("Token")
    axes[0].set_xlabel("# Tokens")

    sns.despine()
    fig.tight_layout()
    fig.savefig("images/08_doc2vec_stats", dpi=300)

    ## Persist Model
    model.save((results_path / "sample.model").as_posix())

    model = Doc2Vec.load((results_path / "sample.model").as_posix())

    ## Evaluate
    y = sample.stars.sub(1)

    size = 300
    X = np.zeros(shape=(len(y), size))
    for i in range(len(sample)):
        X[i] = model.docvecs[i]
    print(X.shape)

    ### Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mode = pd.Series(y_train).mode().iloc[0]
    baseline = accuracy_score(y_true=y_test, y_pred=np.full_like(y_test, fill_value=mode))
    print(f"Baseline Score: {baseline:.2%}")

    class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
    print(class_weights)

    ## LightGBM
    train_data = lgb.Dataset(data=X_train, label=y_train)
    test_data = train_data.create_valid(X_test, label=y_test)

    params = {"objective": "multiclass", "num_classes": 5}
    lgb_model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=5000,
        valid_sets=[train_data, test_data],
        early_stopping_rounds=25,
        verbose_eval=50,
    )
    lgb_pred = np.argmax(lgb_model.predict(X_test), axis=1)
    lgb_acc = accuracy_score(y_true=y_test, y_pred=lgb_pred)
    print(f"Accuracy: {lgb_acc:.2%}")

    ## Random Forest
    rf = RandomForestClassifier(
        n_jobs=-1, n_estimators=500, verbose=1, class_weight="balanced_subsample"
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_true=y_test, y_pred=rf_pred)
    print(f"Accuracy: {rf_acc:.2%}")

    cm = confusion_matrix(y_true=y_test, y_pred=rf_pred)
    sns.heatmap(
        pd.DataFrame(cm / np.sum(cm), index=stars, columns=stars),
        annot=True,
        cmap="Blues",
        fmt=".1%",
    )
    plt.savefig("images/08-02")

    ## Multinomial Logistic Regression
    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", class_weight="balanced")
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_true=y_test, y_pred=lr_pred)
    print(f"Accuracy: {lr_acc:.2%}")

    cm = confusion_matrix(y_true=y_test, y_pred=lr_pred)
    sns.heatmap(
        pd.DataFrame(cm / np.sum(cm), index=stars, columns=stars),
        annot=True,
        cmap="Blues",
        fmt=".1%",
    )
    plt.savefig("images/08-03")

    ## Comparison
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5), sharex=True, sharey=True)
    lgb_cm = confusion_matrix(y_true=y_test, y_pred=lgb_pred)
    sns.heatmap(
        pd.DataFrame(lgb_cm / np.sum(lgb_cm), index=stars, columns=stars),
        annot=True,
        cmap="Blues",
        fmt=".1%",
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title(f"Gradient Boosting: Accuracy {lgb_acc:.2%}")

    rf_cm = confusion_matrix(y_true=y_test, y_pred=rf_pred)
    sns.heatmap(
        pd.DataFrame(rf_cm / np.sum(rf_cm), index=stars, columns=stars),
        annot=True,
        cmap="Blues",
        fmt=".1%",
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title(f"Random Forest: Accuracy {rf_acc:.2%}")

    lr_cm = confusion_matrix(y_true=y_test, y_pred=lr_pred)
    sns.heatmap(
        pd.DataFrame(lr_cm / np.sum(lr_cm), index=stars, columns=stars),
        annot=True,
        cmap="Blues",
        fmt=".1%",
        ax=axes[2],
        cbar=False,
    )
    axes[2].set_title(f"Logistic Regression: Accuracy {lr_acc:.2%}")
    axes[0].set_ylabel("Actuals")
    for i in range(3):
        axes[i].set_xlabel("Predicted")

    sns.despine()
    fig.tight_layout()
    fig.savefig("images/08_confusion_matrix", dpi=300)
