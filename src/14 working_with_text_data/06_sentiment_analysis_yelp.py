# Text classification and sentiment analysis: Yelp Reviews
# Once text data has been converted into numerical features using the natural language processing techniques discussed
# in the previous sections, text classification works just like any other classification task.
# In this notebook, we will apply these preprocessing technique to Yelp business reviews to classify them by review
# scores and sentiment polarity. More specifically, we will apply sentiment analysis to the significantly larger Yelp
# business review dataset with five outcome classes.

from pathlib import Path
import json
from time import time

import numpy as np
import pandas as pd

from scipy import sparse
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format

data_dir = Path("..", "data", "yelp")

yelp_dir = Path("../data/ch14", "yelp")
text_features_dir = yelp_dir / "data"
if not text_features_dir.exists():
    text_features_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    ## Yelp Challenge: business review's dataset
    # Follow the [instructions](../data/create_yelp_review_data.ipynb) to create the dataset.
    yelp_reviews = pd.read_parquet(data_dir / "user_reviews.parquet")
    yelp_reviews.info(show_counts=True)

    # The following figure shows the number of reviews and the average number of stars per year.
    #### Reviews & Stars by Year
    fig, axes = plt.subplots(ncols=3, figsize=(18, 4))
    yelp_reviews.year.value_counts().sort_index().plot.bar(
        title="Reviews per Year", ax=axes[0], rot=0
    )
    sns.lineplot(x="year", y="stars", data=yelp_reviews, ax=axes[1])
    axes[1].set_title("Stars per year")

    stars_dist = yelp_reviews.stars.value_counts(normalize=True).sort_index().mul(100)
    stars_dist.index = stars_dist.index.astype(int)
    stars_dist.plot.barh(title="# Stars Breakdown", ax=axes[2])
    axes[2].set_xlabel("Share of all Ratings (%)")
    axes[2].set_ylabel("Number of Stars")

    sns.despine()
    fig.tight_layout()
    plt.savefig("images/06-01.png")

    #### Years of Membership Breakdown
    ax = (
        yelp_reviews.member_yrs.value_counts()
        .div(1000)
        .sort_index()
        .plot.bar(title="Years of Membership", rot=0)
    )
    ax.set_xlabel("Number of Years")
    ax.set_ylabel("Number of Members  ('000)")
    sns.despine()
    plt.tight_layout()
    plt.savefig("images/06-02.png")

    ### Create train-test split
    train = yelp_reviews[yelp_reviews.year < 2019].sample(frac=0.25)
    test = yelp_reviews[yelp_reviews.year == 2019]
    print(f"# Training Obs: {len(train):,.0f} | # Test Obs: {len(test):,.0f}")
    train.to_parquet(text_features_dir / "train.parquet")
    test.to_parquet(text_features_dir / "test.parquet")

    del yelp_reviews

    #### Reload stored data
    train = pd.read_parquet(text_features_dir / "train.parquet")
    test = pd.read_parquet(text_features_dir / "test.parquet")

    ## Create Yelp review document-term matrix
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
    train_dtm = vectorizer.fit_transform(train.text)
    print(train_dtm)
    sparse.save_npz(text_features_dir / "train_dtm", train_dtm)

    test_dtm = vectorizer.transform(test.text)
    sparse.save_npz(text_features_dir / "test_dtm", test_dtm)

    ### Reload stored data
    train_dtm = sparse.load_npz(text_features_dir / "train_dtm.npz")
    test_dtm = sparse.load_npz(text_features_dir / "test_dtm.npz")

    ## Combine non-text features with the document-term matrix
    # The dataset contains various numerical features. The vectorizers produce [scipy.sparse matrices]
    # (https://docs.scipy.org/doc/scipy/reference/sparse.html). To combine the vectorized text data with other features,
    # we need to first convert these to sparse matrices as well; many sklearn objects and other libraries like lightgbm
    # can handle these very memory-efficient data structures. Converting the sparse matrix to a dense numpy array risks
    # memory overflow.
    # Most variables are categorical, so we use one-hot encoding since we have a fairly large dataset to accommodate the
    # increase in features.
    # We convert the encoded numerical features and combine them with the document-term matrix:

    ### One-hot-encoding
    df = pd.concat(
        [
            train.drop(["text", "stars"], axis=1).assign(source="train"),
            test.drop(["text", "stars"], axis=1).assign(source="test"),
        ]
    )
    uniques = df.nunique()
    binned = pd.concat(
        [
            (
                df.loc[:, uniques[uniques > 20].index].apply(
                    pd.qcut, q=10, labels=False, duplicates="drop"
                )
            ),
            df.loc[:, uniques[uniques <= 20].index],
        ],
        axis=1,
    )
    binned.info(show_counts=True)

    dummies = pd.get_dummies(binned, columns=binned.columns.drop("source"), drop_first=True)
    dummies.info()

    train_dummies = dummies[dummies.source == "train"].drop("source", axis=1)
    train_dummies.info()

    ### Train set
    # Cast other feature columns to float and convert to a sparse matrix.
    train_numeric = sparse.csr_matrix(train_dummies.astype(np.uint8))
    print(train_numeric.shape)

    # Combine sparse matrices.
    train_dtm_numeric = sparse.hstack((train_dtm, train_numeric))
    print(train_dtm_numeric.shape)
    sparse.save_npz(text_features_dir / "train_dtm_numeric", train_dtm_numeric)

    ### Repeat for test set
    test_dummies = dummies[dummies.source == "test"].drop("source", axis=1)
    test_numeric = sparse.csr_matrix(test_dummies.astype(np.int8))
    test_dtm_numeric = sparse.hstack((test_dtm, test_numeric))
    print(test_dtm_numeric.shape)
    sparse.save_npz(text_features_dir / "test_dtm_numeric", test_dtm_numeric)

    ### Reload stored data
    train_dtm_numeric = sparse.load_npz(text_features_dir / "train_dtm_numeric.npz")
    test_dtm_numeric = sparse.load_npz(text_features_dir / "test_dtm_numeric.npz")

    ## Benchmark Accuracy
    accuracy, runtime = {}, {}
    predictions = test[["stars"]].copy()

    # Using the most frequent number of stars (=5) to predict the test set achieve an accuracy close to 51%:
    naive_prediction = np.full_like(predictions.stars, fill_value=train.stars.mode().iloc[0])
    naive_benchmark = accuracy_score(predictions.stars, naive_prediction)
    print(naive_benchmark)

    ## Model Evaluation Helper
    def evaluate_model(model, X_train, X_test, name, store=False):
        start = time()
        model.fit(X_train, train.stars)
        runtime[name] = time() - start
        predictions[name] = model.predict(X_test)
        accuracy[result] = accuracy_score(test.stars, predictions[result])
        if store:
            joblib.dump(model, yelp_dir / f"{result}.joblib")

    ## Multiclass Naive Bayes
    nb = MultinomialNB()

    ### Text Features
    # Next, we train a Naive Bayes classifier using a document-term matrix produced by the CountVectorizer with
    # default settings.
    result = "nb_text"
    evaluate_model(nb, train_dtm, test_dtm, result, store=False)

    #### Accuracy
    # The prediction produces 64.4% accuracy on the test set, a 24.2% improvement over the benchmark:
    print(accuracy[result])

    #### Confusion Matrix
    stars = index = list(range(1, 6))
    print(
        pd.DataFrame(confusion_matrix(test.stars, predictions[result]), columns=stars, index=stars)
    )

    ### Text & Numeric Features
    result = "nb_combined"
    evaluate_model(nb, train_dtm_numeric, test_dtm_numeric, result, store=False)

    #### Accuracy
    print(accuracy[result])

    ## Multinomial Logistic Regression also provides a multinomial training option that is faster and more accurate than the
    # one-vs-all implementation. We use the lbfgs solver (see sklearn [documentation]
    # (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for details).
    Cs = np.logspace(-5, 5, 11)

    ### Text Features
    log_reg_text_accuracy = {}
    log_reg_text_runtime = []
    for i, C in enumerate(Cs):
        start = time()
        model = LogisticRegression(C=C, multi_class="multinomial", solver="lbfgs")

        model.fit(train_dtm, train.stars)
        log_reg_text_runtime.append(time() - start)
        log_reg_text_accuracy[C] = accuracy_score(test.stars, model.predict(test_dtm))
        print(
            f"{C:12.5f}: {log_reg_text_runtime[i]:.2f}s | {log_reg_text_accuracy[C]:.2%}",
            flush=True,
        )
    pd.Series(log_reg_text_accuracy).to_csv(yelp_dir / "logreg_text.csv")

    accuracy["lr_text"] = pd.Series(log_reg_text_accuracy).max()
    runtime["lr_text"] = np.mean(log_reg_text_runtime)

    ### Combined Features
    log_reg_comb_accuracy = {}
    log_reg_comb_runtime = []
    for i, C in enumerate(Cs):
        start = time()
        model = LogisticRegression(C=C, multi_class="multinomial", solver="lbfgs")

        model.fit(train_dtm_numeric, train.stars)
        log_reg_comb_runtime.append(time() - start)
        log_reg_comb_accuracy[C] = accuracy_score(test.stars, model.predict(test_dtm_numeric))
        print(
            f"{C:12.5f}: {log_reg_comb_runtime[i]:.2f}s | {log_reg_comb_accuracy[C]:.2%}",
            flush=True,
        )

    pd.Series(log_reg_comb_accuracy).to_csv(yelp_dir / "logreg_combined.csv")

    accuracy["lr_comb"] = pd.Series(log_reg_comb_accuracy).max()
    runtime["lr_comb"] = np.mean(log_reg_comb_runtime)

    ## Gradient Boosting
    # For illustration, we also train a lightgbm Gradient Boosting tree ensemble with default settings and multiclass
    # objective.
    lgb_train = lgb.Dataset(
        data=train_dtm_numeric.tocsr().astype(np.float32),
        label=train.stars.sub(1),
        categorical_feature=list(range(train_dtm_numeric.shape[1])),
    )

    lgb_test = lgb.Dataset(
        data=test_dtm_numeric.tocsr().astype(np.float32),
        label=test.stars.sub(1),
        reference=lgb_train,
    )

    param = {"objective": "multiclass", "metrics": ["multi_error"], "num_class": 5}
    booster = lgb.train(
        params=param,
        train_set=lgb_train,
        num_boost_round=2000,
        early_stopping_rounds=25,
        valid_sets=[lgb_train, lgb_test],
        verbose_eval=25,
    )

    booster.save_model((yelp_dir / "lgb_model.txt").as_posix())
    y_pred_class = booster.predict(test_dtm_numeric.astype(float))

    # The basic settings did not improve over the multinomial logistic regression, but further parameter tuning remains
    # an unused option.
    accuracy["lgb_comb"] = accuracy_score(test.stars, y_pred_class.argmax(1) + 1)

    ## Comparison
    model_map = {
        "nb_combined": "Naive Bayes",
        "lr_comb": "Logistic Regression",
        "lgb_comb": "LightGBM",
    }
    accuracy_ = {model_map[k]: v for k, v in accuracy.items() if model_map.get(k)}

    log_reg_text = pd.read_csv(yelp_dir / "logreg_text.csv", index_col=0, squeeze=True)
    log_reg_combined = pd.read_csv(yelp_dir / "logreg_combined.csv", index_col=0, squeeze=True)

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    pd.Series(accuracy_).sort_values().plot.barh(
        ax=axes[0], xlim=(0.45, 0.75), title="Accuracy by Model"
    )
    axes[0].axvline(naive_benchmark, ls="--", lw=1, c="k")

    log_reg = log_reg_text.to_frame("text").join(log_reg_combined.to_frame("combined"))
    log_reg.plot(logx=True, ax=axes[1], title="Logistic Regression - Model Tuning")
    axes[1].set_xlabel("Regularization")
    axes[1].set_ylabel("Accuracy")
    axes[0].set_xlabel("Accuracy")
    sns.despine()
    fig.tight_layout()
    plt.savefig("images/06-03.png")

    ## Textblob for Sentiment Analysis
    sample_review = train.text.sample(1).iloc[0]
    print(sample_review)

    # Polarity ranges from -1 (most negative) to 1 (most positive).
    print(TextBlob(sample_review).sentiment.polarity)

    # Define a function that accepts text and returns the polarity.
    def detect_sentiment(text):
        return TextBlob(text).sentiment.polarity

    train["sentiment"] = train.text.apply(detect_sentiment)
    sample_reviews = train[["stars", "text"]].sample(100000)

    # Create a new DataFrame column for sentiment (Warning: SLOW!).
    sample_reviews["sentiment"] = sample_reviews.text.apply(detect_sentiment)

    # Box plot of sentiment grouped by stars
    sns.boxenplot(x="stars", y="sentiment", data=train)
    plt.savefig("images/06-04.png")

    # Widen the column display.
    pd.set_option("max_colwidth", 500)

    # Reviews with most negative sentiment
    print(train[train.sentiment == -1].text.head())

    # Negative sentiment in a 5-star review
    print(train.loc[(train.stars == 5) & (train.sentiment < -0.3), "text"].head(1))

    # Positive sentiment in a 1-star review
    print(train.loc[(train.stars == 1) & (train.sentiment > 0.5), "text"].head(1))

    # Reset the column display width.
    pd.reset_option("max_colwidth")
