# How to evaluate embeddings using linear algebra and analogies
# The dimensions of the word and phrase vectors do not have an explicit meaning. However, the embeddings encode similar
# usage as proximity in the latent space in a way that carries over to semantic relationships. This results in the
# interesting properties that analogies can be expressed by adding and subtracting word vectors.
# Just as words can be used in different contexts, they can be related to other words in different ways, and these
# relationships correspond to different directions in the latent space. Accordingly, there are several types of
# analogies that the embeddings should reflect if the training data permits. The word2vec authors provide a list of
# several thousand relationships spanning aspects of geography, grammar and syntax, and family relationships to evaluate
# the quality of embedding vectors (see directory [analogies](data/analogies)).

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.set_option("float_format", "{:,.2f}".format)


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"


analogy_path = Path("../data/ch16", "analogies-en.txt")


if __name__ == "__main__":
    ## Evaluation: Analogies
    df = pd.read_csv(analogy_path, header=None, names=["category"], squeeze=True)
    categories = df[df.str.startswith(":")]
    analogies = df[~df.str.startswith(":")].str.split(expand=True)
    analogies.columns = list("abcd")

    df = pd.concat([categories, analogies], axis=1)
    df.category = df.category.ffill()
    df = df[df["a"].notnull()]
    print(df.head())

    analogy_cnt = df.groupby("category").size().sort_values(ascending=False).to_frame("n")
    analogy_example = df.groupby("category").first()
    print(analogy_cnt.join(analogy_example))

    analogy_cnt.join(analogy_example)["n"].sort_values().plot.barh(
        title="# Analogies by Category", figsize=(14, 6)
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig("images/02-01")
