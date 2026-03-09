"""
Exploratory data analysis for Reddit corpus.

Generates and saves:
- Post count distribution across subreddits.
- Post length distributions and boxplots by subreddit.
- Overall post length histogram.
- Word cloud from combined text.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from config import OUTPUT_DIR


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_eda(df):
    """
    Generate EDA plots and save to output/.

    Expects df with columns: Subreddit, content
    """
    ensure_output_dir()
    sns.set(style="darkgrid")

    # Normalised subreddit labels for plotting
    df = df.copy()
    df["Subreddit"] = df["Subreddit"].astype(str)
    # Strip 'r/' for display
    df["Subreddit_display"] = df["Subreddit"].str.replace(r"^r/", "", regex=True)

    # Post count distribution across subreddits
    plt.figure(figsize=(8, 6), dpi=120)
    order = df["Subreddit_display"].value_counts().index
    sns.countplot(x="Subreddit_display", data=df, order=order, palette="viridis")
    plt.xlabel("Subreddit")
    plt.ylabel("Post count")
    plt.title("Post count distribution across subreddits")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "counts.png"))
    plt.close()
    print("Saved output/counts.png")

    # Post length (character count)
    df["post_length"] = df["content"].fillna("").astype(str).str.len()

    # Overall post length histogram
    plt.figure(figsize=(8, 5), dpi=120)
    sns.histplot(df["post_length"], bins=40, kde=True, color="steelblue")
    plt.xlabel("Post length (characters)")
    plt.ylabel("Frequency")
    plt.title("Overall post length distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "post_length_overall.png"))
    plt.close()
    print("Saved output/post_length_overall.png")

    # Post lengths by target variable (boxplot)
    plt.figure(figsize=(8, 6), dpi=120)
    sns.boxplot(data=df, x="Subreddit_display", y="post_length", palette="Set2")
    plt.xlabel("Subreddit")
    plt.ylabel("Post length (characters)")
    plt.title("Post length by subreddit (boxplot)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "post_length_boxplot.png"))
    plt.close()
    print("Saved output/post_length_boxplot.png")

    # Post lengths by target variable (violin plot)
    plt.figure(figsize=(8, 6), dpi=120)
    sns.violinplot(data=df, x="Subreddit_display", y="post_length", inner="quartile", palette="Pastel1")
    plt.xlabel("Subreddit")
    plt.ylabel("Post length (characters)")
    plt.title("Post length by subreddit (violin plot)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "post_length_violin.png"))
    plt.close()
    print("Saved output/post_length_violin.png")

    # Word cloud from text data
    text = " ".join(df["content"].astype(str).tolist())
    if text.strip():
        stopwords = set(STOPWORDS)
        wc = WordCloud(
            width=1600,
            height=900,
            background_color="white",
            stopwords=stopwords,
            max_words=200,
            collocations=False,
        ).generate(text)
        plt.figure(figsize=(10, 6), dpi=120)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word cloud of posts")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"))
        plt.close()
        print("Saved output/wordcloud.png")


if __name__ == "__main__":
    from data import load_reddit_corpus
    df = load_reddit_corpus()
    run_eda(df)
