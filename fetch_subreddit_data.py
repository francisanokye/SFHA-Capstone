import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import praw

from config import REDDIT_CORPUS_PATH


def get_reddit_client():
    """
    Create a PRAW Reddit client using environment variables:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get(
        "REDDIT_USER_AGENT", "sfha-capstone-script/0.1 by <your-username>"
    )
    if not client_id or not client_secret:
        raise RuntimeError(
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables "
            "before running fetch_subreddit_data.py."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def query_reddit_multiple(
    subreddits,
    day_window=30,
    max_posts_per_subreddit=1000,
    only_self=True,
    fields=None,
):
    """
    Query multiple subreddits using PRAW (official Reddit API).

    This loop retrieves the specified subfields of data from submissions
    of each subreddit *within the last `day_window` days*.

    Parameters
    ----------
    subreddits : list[str]
        Subreddit names without 'r/' prefix, e.g. ['depression', 'anxiety'].
    day_window : int
        Number of days back from now to include. For each subreddit we
        iterate over `subreddit.new()` and stop once posts are older
        than `now - day_window` days.
    max_posts_per_subreddit : int
        Safety cap on how many posts to fetch per subreddit.
    only_self : bool
        If True, keep only self-posts (text posts).
    fields : list[str] or None
        Submission attributes to extract. Defaults to
        ['title', 'selftext', 'subreddit', 'created_utc',
         'author', 'num_comments', 'score', 'is_self'].

    Returns
    -------
    pandas.DataFrame
    """
    reddit = get_reddit_client()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=day_window)

    default_fields = [
        "title",
        "selftext",
        "subreddit",
        "created_utc",
        "author",
        "num_comments",
        "score",
        "is_self",
    ]
    keep_fields = fields or default_fields

    rows = []

    for sub in subreddits:
        print(f"\n--- Collecting from r/{sub} (last {day_window} days) ---")
        subreddit = reddit.subreddit(sub)
        count = 0

        # Iterate over new submissions and stop when we leave the time window.
        for submission in subreddit.new(limit=None):
            created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            if created < cutoff:
                # Older than our window: stop for this subreddit.
                break

            if only_self and not submission.is_self:
                continue

            row = {}
            for f in keep_fields:
                if f == "created_utc":
                    row[f] = submission.created_utc
                elif f == "subreddit":
                    row[f] = str(submission.subreddit)
                elif f == "author":
                    row[f] = str(submission.author) if submission.author else None
                else:
                    row[f] = getattr(submission, f, None)

            row["queried_subreddit"] = str(subreddit)
            rows.append(row)
            count += 1

            if max_posts_per_subreddit and count >= max_posts_per_subreddit:
                break

        print(f"Collected {count} posts from r/{sub} within last {day_window} days.")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def build_reddit_corpus_from_praw(
    subreddits,
    output_path=None,
    day_window=30,
    max_posts_per_subreddit=1000,
    only_self=True,
):
    """
    Fetch Reddit data using PRAW and save a reddit_corpus-style CSV.

    The CSV schema matches what the rest of the pipeline expects:
    columns: Subreddit, Date (MM/DD/YYYY), Title, Snippet.
    """
    df = query_reddit_multiple(
        subreddits=subreddits,
        day_window=day_window,
        max_posts_per_subreddit=max_posts_per_subreddit,
        only_self=only_self,
    )

    if df.empty:
        print("No data returned from Reddit; corpus not written.")
        return

    corpus = pd.DataFrame()
    corpus["Subreddit"] = df["subreddit"].astype(str)

    dates = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    corpus["Date"] = dates.dt.tz_convert(None).dt.strftime("%m/%d/%Y")

    corpus["Title"] = df.get("title", "").fillna("")
    corpus["Snippet"] = df.get("selftext", "").fillna("")

    out_path = output_path or REDDIT_CORPUS_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    corpus.to_csv(out_path, index=False)
    print(f"Saved Reddit corpus to {out_path} with {len(corpus)} rows.")


if __name__ == "__main__":
    # Example usage: fetch labelled Reddit data directly via PRAW and write reddit_corpus.csv
    subs = ["depression", "anxiety", "bipolar", "schizophrenia"]
    build_reddit_corpus_from_praw(
        subreddits=subs,
        day_window=30,
        max_posts_per_subreddit=1000,
        only_self=True,
    )