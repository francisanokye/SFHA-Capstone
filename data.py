"""
Load and prepare Reddit corpus from a folder of CSVs.

- Merges all CSV files in the data folder (first four columns: subreddit, author, date, post).
- Cleans text (non-ASCII, HTML entities).
- Maps subreddits to mental-health labels and drops unmapped.
- Balances categories to the size of the smallest category.
"""
import os
import re
from pathlib import Path

import pandas as pd

from config import DATA_FOLDER, REDDIT_DATE_END, SUBREDDIT_TO_LABEL


def load_and_combine_dataset(data_folder: str) -> pd.DataFrame:
    """
    Load and combine all CSV files from a folder, using the first four columns of each.

    Args:
        data_folder: Path to the folder containing CSV files (relative or absolute).

    Returns:
        Single DataFrame with combined data. Empty if no CSVs found.
    """
    cwd = Path.cwd()
    folder = Path(data_folder) if os.path.isabs(data_folder) else cwd / data_folder
    if not folder.is_dir():
        return pd.DataFrame()

    csv_files = sorted(folder.glob("*.csv"))
    frames = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] < 4:
            continue
        frames.append(df.iloc[:, :4].copy())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_text(text) -> str:
    """
    Clean input text: remove non-ASCII, strip problematic sequences, replace &amp; with &.
    """
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = "".join(c for c in s if ord(c) < 128)
    s = re.sub(r"[^\x00-\x7F]+", "", s)
    s = s.replace("&amp;", "&")
    return s


def balance_to_min_category(df: pd.DataFrame, category_column: str) -> pd.DataFrame:
    """
    Undersample so every category has the same number of rows as the smallest category.
    """
    if category_column not in df.columns or df.empty:
        return df
    counts = df[category_column].value_counts()
    min_count = counts.min()
    if min_count == 0:
        return df.loc[df[category_column].isin(counts[counts > 0].index)].copy()
    sampled = [
        df.loc[df[category_column] == cat].sample(n=min_count, random_state=42, replace=False)
        for cat in counts.index
    ]
    return pd.concat(sampled, ignore_index=True)


def _normalize_combined_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw combined DataFrame (first 4 cols) to Subreddit, Date, Title, Snippet."""
    cols_lower = [str(c).strip().lower() for c in df.columns[:4]]
    out = pd.DataFrame(index=df.index)
    if "subreddit" in cols_lower and "post" in cols_lower:
        idx_sub = cols_lower.index("subreddit")
        idx_date = cols_lower.index("date")
        idx_post = cols_lower.index("post")
        idx_author = cols_lower.index("author")
        out["Subreddit"] = df.iloc[:, idx_sub].astype(str).str.strip()
        out["Date"] = df.iloc[:, idx_date]
        out["Title"] = df.iloc[:, idx_author].fillna("").astype(str)
        out["Snippet"] = df.iloc[:, idx_post].fillna("").astype(str)
    else:
        out["Subreddit"] = df.iloc[:, 0].astype(str).str.strip()
        out["Date"] = df.iloc[:, 2]
        out["Title"] = df.iloc[:, 1].fillna("").astype(str)
        out["Snippet"] = df.iloc[:, 3].fillna("").astype(str)
    return out


def _map_subreddit_to_label(series: pd.Series) -> pd.Series:
    """Map subreddit names to canonical labels; unmapped become None."""
    def map_one(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        r = "r/" + s if not s.startswith("r/") else s
        return SUBREDDIT_TO_LABEL.get(s) or SUBREDDIT_TO_LABEL.get(r)
    return series.map(map_one)


def load_reddit_corpus(data_folder=None, balance=True):
    """
    Load Reddit corpus from a folder of CSVs: merge, clean, filter by mental-health labels, balance.

    Parameters
    ----------
    data_folder : str, optional
        Path to folder containing CSV files. Defaults to config.DATA_FOLDER.
    balance : bool, default True
        If True, undersample so each category has the same count as the smallest.

    Returns
    -------
    pandas.DataFrame
        Columns: Subreddit, Date, Title, Snippet, content
    """
    folder = data_folder if data_folder is not None else DATA_FOLDER
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")

    raw = load_and_combine_dataset(folder)
    if raw.empty:
        raise ValueError(f"No CSV data found in {folder}")

    df = _normalize_combined_raw(raw)
    df = df[df["Subreddit"].notna() & (df["Subreddit"].str.strip() != "")]
    df["Subreddit"] = _map_subreddit_to_label(df["Subreddit"])
    df = df.dropna(subset=["Subreddit"])
    if df.empty:
        raise ValueError(
            "No rows left after mapping subreddits to labels. "
            "Ensure data contains mental-health subreddits (see config.MENTAL_HEALTH_SUBREDDITS)."
        )

    df["Title"] = df["Title"].apply(clean_text)
    df["Snippet"] = df["Snippet"].apply(clean_text)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] < pd.Timestamp(REDDIT_DATE_END)]

    df = df.drop_duplicates(subset=["Title"])
    df = df.drop_duplicates(subset=["Snippet"])
    df["content"] = (df["Title"] + " " + df["Snippet"]).str.strip()
    df = df.drop_duplicates(subset=["content"])

    if balance:
        df = balance_to_min_category(df, "Subreddit")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_reddit_corpus()
    print(df.shape)
    print(df["Subreddit"].value_counts(dropna=False))
    print(df[["Subreddit", "Date", "content"]].head())
