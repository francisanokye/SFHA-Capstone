"""
Text preprocessing for Reddit content: lowercase, tokenize, remove stopwords, lemmatize.
"""
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available (safe to call multiple times)
def _ensure_nltk_data():
    # punkt_tab required for word_tokenize in NLTK 3.8.1+; fallback to punkt for older NLTK
    for resource in ["stopwords", "wordnet", "punkt_tab", "punkt"]:
        try:
            if resource == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def get_stopwords():
    """Return English stopwords set with custom additions used in the notebook."""
    _ensure_nltk_data()
    sw = set(stopwords.words("english"))
    sw.update(["I", "I'm", "im", "st", "r", "guys", "hey", "hello", "hi"])
    return sw


def preprocess_text(text, stopwords_set=None):
    """
    Clean a single text: lowercase, strip punctuation, tokenize, remove stopwords, lemmatize.

    Parameters
    ----------
    text : str
    stopwords_set : set, optional
        If None, get_stopwords() is used.

    Returns
    -------
    str
        Processed text as a single string.
    """
    if stopwords_set is None:
        stopwords_set = get_stopwords()
    if not isinstance(text, str) or not text.strip():
        return ""

    lower = text.lower()
    no_punct = lower.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(no_punct)
    filtered = [w for w in tokens if w not in stopwords_set]
    wn = WordNetLemmatizer()
    lemmatized = [wn.lemmatize(w, pos="v") for w in filtered]
    return " ".join(lemmatized)


def preprocess_dataframe(df, text_column="content", inplace=False):
    """
    Apply text preprocessing to a DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
    text_column : str
        Column name containing raw text.
    inplace : bool
        If True, modify df in place; else return a copy with updated column.

    Returns
    -------
    pandas.DataFrame
    """
    _ensure_nltk_data()
    stopwords_set = get_stopwords()
    out = df if inplace else df.copy()
    out[text_column] = [
        preprocess_text(t, stopwords_set) for t in out[text_column].astype(str)
    ]
    return out
