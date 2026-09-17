"""
Text preprocessing service module.

Mirrors the preprocessing used when the SVM model was trained, step for step:
    1. Lowercase, drop every character that is not a-z or whitespace, collapse whitespace
    2. Tokenize with NLTK word_tokenize
    3. Remove NLTK English stopwords
    4. Join tokens with single spaces (the form the TF-IDF vectorizer was fitted on)

Any change here must also be made in the training notebook, or predictions drift silently.
"""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}

_stop_words = None

_NON_LETTERS = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def ensure_nltk_data():
    for package, resource in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        ensure_nltk_data()
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def normalize_text(text):
    text = str(text).lower()
    text = _NON_LETTERS.sub("", text)
    text = _SPACES.sub(" ", text).strip()
    return text


@lru_cache(maxsize=100_000)
def _tokenize_word(word):
    return tuple(word_tokenize(word))


def tokenize_text(text):
    # Normalized text holds only a-z and single spaces. For that input word_tokenize only
    # splits inside single words (for example "cannot" -> "can", "not"), so tokenizing each
    # word on its own gives the same tokens, and the per-word cache makes batches fast.
    return [token for word in text.split(" ") if word for token in _tokenize_word(word)]


def remove_stopwords(tokens):
    stop_words = _get_stop_words()
    return [t for t in tokens if t not in stop_words]


def preprocess(text):
    """Return the cleaned string that the TF-IDF vectorizer expects."""
    tokens = remove_stopwords(tokenize_text(normalize_text(text)))
    return " ".join(tokens)
