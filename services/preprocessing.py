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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}

_stop_words = None


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
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    stop_words = _get_stop_words()
    return [t for t in tokens if t not in stop_words]


def preprocess(text):
    """Return the cleaned string that the TF-IDF vectorizer expects."""
    tokens = remove_stopwords(tokenize_text(normalize_text(text)))
    return " ".join(tokens)
