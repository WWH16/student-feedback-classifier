"""
Classifier service module.

Loads the trained SVM model and TF-IDF vectorizer from models/ and classifies
student feedback into Positive, Neutral, or Negative.

When the model files are missing, a placeholder keyword rule stands in so the UI flow
can still be demonstrated. The placeholder is NOT the research model and must never be
reported as its output.

Every result carries processing timings in milliseconds:
    - preprocess_ms: text cleaning (services.preprocessing)
    - model_ms:      TF-IDF transform + SVM predict (or the placeholder rule)
    - total_ms:      preprocess_ms + model_ms

Expected model files:
    - models/svm_model.pkl        (SVC, classes -1 / 0 / 1)
    - models/tfidf_vectorizer.pkl (TfidfVectorizer fitted on preprocessed text)
"""

import re
from pathlib import Path
from time import perf_counter

import joblib

from services.preprocessing import ensure_nltk_data, preprocess

LABELS = ("Positive", "Neutral", "Negative")

# Numeric classes used in the training dataset.
CLASS_TO_LABEL = {1: "Positive", 0: "Neutral", -1: "Negative"}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "svm_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

_POSITIVE_WORDS = {
    "good", "great", "excellent", "helpful", "clear", "love", "enjoyed",
    "amazing", "best", "engaging", "patient", "kind", "well", "effective",
}
_NEGATIVE_WORDS = {
    "bad", "poor", "boring", "confusing", "unclear", "late", "rude",
    "worst", "hard", "difficult", "hate", "slow", "never", "unfair",
}

_model = None
_vectorizer = None


class EmptyAfterPreprocessing(ValueError):
    """Raised when no words remain after preprocessing, so the model has nothing to judge."""


def model_available():
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


def _load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def warm_up():
    """Load model files and NLTK data up front so the first timing is not inflated by disk loads."""
    ensure_nltk_data()
    preprocess("warm up")
    if model_available():
        _load_artifacts()


def _ms(start, end):
    return round((end - start) * 1000, 3)


def _placeholder_predict(text):
    words = re.findall(r"[a-z']+", text.lower())
    score = sum(w in _POSITIVE_WORDS for w in words) - sum(w in _NEGATIVE_WORDS for w in words)
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def _predict_many(raw_texts, cleaned_texts):
    """Classify non-empty cleaned texts in one call. Returns labels in the same order."""
    if not cleaned_texts:
        return []
    if model_available():
        model, vectorizer = _load_artifacts()
        predictions = model.predict(vectorizer.transform(cleaned_texts))
        return [CLASS_TO_LABEL[int(p)] for p in predictions]
    return [_placeholder_predict(text) for text in raw_texts]


def classify(text):
    """Return {"label", "placeholder", "timing"} for one comment."""
    start = perf_counter()
    cleaned = preprocess(text)
    preprocessed = perf_counter()
    if not cleaned:
        # An empty TF-IDF vector would only return the SVM's intercept bias, not a real judgement.
        raise EmptyAfterPreprocessing
    label = _predict_many([text], [cleaned])[0]
    done = perf_counter()
    return {
        "label": label,
        "placeholder": not model_available(),
        "timing": {
            "preprocess_ms": _ms(start, preprocessed),
            "model_ms": _ms(preprocessed, done),
            "total_ms": _ms(start, done),
        },
    }


def classify_batch(texts):
    """
    Classify many comments at once.

    Returns {"labels", "placeholder", "timing"}. labels has one entry per input text:
    a sentiment label, or None when the comment is blank or empty after preprocessing.
    """
    start = perf_counter()
    cleaned = [preprocess(text) if text else "" for text in texts]
    preprocessed = perf_counter()

    keep = [i for i, value in enumerate(cleaned) if value]
    predicted = _predict_many([texts[i] for i in keep], [cleaned[i] for i in keep])
    done = perf_counter()

    labels = [None] * len(texts)
    for i, label in zip(keep, predicted):
        labels[i] = label

    total = _ms(start, done)
    return {
        "labels": labels,
        "placeholder": not model_available(),
        "timing": {
            "preprocess_ms": _ms(start, preprocessed),
            "model_ms": _ms(preprocessed, done),
            "total_ms": total,
            "per_comment_ms": round(total / len(keep), 4) if keep else None,
        },
    }
