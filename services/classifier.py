"""
Classifier service module.

Loads the trained SVM model and TF-IDF vectorizer from models/ and classifies
student feedback into Positive, Neutral, or Negative.

When the model files are missing, a placeholder keyword rule stands in so the UI flow
can still be demonstrated. The placeholder is NOT the research model and must never be
reported as its output.

Expected model files:
    - models/svm_model.pkl        (SVC, classes -1 / 0 / 1)
    - models/tfidf_vectorizer.pkl (TfidfVectorizer fitted on preprocessed text)
"""

import re
from pathlib import Path

import joblib

from services.preprocessing import preprocess

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


def model_available():
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


def _load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


class EmptyAfterPreprocessing(ValueError):
    """Raised when no words remain after preprocessing, so the model has nothing to judge."""


def _model_predict(text):
    model, vectorizer = _load_artifacts()
    cleaned = preprocess(text)
    if not cleaned:
        # An empty TF-IDF vector would only return the SVM's intercept bias, not a real judgement.
        raise EmptyAfterPreprocessing
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    return CLASS_TO_LABEL[prediction]


def _placeholder_predict(text):
    words = re.findall(r"[a-z']+", text.lower())
    score = sum(w in _POSITIVE_WORDS for w in words) - sum(w in _NEGATIVE_WORDS for w in words)
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def classify(text):
    """Return {"label": one of LABELS, "placeholder": bool}."""
    if model_available():
        return {"label": _model_predict(text), "placeholder": False}
    return {"label": _placeholder_predict(text), "placeholder": True}
