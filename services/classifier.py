"""
Classifier service module.

Loads the trained SVM model and TF-IDF vectorizer when they exist in models/.
Until then, a placeholder keyword rule stands in so the UI flow can be demonstrated.
The placeholder is NOT the research model and must never be reported as its output.

Expected model files:
    - models/svm_model.pkl
    - models/tfidf_vectorizer.pkl
"""

import re
from pathlib import Path

LABELS = ("Positive", "Neutral", "Negative")

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


def model_available():
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


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
    # TODO: when model_available(), load artifacts with joblib, run services.preprocessing, and predict.
    return {"label": _placeholder_predict(text), "placeholder": True}
