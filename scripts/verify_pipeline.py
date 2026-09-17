"""
Check that the app returns exactly what the training notebook pipeline returns.

The reference below is the notebook code (normalize_text, NLTK word_tokenize on the whole
text, NLTK English stopwords, " ".join, TfidfVectorizer.transform, SVC.predict). The app's
own code in services/ is compared against it on several text sets.

Usage (from the project root):
    python scripts/verify_pipeline.py [path/to/finalDataset0.2_labeled.csv]

Without a dataset path, the real-dataset checks are skipped.
"""
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from services import classifier
from services.preprocessing import preprocess

# ---- Reference: notebook code ------------------------------------------------------------
stop_words = set(stopwords.words("english"))


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reference_clean(text):
    tokens = word_tokenize(normalize_text(text))
    return " ".join(t for t in tokens if t not in stop_words)


model = joblib.load(ROOT / "models/svm_model.pkl")
vectorizer = joblib.load(ROOT / "models/tfidf_vectorizer.pkl")
LABEL = classifier.CLASS_TO_LABEL


def reference_labels(texts):
    cleaned = [reference_clean(t) for t in texts]
    keep = [i for i, c in enumerate(cleaned) if c]
    labels = [None] * len(texts)
    if keep:
        for i, p in zip(keep, model.predict(vectorizer.transform([cleaned[i] for i in keep]))):
            labels[i] = LABEL[int(p)]
    return cleaned, labels


# ---- Test sets ---------------------------------------------------------------------------
classifier.warm_up()
print("Fast linear path enabled:", classifier._fast_predict_ok)

sets = {}
dataset_texts = []
if len(sys.argv) > 1:
    import csv

    comment_cols = [
        "Teaching_comments", "Coursecontent_comments", "Examination_comments",
        "Labwork_comments", "library_facilities_comments", "extracurricular_comments",
    ]
    with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    dataset_texts = [row[col] for col in comment_cols for row in rows]
    sets["labeled dataset (6 comment columns)"] = dataset_texts
    sets["every unique dataset word alone"] = sorted({w for t in dataset_texts for w in normalize_text(t).split()})

sets["every model vocabulary word alone"] = sorted({f for f in vectorizer.vocabulary_ if " " not in f})
sets["edge cases"] = [
    "I cannot believe it!!", "gonna, wanna, gotta, gimme, lemme", "Can't won't don't isn't",
    "Café résumé naïve — teacher’s “quotes”", "Tabs\tand\nnew\r\nlines", "123 456 !!!", "😀 good 👍",
    "GOOD TEACHER", "   spaced    out   ", "not good at all", "the and of", "'tis 'twas",
    "Mr. Smith's class. Dr. Lee!? e.g. i.e.", "a" * 3000, ("very " * 400) + "bad", "good." * 200,
]
random.seed(7)
words = sorted({f for f in vectorizer.vocabulary_ if " " not in f}) + ["cannot", "gonna", "wanna", "not", "no"]
sets["20k random word mixes"] = [" ".join(random.choices(words, k=random.randint(1, 60))) for _ in range(20000)]

# ---- Compare -----------------------------------------------------------------------------
all_ok = True
for name, texts in sets.items():
    started = time.perf_counter()
    ref_cleaned, ref_labels = reference_labels(texts)
    app_cleaned = [preprocess(t) for t in texts]
    app_labels = classifier.classify_batch(texts)["labels"]
    clean_diff = sum(a != b for a, b in zip(ref_cleaned, app_cleaned))
    label_diff = sum(a != b for a, b in zip(ref_labels, app_labels))
    ok = clean_diff == 0 and label_diff == 0
    all_ok &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {len(texts)} texts, "
          f"cleaned-text differences {clean_diff}, label differences {label_diff} "
          f"({time.perf_counter() - started:.1f}s)")

if dataset_texts:
    single_diff = 0
    for text in dataset_texts:
        cleaned, labels = reference_labels([text])
        if cleaned[0] and classifier.classify(text)["label"] != labels[0]:
            single_diff += 1
    all_ok &= single_diff == 0
    print(f"{'PASS' if single_diff == 0 else 'FAIL'}  single-comment path: label differences {single_diff}")

print("\nOVERALL:", "IDENTICAL TO NOTEBOOK PIPELINE" if all_ok else "MISMATCH FOUND")
sys.exit(0 if all_ok else 1)
