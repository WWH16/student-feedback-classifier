# Student Feedback Classifier

A simple Flask research prototype for the study:

"Mining Student Feedback Using Classification Techniques"

## Project Purpose

This project serves as a research prototype demonstrating how the best-performing classification model from the study can automatically categorize new student feedback into **Positive**, **Neutral**, or **Negative** sentiment classes.

## Current Status

- **Single feedback classification** works and uses the trained SVM model.
- **Batch classification** (CSV upload, results display, visualization, CSV export) is not built yet.

## Planned Functionality

1. **Single Feedback Classification**: Real-time sentiment classification for individual student feedback entries.
2. **Batch Classification**: Automated bulk classification through CSV file uploads.
3. **Classification Results Display**: Clear presentation of automatically generated classifications.
4. **Result Visualization**: Simple visual summary of batch classification distributions.
5. **CSV Export**: Ability to download the processed CSV containing predicted classifications.

## Machine Learning Model

The trained model and vectorizer come from the research/model-training environment. They are not committed to the repository (see `.gitignore`), so copy them into the `models/` directory before running the app:

- `models/svm_model.pkl`: trained Support Vector Machine (`SVC`, linear kernel, `class_weight="balanced"`)
- `models/tfidf_vectorizer.pkl`: fitted TF-IDF vectorizer (`max_features=5000`, `ngram_range=(1, 2)`)

The files were saved with scikit-learn 1.9.0, so `requirements.txt` pins that version. Other versions can fail to load the files or give different results.

`services/classifier.py` loads both files and maps the model's classes to labels: `1` = Positive, `0` = Neutral, `-1` = Negative. If either file is missing, the app falls back to a simple keyword rule and shows a notice that the trained model is not in use.

### Preprocessing

`services/preprocessing.py` repeats the training notebook's preprocessing exactly:

1. Lowercase the text, remove every character that is not a letter a-z or whitespace, and collapse repeated whitespace.
2. Tokenize with NLTK `word_tokenize`.
3. Remove NLTK English stopwords.
4. Join the remaining tokens with single spaces and pass the result to the vectorizer.

If you change preprocessing in the training notebook, change this module the same way and export the model again. The app downloads the NLTK data it needs (`punkt`, `punkt_tab`, `stopwords`) on first use.

If no words remain after preprocessing (for example, the input is only stopwords or punctuation), the API returns an error instead of a label.

## Running the App

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Project Structure

```text
student-feedback-classifier/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── .gitkeep
│   ├── svm_model.pkl          (not committed)
│   └── tfidf_vectorizer.pkl   (not committed)
│
├── services/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── classifier.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── batch.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── main.js
│
└── uploads/
    └── .gitkeep
```