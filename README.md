# Student Feedback Classifier

A simple Flask research prototype for the study:

"Mining Student Feedback Using Classification Techniques"

## Project Purpose

This project serves as a research prototype demonstrating how the best-performing classification model from the study can automatically categorize new student feedback into **Positive**, **Neutral**, or **Negative** sentiment classes.

## Current Status

- **Single feedback classification** works and uses the trained SVM model.
- **Batch classification** works: upload a CSV (up to 10 MB and 20,000 rows), pick the comment column, and get a sentiment tally and a results table with every row, 100 rows per page.
- **Processing time** is shown for both modes. The app measures it in Python with `time.perf_counter()` and split into preprocessing, model (TF-IDF transform + SVM predict), and total. Batch also shows the average time per classified comment. Model files and NLTK data are loaded when the app starts, so the first request is not slowed by disk loading.
- **Responsive layout** for phones, tablets, and desktops, including touch-sized controls, phone landscape, and notched screens.
- **Color coding by sentiment:** Positive is green, Neutral is slate gray, and Negative is red. The form itself uses deep blue, so red appears only for Negative results and errors. Every color also has a text label.

## Planned Functionality

1. **Single Feedback Classification**: Real-time sentiment classification for individual student feedback entries.
2. **Batch Classification**: Automated bulk classification through CSV file uploads.
3. **Classification Results Display**: Clear presentation of automatically generated classifications.
4. **Result Visualization**: Simple visual summary of batch classification distributions.

## Machine Learning Model

The trained model and vectorizer come from the research/model-training environment and are committed in the `models/` directory:

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
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── services/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── classifier.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── batch.html
│   └── _timing.html
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