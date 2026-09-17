# Student Feedback Classifier

A simple Flask research prototype for the study:

"Mining Student Feedback Using Classification Techniques"

## Project Purpose

This project serves as a research prototype demonstrating how the best-performing classification model from the study can automatically categorize new student feedback into **Positive**, **Neutral**, or **Negative** sentiment classes.

## Current Status

- **Single feedback classification** works and uses the trained SVM model.
- **Batch classification** works: upload a CSV (up to 10 MB locally or 4 MB on Vercel, and 20,000 rows), pick the comment column, and get a sentiment tally and a results table with every row, 100 rows per page.
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

If you change preprocessing in the training notebook, change this module the same way and export the model again. The NLTK data it needs (the English `punkt_tab` tables and the English stopword list) is bundled in `nltk_data/`, so the app never downloads anything at runtime.

For speed, the app tokenizes one word at a time with a cache, and computes the linear SVM's pair votes with a sparse matrix product instead of calling `SVC.predict`. Both give the same output as the notebook pipeline. At startup the app compares the fast prediction with `SVC.predict` on 1,000 probe texts and falls back to `SVC.predict` if they ever differ. Any row whose pair score is within 1e-9 of zero is also sent to `SVC.predict`, so rounding can never flip a vote. A 20,000-row batch takes about half a second.

If no words remain after preprocessing (for example, the input is only stopwords or punctuation), the API returns an error instead of a label.

## Running the App

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Research Guide

See [docs/RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md) for the exact code that loads and runs the model, how the preprocessing and stopword list match the training notebook, and how the processing times are measured. To check that the app still gives the same results as the notebook pipeline, run:

```bash
python scripts/verify_pipeline.py "path/to/finalDataset0.2_labeled.csv"
```

## Deploying to Vercel

The app is set up for Vercel's zero-configuration Flask support:

- Vercel loads the `app` object from `app.py`.
- Static files live in `public/static/`, which Vercel serves from its CDN. Locally, Flask serves the same folder at `/static`.
- `.python-version` selects Python 3.13, and `requirements.txt` pins the package versions the model was saved with.
- `vercel.json` gives the function up to 60 seconds and keeps `public/` and local-only files out of the function bundle. `.vercelignore` keeps the virtual environment and editor files out of the upload.
- Vercel Functions accept request and response bodies up to 4.5 MB. On Vercel (detected with the `VERCEL` environment variable) the CSV upload limit is 4 MB. The batch response carries only the labels; the page pairs them with the comments it read from the file itself, using the same CSV rules as the app.

Deploy with the Vercel CLI (version 48.2.10 or later):

```bash
npm i -g vercel
vercel          # preview deployment
vercel --prod   # production deployment
```

Or import the Git repository in the Vercel dashboard. No build command or environment variables are needed.

## Project Structure

```text
student-feedback-classifier/
│
├── app.py
├── requirements.txt
├── .python-version
├── vercel.json
├── .vercelignore
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
├── nltk_data/                 (bundled English tokenizer tables and stopwords)
│
├── public/
│   └── static/
│       ├── css/
│       │   └── style.css
│       ├── fonts/
│       │   └── archivo-latin-var.woff2
│       └── js/
│           └── main.js
│
└── uploads/
    └── .gitkeep
```