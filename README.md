# Student Feedback Classifier

A simple Flask research prototype for the study:

"Mining Student Feedback Using Classification Techniques"

## Project Purpose

This project serves as a research prototype demonstrating how the best-performing classification model from the study can automatically categorize new student feedback into **Positive**, **Neutral**, or **Negative** sentiment classes.

## Current Status

The project is currently **only initialized**. The foundational directory and file structure have been created. No machine learning models, text preprocessing logic, CSV processing, or final user interfaces have been implemented yet. These components will be integrated incrementally in subsequent stages.

## Planned Functionality

1. **Single Feedback Classification**: Real-time sentiment classification for individual student feedback entries.
2. **Batch Classification**: Automated bulk classification through CSV file uploads.
3. **Classification Results Display**: Clear presentation of automatically generated classifications.
4. **Result Visualization**: Simple visual summary of batch classification distributions.
5. **CSV Export**: Ability to download the processed CSV containing predicted classifications.

## Machine Learning Model

The actual trained classification model and vectorizer will be provided from the research/model-training environment and placed in the `models/` directory:

- `models/svm_model.pkl` (Trained Support Vector Machine model)
- `models/tfidf_vectorizer.pkl` (Fitted TF-IDF vectorizer)

The future `services/classifier.py` module will load these serialized artifacts to perform inference.

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
│   └── .gitkeep
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