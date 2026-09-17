# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Flask (Jinja2 templates) + vanilla CSS/JS. Established by existing scaffold (app.py, templates/, public/static/); deployed on Vercel; not re-asked.

## Users

Primary user: researcher/instructor analyzing student feedback text, evaluating sentiment classification results ad hoc. Matches the study context in README ("Mining Student Feedback Using Classification Techniques").

## Product Purpose

Research prototype demonstrating the best-performing classification model from the study, automatically categorizing student feedback text into Positive, Neutral, or Negative sentiment. Two entry points: single feedback classification (real-time, one entry) and batch classification (CSV upload).

## Positioning

Applies a specific trained SVM + TF-IDF pipeline (produced in a separate research/model-training environment) to real student feedback text, rather than a generic off-the-shelf sentiment API.

## Operating Context

Local/research-tool usage, not a public multi-tenant deploy. Single feedback flow: user types or pastes one feedback text, submits, sees predicted sentiment class. Batch flow (`/batch`): CSV upload, comment column choice, bulk classification, sentiment tally, results table. No CSV export.

## Capabilities and Constraints

- Single Feedback backend uses the trained model artifacts (`models/svm_model.pkl`, `models/tfidf_vectorizer.pkl`) and the training preprocessing pipeline (`services/preprocessing.py`). The artifacts are committed in `models/`.
- If the artifacts are missing, the backend falls back to a keyword placeholder and the UI labels it as not the trained model.
- The model is a linear `SVC` without probability estimates, so there is no real confidence score to show.
- Input that is empty after preprocessing (only stopwords or punctuation) is rejected with an error, not classified.
- Batch classification is built. Rows that are blank or empty after preprocessing are reported as not classified.
- Both flows show processing time measured by the app (preprocessing, model, total; batch adds per comment). These are measured timings, not accuracy claims.
- Sentiment colors: Positive green, Neutral slate, Negative red; form ink is deep blue so red only signals Negative or errors. Color is always paired with a text label.
- Sentiment classes are fixed: Positive, Neutral, Negative.

## Evidence on Hand

The trained model is on hand, but no accuracy numbers or evaluation metrics are recorded in this repo. Do not fabricate sample feedback text, accuracy numbers, or model metrics anywhere in the UI.

## Product Principles

1. Research-tool clarity over marketing polish — the interface exists to demonstrate a classification result clearly, not to persuade.
2. Single feedback flow must stay fast: type/paste, submit, see result, repeat.
3. UI must not imply the model is production-grade or validated beyond the study; no fabricated confidence/accuracy claims.
4. Keep the stub-vs-real-model swap invisible to the UI layer — the frontend contract (input text -> {label, placeholder}) must stay stable.
