# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Flask (Jinja2 templates) + vanilla CSS/JS. Established by existing scaffold (app.py, templates/, static/); not re-asked.

## Users

Primary user: researcher/instructor analyzing student feedback text, evaluating sentiment classification results ad hoc. Matches the study context in README ("Mining Student Feedback Using Classification Techniques").

## Product Purpose

Research prototype demonstrating the best-performing classification model from the study, automatically categorizing student feedback text into Positive, Neutral, or Negative sentiment. Two entry points: single feedback classification (real-time, one entry) and batch classification (CSV upload).

## Positioning

Applies a specific trained SVM + TF-IDF pipeline (produced in a separate research/model-training environment) to real student feedback text, rather than a generic off-the-shelf sentiment API.

## Operating Context

Local/research-tool usage, not a public multi-tenant deploy. Single feedback flow: user types or pastes one feedback text, submits, sees predicted sentiment class. Batch flow (separate surface, not this build): CSV upload, bulk classification, results table, visualization, CSV export.

## Capabilities and Constraints

- Trained model artifacts (`models/svm_model.pkl`, `models/tfidf_vectorizer.pkl`) and preprocessing pipeline are not yet provided; `services/classifier.py` and `services/preprocessing.py` are stubs.
- Decision: Single Feedback backend returns a stubbed/mock classification for now so the full submit -> result UI flow is demoable end to end. Swap in real model loading later without changing the UI contract.
- Sentiment classes are fixed: Positive, Neutral, Negative.

## Evidence on Hand

No real classification examples, datasets, or trained model outputs on hand yet. Do not fabricate sample feedback text, accuracy numbers, or model metrics anywhere in the UI.

## Product Principles

1. Research-tool clarity over marketing polish — the interface exists to demonstrate a classification result clearly, not to persuade.
2. Single feedback flow must stay fast: type/paste, submit, see result, repeat.
3. UI must not imply the model is production-grade or validated beyond the study; no fabricated confidence/accuracy claims.
4. Keep the stub-vs-real-model swap invisible to the UI layer — the frontend contract (input text -> {label, maybe score}) must not change when the real model lands.
