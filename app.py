import io

import pandas as pd
from flask import Flask, jsonify, render_template, request

from services.classifier import (
    LABELS,
    EmptyAfterPreprocessing,
    classify,
    classify_batch,
    model_available,
    warm_up,
)

app = Flask(__name__)

MAX_FEEDBACK_CHARS = 2000
MAX_UPLOAD_MB = 10
MAX_BATCH_ROWS = 20000
PREVIEW_ROWS = 500
COLUMN_HINTS = ("comment", "feedback", "text", "review", "remark", "response")

app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

warm_up()


@app.context_processor
def inject_model_state():
    return {'model_ready': model_available()}


@app.route('/')
def index():
    return render_template('index.html', max_chars=MAX_FEEDBACK_CHARS)


@app.route('/batch')
def batch():
    return render_template(
        'batch.html', max_upload_mb=MAX_UPLOAD_MB, max_rows=MAX_BATCH_ROWS, preview_rows=PREVIEW_ROWS
    )


@app.post('/api/classify')
def api_classify():
    data = request.get_json(silent=True) or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify(error='Write a comment first. The sheet is blank.'), 400
    if len(text) > MAX_FEEDBACK_CHARS:
        return jsonify(error=f'Comment is too long. Keep it under {MAX_FEEDBACK_CHARS} characters.'), 400
    try:
        return jsonify(classify(text))
    except EmptyAfterPreprocessing:
        return jsonify(error='No words left to classify after cleaning. Write a comment with real words.'), 400


class CsvError(ValueError):
    pass


def _read_upload():
    upload = request.files.get('file')
    if upload is None or not upload.filename:
        raise CsvError('Choose a CSV file first.')
    if not upload.filename.lower().endswith('.csv'):
        raise CsvError('This file is not a .csv. Export the sheet as CSV and upload it again.')

    raw = upload.read()
    if not raw.strip():
        raise CsvError('The CSV file is empty.')

    for encoding in ('utf-8-sig', 'cp1252'):
        try:
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            raise CsvError('Could not read this CSV. Check that it has a header row and comma-separated columns.')
    else:
        raise CsvError('Could not read the text encoding. Save the CSV as UTF-8 and upload it again.')

    if frame.empty or not len(frame.columns):
        raise CsvError('The CSV has a header but no rows.')
    if len(frame) > MAX_BATCH_ROWS:
        raise CsvError(f'The CSV has {len(frame):,} rows. The limit is {MAX_BATCH_ROWS:,}. Split the file and try again.')
    return frame


def _suggest_column(frame):
    for hint in COLUMN_HINTS:
        for column in frame.columns:
            if hint in str(column).lower():
                return column
    # Otherwise pick the column with the longest average text.
    lengths = {column: frame[column].str.len().mean() for column in frame.columns}
    return max(lengths, key=lengths.get)


@app.post('/api/batch/inspect')
def api_batch_inspect():
    try:
        frame = _read_upload()
    except CsvError as error:
        return jsonify(error=str(error)), 400
    return jsonify(
        columns=[str(column) for column in frame.columns],
        suggested=str(_suggest_column(frame)),
        rows=len(frame),
    )


@app.post('/api/batch')
def api_batch():
    try:
        frame = _read_upload()
    except CsvError as error:
        return jsonify(error=str(error)), 400

    column = request.form.get('column', '')
    if column not in frame.columns:
        return jsonify(error='Pick the column that holds the comments.'), 400

    texts = [value.strip() for value in frame[column].tolist()]
    result = classify_batch(texts)
    labels = result['labels']

    counts = {label: labels.count(label) for label in LABELS}
    skipped = labels.count(None)

    preview = [
        {'row': i + 1, 'comment': texts[i], 'label': labels[i]}
        for i in range(min(len(texts), PREVIEW_ROWS))
    ]

    return jsonify(
        column=column,
        rows=len(texts),
        classified=len(texts) - skipped,
        skipped=skipped,
        counts=counts,
        preview=preview,
        placeholder=result['placeholder'],
        timing=result['timing'],
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f'The file is larger than {MAX_UPLOAD_MB} MB. Split it and try again.'), 413


if __name__ == '__main__':
    app.run(debug=True)
