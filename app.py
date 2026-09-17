import csv
import io
import os

from flask import Flask, jsonify, render_template, request

from services.classifier import (
    LABELS,
    EmptyAfterPreprocessing,
    classify,
    classify_batch,
    model_available,
    warm_up,
)

# Static files live in public/static: Vercel serves public/ from its CDN, Flask serves it locally.
app = Flask(__name__, static_folder='public/static', static_url_path='/static')

ON_VERCEL = bool(os.environ.get('VERCEL'))

MAX_FEEDBACK_CHARS = 2000
# Vercel Functions accept request bodies up to 4.5 MB, so uploads stay under that there.
MAX_UPLOAD_MB = 4 if ON_VERCEL else 10
MAX_BATCH_ROWS = 20000
PLACEHOLDER_NOTE = (
    'Placeholder rule, not the trained model. '
    'Add models/svm_model.pkl and models/tfidf_vectorizer.pkl to use the study model.'
)

# Allow a little room for the multipart form wrapper around the file.
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024 + 64 * 1024

warm_up()


@app.context_processor
def inject_model_state():
    return {'model_ready': model_available(), 'placeholder_note': PLACEHOLDER_NOTE}


@app.route('/')
def index():
    return render_template('index.html', max_chars=MAX_FEEDBACK_CHARS)


@app.route('/batch')
def batch():
    return render_template(
        'batch.html',
        max_upload_mb=MAX_UPLOAD_MB,
        max_upload_bytes=MAX_UPLOAD_MB * 1024 * 1024,
        max_rows=MAX_BATCH_ROWS,
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


def _is_blank(record):
    # Same rule as parseCsv in main.js: a line with one empty or whitespace-only field is skipped.
    return len(record) == 1 and not record[0].strip()


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
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise CsvError('Could not read the text encoding. Save the CSV as UTF-8 and upload it again.')

    # The browser reads the same file with matching rules, so row N here is row N on the page.
    try:
        records = [record for record in csv.reader(io.StringIO(text, newline='')) if record and not _is_blank(record)]
    except csv.Error:
        raise CsvError('Could not read this CSV. Check that it has a header row and comma-separated columns.')

    if not records:
        raise CsvError('The CSV file is empty.')
    header, rows = records[0], records[1:]
    if not rows:
        raise CsvError('The CSV has a header but no rows.')
    if len(rows) > MAX_BATCH_ROWS:
        raise CsvError(f'The CSV has {len(rows):,} rows. The limit is {MAX_BATCH_ROWS:,}. Split the file and try again.')
    return header, rows


@app.post('/api/batch')
def api_batch():
    try:
        header, rows = _read_upload()
    except CsvError as error:
        return jsonify(error=str(error)), 400

    # The page sends the column position, which stays exact for blank or duplicate header names.
    position = request.form.get('column', '')
    if not position.isdigit() or int(position) >= len(header):
        return jsonify(error='Pick the column that holds the comments.'), 400
    index = int(position)

    texts = [(row[index] if index < len(row) else '').strip() for row in rows]
    result = classify_batch(texts)
    labels = result['labels']

    counts = {label: labels.count(label) for label in LABELS}
    skipped = labels.count(None)

    return jsonify(
        column=header[index],
        rows=len(texts),
        classified=len(texts) - skipped,
        skipped=skipped,
        counts=counts,
        # Labels only, in row order: the page already has the comments from its own read of the file,
        # and echoing them back could push the response past Vercel's 4.5 MB limit.
        labels=labels,
        placeholder=result['placeholder'],
        timing=result['timing'],
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f'The file is larger than {MAX_UPLOAD_MB} MB. Split it and try again.'), 413


if __name__ == '__main__':
    app.run(debug=True)
