from flask import Flask, jsonify, render_template, request

from services.classifier import EmptyAfterPreprocessing, classify, model_available

app = Flask(__name__)

MAX_FEEDBACK_CHARS = 2000


@app.context_processor
def inject_model_state():
    return {'model_ready': model_available()}


@app.route('/')
def index():
    return render_template('index.html', max_chars=MAX_FEEDBACK_CHARS)


@app.route('/batch')
def batch():
    return render_template('batch.html')


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


if __name__ == '__main__':
    app.run(debug=True)
