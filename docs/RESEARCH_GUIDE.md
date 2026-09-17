# Research Guide: Model Integration, Preprocessing, and Timing

This guide explains, with exact file and line references, how the Student Feedback Classifier prototype uses the trained model, how it reproduces the training preprocessing (including the exact stopword list), and how the processing times shown on the pages are measured.

Line numbers refer to the code as of the commit that added this guide. If the code changes later, search for the function names given here.

---

## 1. The model files

| File | What it is | MD5 |
|---|---|---|
| `models/svm_model.pkl` | `SVC(class_weight='balanced', kernel='linear', random_state=42)`, classes `[-1, 0, 1]` | `76457d6680cb69adcfa2735e8030f810` |
| `models/tfidf_vectorizer.pkl` | `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))`, 5,000 terms | `3f127246bb0663190ed560f5f4577966` |

Both files are byte-for-byte copies of the files written by the `joblib.dump(...)` cell of the training notebook into `Sentiment Analysis Data/models/`. The MD5 hashes of the two copies are identical.

The files were saved with scikit-learn 1.9.0, so `requirements.txt` pins `scikit-learn==1.9.0` (together with the numpy, scipy, and joblib versions of the environment the app was verified in).

To re-check the hashes:

```bash
# Git Bash / macOS / Linux
md5sum models/*.pkl "../Sentiment Analysis Data/models/"*.pkl
```

```powershell
# PowerShell
Get-FileHash models\*.pkl -Algorithm MD5
```

---

## 2. How the model is integrated

All model code is in `services/classifier.py`. The web routes are in `app.py`.

### 2.1 Loading the files

`services/classifier.py`

```python
35  MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
36  MODEL_PATH = MODELS_DIR / "svm_model.pkl"
37  VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
```

```python
62  def _load_artifacts():
63      global _model, _vectorizer
64      if _model is None or _vectorizer is None:
65          _model = joblib.load(MODEL_PATH)
66          _vectorizer = joblib.load(VECTORIZER_PATH)
67      return _model, _vectorizer
```

The files are loaded once and kept in memory. `warm_up()` (lines 110–117) loads them when the app starts, and `app.py` line 33 calls `warm_up()`. Because of this, disk loading is never part of a measured request time.

### 2.2 Mapping the model's classes to labels

`services/classifier.py`

```python
32  # Numeric classes used in the training dataset.
33  CLASS_TO_LABEL = {1: "Positive", 0: "Neutral", -1: "Negative"}
```

This is the same mapping as the training notebook: `1` positive, `0` neutral, `-1` negative.

### 2.3 Turning text into a prediction

`services/classifier.py`, `_predict_many` (lines 134–149):

```python
138      if model_available():
139          model, vectorizer = _load_artifacts()
140          features = vectorizer.transform(cleaned_texts)
141          if _fast_predict_ok:
142              predictions, unsure = _fast_linear_predict(model, features)
143              if unsure.any():
144                  # A pair score within rounding distance of zero could flip sign; let libsvm decide those rows.
145                  predictions[unsure] = model.predict(features[unsure])
146          else:
147              predictions = model.predict(features)
148          return [CLASS_TO_LABEL[int(p)] for p in predictions]
```

Step by step:

1. `cleaned_texts` are the outputs of `preprocess()` (section 3), which equal the notebook's `" ".join(tokens_nostop)`.
2. Line 140 applies the fitted TF-IDF vectorizer with `transform` (never `fit`), exactly as the notebook does for `X_test`.
3. Lines 141–147 get the SVM's prediction (see 2.5).
4. Line 148 maps `-1 / 0 / 1` to `Negative / Neutral / Positive`.

### 2.4 Use of the saved TF-IDF vectorizer (`tfidf_vectorizer.pkl`)

The app uses the TF-IDF vectorizer saved by the training notebook. It does not create or refit a vectorizer.

| Where | Line in `services/classifier.py` | What happens |
|---|---|---|
| File path | 37 | `VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"` |
| Loading | 66 | `_vectorizer = joblib.load(VECTORIZER_PATH)`, once at startup |
| Every prediction | 140 | `features = vectorizer.transform(cleaned_texts)` |
| Startup check | 100 and 105 | builds probe texts from `vectorizer.vocabulary_` and transforms them |

- The app only calls `transform`, never `fit` or `fit_transform`. The vocabulary (5,000 unigrams and bigrams) and IDF weights are exactly the ones learned in the notebook, the same way the notebook produced `X_test = tfidf.transform(test_text)`.
- Both single and batch classification go through line 140.
- If `tfidf_vectorizer.pkl` (or `svm_model.pkl`) is missing, the app does not classify with the model. It switches to the placeholder rule (section 2.7) and the page says so.

### 2.5 The prediction step (`SVC.predict` and the equivalent fast path)

For speed on large batches, the app computes the linear SVM's decision directly instead of calling `SVC.predict` for every row. It gives the same class as `SVC.predict`.

`services/classifier.py`, `_fast_linear_predict` (lines 70–92):

```python
79      scores = features @ model.coef_.T
80      scores = (scores.toarray() if hasattr(scores, "toarray") else np.asarray(scores)) + model.intercept_
...
85      for i in range(n_classes):
86          for j in range(i + 1, n_classes):
87              positive = scores[:, pair] > 0
88              unsure |= np.abs(scores[:, pair]) < _ZERO_MARGIN
89              votes[positive, i] += 1
90              votes[~positive, j] += 1
91              pair += 1
92      return model.classes_[votes.argmax(axis=1)], unsure
```

Why this equals `SVC.predict` for this model:

- `SVC` with three classes trains one binary classifier per class pair (one-vs-one): (-1, 0), (-1, 1), (0, 1).
- With `kernel="linear"`, each pair's decision value is `coef · x + intercept`. scikit-learn exposes these as `model.coef_` and `model.intercept_`.
- libsvm gives the vote to the first class of the pair when the value is positive, otherwise to the second class, and picks the class with the most votes. Ties go to the lower class index, which is what `argmax` does.

Safeguards:

1. **Startup check** (`_check_fast_predict`, lines 95–107). When the app starts, it builds 1,000 probe texts from the vectorizer's vocabulary and compares the fast path with `model.predict`. The fast path is used only if every probe matches (`_fast_predict_ok`, line 117). It is also disabled for any non-linear kernel. Otherwise the app uses `model.predict` for everything (line 147).
2. **Rounding guard** (line 88 and lines 143–145). Any row whose pair score is within `1e-9` of zero (`_ZERO_MARGIN`, line 51) is sent to `model.predict`, so floating-point rounding can never flip a vote.

### 2.6 Single comment and batch entry points

**Single comment:** `app.py` lines 56–67 (`POST /api/classify`) calls `classify(text)` (`services/classifier.py` lines 152–170).

```python
155      cleaned = preprocess(text)
...
157      if not cleaned:
158          # An empty TF-IDF vector would only return the SVM's intercept bias, not a real judgement.
159          raise EmptyAfterPreprocessing
160      label = _predict_many([text], [cleaned])[0]
```

If no words remain after preprocessing (for example, the input is only stopwords or punctuation), the app returns an error message instead of a label (`app.py` lines 66–67).

**Batch (CSV):** `app.py` lines 115–146 (`POST /api/batch`) reads the CSV, takes the chosen column, and calls `classify_batch(texts)` (`services/classifier.py` lines 173–202).

```python
128      texts = [(row[index] if index < len(row) else '').strip() for row in rows]
129      result = classify_batch(texts)
```

```python
181      cleaned = [preprocess(text) if text else "" for text in texts]
...
184      keep = [i for i, value in enumerate(cleaned) if value]
185      predicted = _predict_many([texts[i] for i in keep], [cleaned[i] for i in keep])
```

Rows that are blank, or empty after preprocessing, get no label and are shown as "Not classified". All other rows are classified together in one `transform` + predict call.

### 2.7 Placeholder fallback

If either model file is missing, `model_available()` (lines 58–59) is false and a simple keyword rule (`_placeholder_predict`, lines 124–131) is used instead. The page then shows "Placeholder classifier active · not the trained study model". With both files present, as in this repository, the placeholder is never used.

---

## 3. Preprocessing: same steps as the training notebook

All preprocessing is in `services/preprocessing.py`.

| Training notebook | App (`services/preprocessing.py`) |
|---|---|
| `text = str(text).lower()` | line 60 `text = str(text).lower()` |
| `re.sub(r"[^a-z\s]", "", text)` | line 35 `_NON_LETTERS = re.compile(r"[^a-z\s]")`, used on line 61 |
| `re.sub(r"\s+", " ", text).strip()` | line 36 `_SPACES = re.compile(r"\s+")`, used on line 62 |
| `word_tokenize(text)` | lines 66–75 (`word_tokenize` applied per word, see below) |
| `stop_words = set(stopwords.words("english"))` | line 55 `_stop_words = set(stopwords.words("english"))` |
| `[t for t in tokens if t not in stop_words]` | line 80 `[t for t in tokens if t not in stop_words]` |
| `" ".join(tokens)` before `tfidf.transform` | line 86 `" ".join(tokens)` |

The regular expressions are compiled once (lines 35–36). A compiled pattern behaves exactly like the same pattern passed to `re.sub`.

### 3.1 Tokenizing per word

```python
66  @lru_cache(maxsize=100_000)
67  def _tokenize_word(word):
68      return tuple(word_tokenize(word))
69
70
71  def tokenize_text(text):
...
75      return [token for word in text.split(" ") if word for token in _tokenize_word(word)]
```

After normalization, the text contains only the letters a–z and single spaces. On that input, NLTK's `word_tokenize` never splits across spaces and only splits inside single words, for example `cannot` → `can`, `not` and `gonna` → `gon`, `na`. Calling `word_tokenize` on each word and joining the results therefore gives the same token list as calling it on the whole text. The cache only avoids repeating that work for words that were already seen.

This equivalence is checked by `scripts/verify_pipeline.py` (section 5).

---

## 4. The stopword list

### 4.1 Where it comes from

- The app reads the English stopword list from `nltk_data/corpora/stopwords/english`, which is bundled in this repository (`services/preprocessing.py` lines 24–26 put `nltk_data/` first on NLTK's search path).
- That file was copied from `C:\Users\john ansley rocel\AppData\Roaming\nltk_data\corpora\stopwords\english`.
- The training notebook used that same folder. Its `nltk.download("stopwords")` cell printed:

  ```text
  [nltk_data] Downloading package stopwords to C:\Users\john ansley
  [nltk_data]     rocel\AppData\Roaming\nltk_data...
  [nltk_data]   Package stopwords is already up-to-date!
  ```

- The source file was last modified on 2026-03-03, before the model files were saved (2026-09-15 21:51).
- The bundled copy and the source file have the same MD5: `2ac858524b32b963d32e0088b836a78c`.

### 4.2 Evidence that it matches training

1. **Identical file.** Same MD5 as the file in the folder the notebook loaded from (4.1).
2. **Same output as the notebook.** The notebook printed this for comment 0:

   ```text
   Comment: teacher are punctual but they should also give us the some practical knowledge other than theortical
   Tokens (No Stopwords): ['teacher', 'punctual', 'also', 'give', 'us', 'practical', 'knowledge', 'theortical']
   ```

   The app, with only the bundled `nltk_data/` visible to NLTK, produces exactly the same list.
3. **Consistent with the model.** None of the 198 stopwords appears in any of the 5,000 terms of the fitted TF-IDF vocabulary.
4. **Same results on the whole dataset.** `scripts/verify_pipeline.py` compares the app with the notebook pipeline (section 5).

### 4.3 The exact list (198 entries)

This is the full content of `nltk_data/corpora/stopwords/english`, in file order:

```text
a about above after again against ain all am an and any are aren aren't as at be
because been before being below between both but by can couldn couldn't d did
didn didn't do does doesn doesn't doing don don't down during each few for from
further had hadn hadn't has hasn hasn't have haven haven't having he he'd he'll
her here hers herself he's him himself his how i i'd if i'll i'm in into is isn
isn't it it'd it'll it's its itself i've just ll m ma me mightn mightn't more most
mustn mustn't my myself needn needn't no nor not now o of off on once only or
other our ours ourselves out over own re s same shan shan't she she'd she'll
she's should shouldn shouldn't should've so some such t than that that'll the
their theirs them themselves then there these they they'd they'll they're
they've this those through to too under until up ve very was wasn wasn't we
we'd we'll we're were weren weren't we've what when where which while who whom
why will with won won't wouldn wouldn't y you you'd you'll your you're yours
yourself yourselves you've
```

Because normalization removes apostrophes before the stopword step, entries that contain an apostrophe (such as `don't`) never match a token. This is the same in the notebook and in the app.

To print the list yourself:

```bash
python -c "import sys; sys.path.insert(0, '.'); import services.preprocessing as p; from nltk.corpus import stopwords; w = stopwords.words('english'); print(len(w)); print(w)"
```

### 4.4 Tokenizer tables

`word_tokenize` also needs NLTK's English `punkt_tab` tables. These are bundled in `nltk_data/tokenizers/punkt_tab/english/`, copied from the same `AppData\Roaming\nltk_data` folder. All four files match the originals:

| File | MD5 |
|---|---|
| `abbrev_types.txt` | `2bf81dc7dbfebd59c26aa9f23ed2729c` |
| `collocations.tab` | `c3de69dd6c3c7ee971c8769bee27b35f` |
| `ortho_context.tab` | `a5b1ca76814884181a919d83929478d0` |
| `sent_starters.txt` | `086bf65c05ec4f2be7bbe5dabb72b2ec` |

The app never downloads NLTK data while running. `ensure_nltk_data()` (lines 39–48) only downloads if a bundled file is missing.

---

## 5. Verifying that the app matches the training pipeline

`scripts/verify_pipeline.py` contains a separate copy of the notebook pipeline (`normalize_text`, `word_tokenize` on the whole text, the stopword filter, `" ".join`, `tfidf.transform`, `SVC.predict`) and compares it with the app's code.

```bash
python scripts/verify_pipeline.py "../Sentiment Analysis Data/finalDataset0.2_labeled.csv"
```

Result when this guide was written:

```text
Fast linear path enabled: True
PASS  labeled dataset (6 comment columns): 1110 texts, cleaned-text differences 0, label differences 0
PASS  every unique dataset word alone: 1165 texts, cleaned-text differences 0, label differences 0
PASS  every model vocabulary word alone: 1412 texts, cleaned-text differences 0, label differences 0
PASS  edge cases: 16 texts, cleaned-text differences 0, label differences 0
PASS  20k random word mixes: 20000 texts, cleaned-text differences 0, label differences 0
PASS  single-comment path: label differences 0

OVERALL: IDENTICAL TO NOTEBOOK PIPELINE
```

- "cleaned-text differences" compares the preprocessed strings.
- "label differences" compares the predicted classes.
- The edge cases include contractions (`cannot`, `gonna`), accented and curly-quote characters, emoji, tabs and new lines, digits, all caps, and a 3,000-character input.
- Re-run the script after replacing the model files or changing any code in `services/`.

---

## 6. How the processing time is measured

### 6.1 What is measured

The timings come from Python's `time.perf_counter()`, a high-resolution monotonic clock. They are measured inside the app, around the preprocessing and model steps only.

`services/classifier.py`

```python
23  from time import perf_counter
```

```python
120  def _ms(start, end):
121      return round((end - start) * 1000, 3)
```

`_ms` converts the difference between two clock readings from seconds to milliseconds, rounded to 3 decimals.

### 6.2 Single comment (`classify`, lines 152–170)

```python
154      start = perf_counter()
155      cleaned = preprocess(text)
156      preprocessed = perf_counter()
...
160      label = _predict_many([text], [cleaned])[0]
161      done = perf_counter()
...
165          "timing": {
166              "preprocess_ms": _ms(start, preprocessed),
167              "model_ms": _ms(preprocessed, done),
168              "total_ms": _ms(start, done),
169          },
```

| Shown on page | Field | Measured from → to | Includes |
|---|---|---|---|
| Preprocessing | `preprocess_ms` | `start` → `preprocessed` | lowercase, regex cleaning, tokenizing, stopword removal, join (`preprocess()`) |
| Model | `model_ms` | `preprocessed` → `done` | the empty-text check, `vectorizer.transform`, the SVM prediction, class-to-label mapping (`_predict_many`) |
| Total | `total_ms` | `start` → `done` | both of the above |

### 6.3 Batch (`classify_batch`, lines 173–202)

```python
180      start = perf_counter()
181      cleaned = [preprocess(text) if text else "" for text in texts]
182      preprocessed = perf_counter()
183
184      keep = [i for i, value in enumerate(cleaned) if value]
185      predicted = _predict_many([texts[i] for i in keep], [cleaned[i] for i in keep])
186      done = perf_counter()
...
192      total = _ms(start, done)
...
197              "preprocess_ms": _ms(start, preprocessed),
198              "model_ms": _ms(preprocessed, done),
199              "total_ms": total,
200              "per_comment_ms": round(total / len(keep), 4) if keep else None,
```

| Shown on page | Field | Meaning |
|---|---|---|
| Preprocessing | `preprocess_ms` | preprocessing of every row in the chosen column |
| Model | `model_ms` | selecting the non-empty rows, one `vectorizer.transform` for all of them, the SVM prediction, label mapping |
| Total | `total_ms` | preprocessing + model |
| Per comment | `per_comment_ms` | `total_ms` ÷ number of classified rows (blank and empty-after-cleaning rows are not counted), rounded to 4 decimals |

### 6.4 What is not included

The timings cover only the work of the classification pipeline. They do not include:

- the network or browser round trip;
- Flask request handling and JSON encoding;
- file upload and CSV reading (`app.py` `_read_upload`, lines 79–112);
- loading the model files and NLTK data, which happens once at startup (`warm_up`, lines 110–117) and never during a request;
- the page's animations.

On the first request after the app starts, the per-word tokenizer cache (section 3.1) is still empty, so preprocessing can be slightly slower than on later requests with the same words.

The values depend on the machine running the app. Times measured locally and on Vercel will differ.

### 6.5 How the page shows the times

- `app.py` returns the `timing` object unchanged in the JSON response (single: line 65; batch: line 145).
- `public/static/js/main.js`:
  - `fillTiming` (lines 15–23) writes each value into the matching `data-time` cell of `templates/_timing.html` (lines 7, 12, 17, 23);
  - `formatMs` (lines 7–13) only formats the number for display: under 10 ms with 2 decimals, under 100 ms with 1 decimal, under 1,000 ms with no decimals, and 1,000 ms or more in seconds with 2 decimals.
- The single page includes the panel at `templates/index.html` line 62. The batch page includes it at `templates/batch.html` line 82, with the extra "Per comment" cell.

---

## 7. Quick reference

| Question | Where |
|---|---|
| Where are the model files loaded? | `services/classifier.py` lines 62–67 |
| Where is the TF-IDF transform? | `services/classifier.py` line 140 |
| Where is the SVM prediction? | `services/classifier.py` lines 141–147 |
| Where are classes mapped to labels? | `services/classifier.py` lines 33 and 148 |
| Where is the preprocessing? | `services/preprocessing.py` lines 59–86 |
| Where is the stopword list loaded? | `services/preprocessing.py` line 55, file `nltk_data/corpora/stopwords/english` |
| Where are times measured? | `services/classifier.py` lines 154–168 (single) and 180–200 (batch) |
| Where are times displayed? | `public/static/js/main.js` lines 7–23, `templates/_timing.html` |
| How do I prove the app matches training? | `python scripts/verify_pipeline.py <labeled dataset CSV>` |
