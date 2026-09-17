function escapeHtml(value) {
    const span = document.createElement('span');
    span.textContent = value;
    return span.innerHTML;
}

function formatMs(ms) {
    if (ms === null || ms === undefined) return { value: '—', unit: '' };
    if (ms >= 1000) return { value: (ms / 1000).toFixed(2), unit: 's' };
    if (ms >= 100) return { value: ms.toFixed(0), unit: 'ms' };
    if (ms >= 10) return { value: ms.toFixed(1), unit: 'ms' };
    return { value: ms.toFixed(2), unit: 'ms' };
}

function fillTiming(section, timing) {
    if (!section) return;
    section.querySelectorAll('[data-time]').forEach((cell) => {
        const { value, unit } = formatMs(timing ? timing[cell.dataset.time] : null);
        cell.innerHTML = `<span class="t-value">${value}</span>${unit ? `<span class="t-unit">${unit}</span>` : ''}`;
    });
    section.hidden = !timing;
    if (!timing) delete section.dataset.reveal;
}

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// Scroll the answer into view and resolve once the page has settled there.
function bringIntoView(element) {
    return new Promise((resolve) => {
        const rect = element.getBoundingClientRect();
        const margin = 16;
        if (rect.top >= margin && rect.bottom <= innerHeight - margin) {
            resolve();
            return;
        }

        const instant = reduceMotion.matches;
        let settled = false;
        const finish = () => {
            if (settled) return;
            settled = true;
            removeEventListener('scrollend', finish);
            resolve();
        };

        if (!instant && 'onscrollend' in window) addEventListener('scrollend', finish);
        element.scrollIntoView({
            behavior: instant ? 'auto' : 'smooth',
            block: rect.height > innerHeight - margin * 2 ? 'start' : 'nearest',
        });
        setTimeout(finish, instant ? 0 : 700);
    });
}

// "pending" keeps a filled answer masked; "run" plays the scan-line reveal.
function setReveal(elements, state) {
    elements.forEach((element) => {
        if (!element) return;
        if (!state) {
            delete element.dataset.reveal;
            return;
        }
        if (state === 'run') {
            delete element.dataset.reveal;
            void element.offsetWidth;
        }
        element.dataset.reveal = state;
    });
}

// Busy look and label wait a beat, so instant answers do not flash a loading state.
const BUSY_DELAY_MS = 150;

function setButtonBusy(button, busy, label) {
    clearTimeout(button.busyTimer);
    if (busy) {
        button.setAttribute('aria-busy', 'true');
        button.busyTimer = setTimeout(() => {
            button.textContent = label;
        }, BUSY_DELAY_MS);
    } else {
        button.removeAttribute('aria-busy');
        button.textContent = label;
    }
}

const PLACEHOLDER_NOTE = escapeHtml(document.body.dataset.placeholderNote || '');

/* Single feedback */

(function () {
    const form = document.getElementById('classify-form');
    if (!form) return;

    const textarea = document.getElementById('feedback');
    const markButton = document.getElementById('mark');
    const eraseButton = document.getElementById('erase');
    const bubbles = document.getElementById('bubbles');
    const result = document.getElementById('result');
    const errorBox = document.getElementById('form-error');
    const counter = document.getElementById('char-count');
    const timing = form.querySelector('[data-timing]');
    const maxChars = Number(counter.dataset.max);
    const items = Array.from(bubbles.querySelectorAll('.bubble-item'));
    const answer = bubbles.closest('.field');

    let pending = null;

    function setResult(state, html, tone) {
        result.dataset.state = state;
        delete result.dataset.reveal;
        if (tone) result.dataset.tone = tone;
        else delete result.dataset.tone;
        result.innerHTML = html;
    }

    function clearMark() {
        items.forEach((item) => item.classList.remove('is-marked'));
        fillTiming(timing, null);
    }

    function updateControls() {
        const length = textarea.value.length;
        counter.textContent = length;
        counter.parentElement.classList.toggle('is-near', length > maxChars * 0.9);
        const marked = items.some((item) => item.classList.contains('is-marked'));
        eraseButton.disabled = length === 0 && !marked;
    }

    textarea.addEventListener('input', () => {
        if (errorBox.textContent) errorBox.textContent = '';
        if (result.dataset.state === 'marked') {
            clearMark();
            setResult('empty', 'Comment changed. Classify it again.');
        }
        updateControls();
    });

    textarea.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    eraseButton.addEventListener('click', () => {
        if (pending) pending.abort();
        bubbles.classList.add('is-erasing');
        clearMark();
        textarea.value = '';
        errorBox.textContent = '';
        setResult('empty', 'No mark yet.');
        updateControls();
        textarea.focus();
        setTimeout(() => bubbles.classList.remove('is-erasing'), 340);
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const text = textarea.value.trim();

        if (!text) {
            errorBox.textContent = 'Write a comment first. The sheet is blank.';
            textarea.focus();
            return;
        }

        if (pending) pending.abort();
        const controller = new AbortController();
        pending = controller;

        clearMark();
        errorBox.textContent = '';
        setButtonBusy(markButton, true, 'Classifying…');
        bubbles.setAttribute('aria-busy', 'true');
        setResult('loading', 'Classifying…');

        try {
            const response = await fetch(form.action, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
                signal: controller.signal,
            });
            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                const message = data.error || 'The classifier could not read this comment. Try again.';
                errorBox.textContent = message;
                setResult('empty', 'No mark yet.');
                return;
            }

            const item = items.find((entry) => entry.dataset.label === data.label);
            if (!item) {
                setResult('error', 'The classifier returned an unknown class. Check the terminal where the app is running.');
                return;
            }

            const note = data.placeholder ? `<span class="note">${PLACEHOLDER_NOTE}</span>` : '';
            setResult('marked', `Marked <strong>${escapeHtml(data.label)}</strong>.${note}`, item.dataset.tone);
            fillTiming(timing, data.timing);
            setReveal([result, timing], 'pending');

            // Move to row B first, then mark the bubble and scan the answer in.
            await bringIntoView(answer);
            if (pending !== controller) return;
            void item.offsetWidth;
            item.classList.add('is-marked');
            setReveal([result, timing], 'run');
        } catch (error) {
            if (error.name === 'AbortError') return;
            setResult('error', 'Could not reach the classifier. Check that the app is running (python app.py), then try again.');
        } finally {
            if (pending !== controller) return;
            pending = null;
            setButtonBusy(markButton, false, 'Classify sentiment');
            bubbles.removeAttribute('aria-busy');
            updateControls();
        }
    });

    updateControls();
})();

/* Batch */

const COLUMN_HINTS = ['comment', 'feedback', 'text', 'review', 'remark', 'response'];

// CSV reader with the same rules as Python's csv module used by app.py: quotes only open a
// field at its start, and a line with one empty or whitespace-only field is skipped.
// Row N here is row N in the app, so labels returned by the app line up with these rows.
function parseCsv(text) {
    let header = null;
    const rows = [];
    let record = [];
    let field = '';
    let quoted = false;
    let fieldStart = true;

    const endField = () => {
        record.push(field);
        field = '';
        fieldStart = true;
    };
    const endRecord = () => {
        endField();
        const blank = record.length === 1 && !record[0].trim();
        if (!blank) {
            if (!header) header = record;
            else rows.push(record);
        }
        record = [];
    };

    for (let i = 0; i < text.length; i += 1) {
        const ch = text[i];
        if (quoted) {
            if (ch !== '"') {
                field += ch;
            } else if (text[i + 1] === '"') {
                field += '"';
                i += 1;
            } else {
                quoted = false;
            }
        } else if (ch === '"' && fieldStart) {
            quoted = true;
            fieldStart = false;
        } else if (ch === ',') {
            endField();
        } else if (ch === '\r' || ch === '\n') {
            endRecord();
            if (ch === '\r' && text[i + 1] === '\n') i += 1;
        } else {
            field += ch;
            fieldStart = false;
        }
    }
    if (field !== '' || record.length || quoted) endRecord();

    return { header: header || [], rows };
}

// Returns the index of the column most likely to hold the comments.
function suggestColumn(header, sample) {
    for (const hint of COLUMN_HINTS) {
        const match = header.findIndex((name) => name.toLowerCase().includes(hint));
        if (match !== -1) return match;
    }
    // Otherwise pick the column with the longest average text.
    let best = 0;
    let bestLength = -1;
    header.forEach((_name, index) => {
        const total = sample.reduce((sum, row) => sum + (row[index] || '').length, 0);
        const average = sample.length ? total / sample.length : 0;
        if (average > bestLength) {
            best = index;
            bestLength = average;
        }
    });
    return best;
}

(function () {
    const form = document.getElementById('batch-form');
    if (!form) return;

    const fileInput = document.getElementById('csv-file');
    const dropBox = document.getElementById('drop-box');
    const fileStatus = document.getElementById('file-status');
    const columnSelect = document.getElementById('column');
    const columnNote = document.getElementById('column-note');
    const errorBox = document.getElementById('batch-error');
    const busyStatus = document.getElementById('batch-status');
    const markButton = document.getElementById('batch-mark');
    const resetButton = document.getElementById('batch-reset');
    const results = document.getElementById('results');
    const resultsNote = document.getElementById('results-note');
    const placeholderNote = document.getElementById('batch-placeholder');
    const tallyRows = Array.from(document.querySelectorAll('.tally-row'));
    const tallySkip = document.getElementById('tally-skip');
    const timing = results.querySelector('[data-timing]');
    const resultsHeading = document.getElementById('field-c');
    const tableHead = results.querySelector('.table-head');
    const tableWrap = results.querySelector('.table-wrap');
    const tableCount = document.getElementById('table-count');
    const rowsBody = document.getElementById('rows');
    const pager = document.getElementById('pager');
    const pagePrev = document.getElementById('page-prev');
    const pageNext = document.getElementById('page-next');
    const pageStatus = document.getElementById('page-status');
    const maxBytes = Number(form.dataset.maxBytes);
    const maxMb = Number(form.dataset.maxMb);
    const maxRows = Number(form.dataset.maxRows);
    const initialStatus = fileStatus.innerHTML;
    const numberFormat = new Intl.NumberFormat();
    const PAGE_SIZE = 100;

    let file = null;
    let fileRows = [];
    let pending = null;
    let reading = null;
    let items = [];
    let page = 0;

    function setError(message) {
        errorBox.textContent = message || '';
    }

    function setBusy(busy, label) {
        setButtonBusy(markButton, busy, label);
        if (busy) form.setAttribute('aria-busy', 'true');
        else form.removeAttribute('aria-busy');
        busyStatus.textContent = busy ? label : '';
        // Stay enabled while busy so keyboard focus is not dropped; the submit handler ignores repeats.
        markButton.disabled = !busy && (!file || !columnSelect.value);
        resetButton.disabled = !file && results.hidden;
    }

    function resetColumns(message) {
        columnSelect.innerHTML = `<option value="">${escapeHtml(message)}</option>`;
        columnSelect.disabled = true;
        columnNote.textContent = '';
    }

    function hideResults() {
        results.hidden = true;
        results.classList.remove('is-shown');
        setReveal([results], null);
        rowsBody.innerHTML = '';
        items = [];
        pager.hidden = true;
        fillTiming(timing, null);
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }

    function checkFile(chosen) {
        if (!/\.csv$/i.test(chosen.name)) return `${chosen.name} is not a .csv file. Export the sheet as CSV and choose it again.`;
        if (chosen.size > maxBytes) return `${chosen.name} is larger than ${maxMb} MB. Split it and try again.`;
        if (!chosen.size) return `${chosen.name} is empty.`;
        return '';
    }

    function clearFile(message) {
        file = null;
        fileRows = [];
        fileInput.value = '';
        fileStatus.innerHTML = initialStatus;
        dropBox.classList.remove('has-file');
        resetColumns('Choose a file first');
        setError(message);
        setBusy(false, 'Classify all rows');
    }

    // Read the header and row count in the browser, so the file is uploaded only once.
    async function useFile(chosen) {
        if (pending) pending.abort();
        setError('');
        hideResults();
        reading = null;

        if (!chosen) {
            clearFile('');
            return;
        }

        const problem = checkFile(chosen);
        if (problem) {
            clearFile(problem);
            return;
        }

        file = chosen;
        dropBox.classList.add('has-file');
        fileStatus.innerHTML = `<strong>${escapeHtml(chosen.name)}</strong> <span>${formatBytes(chosen.size)}</span>`;
        resetColumns('Reading columns…');
        setBusy(true, 'Reading file…');

        const token = {};
        reading = token;
        try {
            const parsed = parseCsv(await chosen.text());
            if (reading !== token) return;

            const header = parsed.header;
            if (!header.length || header.every((name) => !name.trim())) {
                throw new Error('Could not find a header row. The first line of the CSV must name the columns.');
            }
            const count = parsed.rows.length;
            if (!count) throw new Error('The CSV has a header but no rows.');
            if (count > maxRows) {
                throw new Error(`The CSV has ${numberFormat.format(count)} rows. The limit is ${numberFormat.format(maxRows)}. Split the file and try again.`);
            }
            fileRows = parsed.rows;

            // Options carry the column position, so duplicate or blank header names still map exactly.
            const suggested = suggestColumn(header, parsed.rows.slice(0, 200));
            columnSelect.innerHTML = header
                .map((name, index) => {
                    const label = name.trim() ? escapeHtml(name) : `Column ${index + 1} (no name)`;
                    return `<option value="${index}"${index === suggested ? ' selected' : ''}>${label}</option>`;
                })
                .join('');
            columnSelect.disabled = false;
            columnNote.textContent = `${numberFormat.format(count)} rows · ${header.length} columns`;
            setBusy(false, 'Classify all rows');
        } catch (error) {
            if (reading !== token) return;
            clearFile(error.message || 'Could not read this CSV. Check that it has a header row and comma-separated columns.');
        } finally {
            if (reading === token) reading = null;
        }
    }

    function rowHtml(number, [comment, label]) {
        const tag = label
            ? `<span class="tag" data-tone="${label.toLowerCase()}"><span class="dot" aria-hidden="true"></span>${label}</span>`
            : '<span class="tag is-skipped">Not classified</span>';
        const text = comment ? escapeHtml(comment) : '<span class="blank">Blank</span>';
        return `<tr role="row"><td role="cell" class="col-row">${number}</td><td role="cell" class="col-comment">${text}</td><td role="cell" class="col-label">${tag}</td></tr>`;
    }

    function renderPage() {
        const pages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
        const start = page * PAGE_SIZE;
        const slice = items.slice(start, start + PAGE_SIZE);
        rowsBody.innerHTML = slice.map((item, offset) => rowHtml(start + offset + 1, item)).join('');

        pager.hidden = pages <= 1;
        pagePrev.disabled = page === 0;
        pageNext.disabled = page >= pages - 1;
        pageStatus.textContent = `Rows ${numberFormat.format(start + 1)}–${numberFormat.format(start + slice.length)} of ${numberFormat.format(items.length)} · page ${page + 1} of ${pages}`;
    }

    function turnPage(step) {
        page += step;
        renderPage();
        tableWrap.scrollTop = 0;
        if (tableHead.getBoundingClientRect().top < 0) {
            tableHead.scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth', block: 'start' });
        }
        // Keep keyboard focus on a usable control when an edge button turns disabled.
        const pressed = step < 0 ? pagePrev : pageNext;
        if (pressed.disabled) (step < 0 ? pageNext : pagePrev).focus();
    }

    function renderResults(data) {
        const classified = data.classified;
        tallyRows.forEach((row) => {
            const count = data.counts[row.dataset.label] || 0;
            const share = classified ? (count / classified) * 100 : 0;
            row.querySelector('[data-count]').textContent = numberFormat.format(count);
            row.querySelector('[data-share]').textContent = `${share.toFixed(share > 0 && share < 10 ? 1 : 0)}%`;
            row.querySelector('.tally-fill').style.setProperty('--share', String(share / 100));
        });

        tallySkip.hidden = !data.skipped;
        tallySkip.textContent = data.skipped
            ? `${numberFormat.format(data.skipped)} ${data.skipped === 1 ? 'row was' : 'rows were'} not classified: blank, or no words left after cleaning.`
            : '';

        placeholderNote.hidden = !data.placeholder;
        resultsNote.textContent = `${numberFormat.format(classified)} of ${numberFormat.format(data.rows)} rows marked`;
        fillTiming(timing, data.timing);

        tableCount.textContent = `${numberFormat.format(data.rows)} rows`;
        const column = Number(columnSelect.value);
        items = fileRows.map((row, index) => [(row[column] || '').trim(), data.labels[index]]);
        page = 0;
        renderPage();

        results.classList.remove('is-shown');
        setReveal([results], 'pending');
        results.hidden = false;
    }

    async function revealResults(controller) {
        await bringIntoView(results);
        if (pending !== controller) return;
        setReveal([results], 'run');
        // Bars grow once the scan line has passed the tally.
        results.classList.add('is-shown');
        resultsHeading.focus({ preventScroll: true });
    }

    fileInput.addEventListener('change', () => useFile(fileInput.files[0] || null));

    columnSelect.addEventListener('change', () => {
        hideResults();
        setBusy(false, 'Classify all rows');
    });

    ['dragenter', 'dragover'].forEach((type) => dropBox.addEventListener(type, (event) => {
        event.preventDefault();
        dropBox.classList.add('is-over');
    }));
    ['dragleave', 'drop'].forEach((type) => dropBox.addEventListener(type, (event) => {
        event.preventDefault();
        dropBox.classList.remove('is-over');
    }));
    dropBox.addEventListener('drop', (event) => {
        const dropped = event.dataTransfer.files[0];
        if (dropped) useFile(dropped);
    });

    resetButton.addEventListener('click', () => {
        useFile(null);
        fileInput.focus();
    });

    pagePrev.addEventListener('click', () => turnPage(-1));
    pageNext.addEventListener('click', () => turnPage(1));

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (markButton.hasAttribute('aria-busy')) return;
        if (!file) {
            setError('Choose a CSV file first.');
            return;
        }
        if (!columnSelect.value) {
            setError('Pick the column that holds the comments.');
            return;
        }

        if (pending) pending.abort();
        const controller = new AbortController();
        pending = controller;
        setError('');
        hideResults();
        setBusy(true, 'Classifying rows…');

        try {
            const body = new FormData();
            body.append('file', file);
            body.append('column', columnSelect.value);
            const response = await fetch(form.action, { method: 'POST', body, signal: controller.signal });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.error || 'The app could not read this file. Try again.');
            if (!Array.isArray(data.labels) || data.labels.length !== fileRows.length) {
                throw new Error('The app and the page read a different number of rows. Save the file as a standard CSV and try again.');
            }
            renderResults(data);
            await revealResults(controller);
        } catch (error) {
            if (error.name === 'AbortError') return;
            setError(error instanceof TypeError
                ? 'Could not reach the classifier. Check that the app is running (python app.py), then try again.'
                : error.message);
        } finally {
            if (pending === controller) {
                pending = null;
                setBusy(false, 'Classify all rows');
            }
        }
    });

    setBusy(false, 'Classify all rows');
})();
