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

const PLACEHOLDER_NOTE = 'Placeholder rule, not the trained model. Add models/svm_model.pkl and models/tfidf_vectorizer.pkl to use the study model.';

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
            setResult('empty', 'Comment changed. Mark sentiment again.');
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
        markButton.setAttribute('aria-busy', 'true');
        markButton.textContent = 'Reading sheet';
        bubbles.setAttribute('aria-busy', 'true');
        setResult('loading', 'Reading sheet…');

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
                setResult('error', 'The classifier returned an unknown class. Check the server log.');
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
            setResult('error', 'Could not reach the classifier. Check that the Flask server is running, then try again.');
        } finally {
            if (pending !== controller) return;
            pending = null;
            markButton.removeAttribute('aria-busy');
            markButton.textContent = 'Mark sentiment';
            bubbles.removeAttribute('aria-busy');
            updateControls();
        }
    });

    updateControls();
})();

/* Batch */

(function () {
    const form = document.getElementById('batch-form');
    if (!form) return;

    const fileInput = document.getElementById('csv-file');
    const dropBox = document.getElementById('drop-box');
    const fileStatus = document.getElementById('file-status');
    const columnSelect = document.getElementById('column');
    const columnNote = document.getElementById('column-note');
    const errorBox = document.getElementById('batch-error');
    const markButton = document.getElementById('batch-mark');
    const resetButton = document.getElementById('batch-reset');
    const results = document.getElementById('results');
    const resultsNote = document.getElementById('results-note');
    const placeholderNote = document.getElementById('batch-placeholder');
    const tallyRows = Array.from(document.querySelectorAll('.tally-row'));
    const tallySkip = document.getElementById('tally-skip');
    const timing = results.querySelector('[data-timing]');
    const resultsHeading = document.getElementById('field-c');
    const tableCount = document.getElementById('table-count');
    const rowsBody = document.getElementById('rows');
    const downloadButton = document.getElementById('download');
    const previewRows = Number(form.dataset.previewRows);
    const initialStatus = fileStatus.innerHTML;
    const numberFormat = new Intl.NumberFormat();

    let file = null;
    let csvText = '';
    let pending = null;

    function setError(message) {
        errorBox.textContent = message || '';
    }

    function setBusy(busy, label) {
        if (busy) {
            markButton.setAttribute('aria-busy', 'true');
            form.setAttribute('aria-busy', 'true');
        } else {
            markButton.removeAttribute('aria-busy');
            form.removeAttribute('aria-busy');
        }
        markButton.textContent = label;
        markButton.disabled = busy || !file || !columnSelect.value;
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
        csvText = '';
        fillTiming(timing, null);
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }

    async function postForm(url, body, signal) {
        const response = await fetch(url, { method: 'POST', body, signal });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'The server could not read this file. Try again.');
        return data;
    }

    async function useFile(chosen) {
        if (pending) pending.abort();
        setError('');
        hideResults();

        if (!chosen) {
            file = null;
            fileStatus.innerHTML = initialStatus;
            dropBox.classList.remove('has-file');
            resetColumns('Choose a file first');
            setBusy(false, 'Mark all rows');
            return;
        }

        file = chosen;
        dropBox.classList.add('has-file');
        fileStatus.innerHTML = `<strong>${escapeHtml(chosen.name)}</strong> <span>${formatBytes(chosen.size)}</span>`;
        resetColumns('Reading columns…');

        const controller = new AbortController();
        pending = controller;
        setBusy(true, 'Reading file');

        try {
            const body = new FormData();
            body.append('file', chosen);
            const data = await postForm(form.dataset.inspect, body, controller.signal);

            columnSelect.innerHTML = data.columns
                .map((name) => `<option value="${escapeHtml(name)}"${name === data.suggested ? ' selected' : ''}>${escapeHtml(name)}</option>`)
                .join('');
            columnSelect.disabled = false;
            columnNote.textContent = `${numberFormat.format(data.rows)} rows · ${data.columns.length} columns`;
        } catch (error) {
            if (error.name === 'AbortError') return;
            file = null;
            dropBox.classList.remove('has-file');
            resetColumns('Choose a file first');
            setError(error.message);
        } finally {
            if (pending === controller) {
                pending = null;
                setBusy(false, 'Mark all rows');
            }
        }
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

        const shown = Math.min(data.rows, previewRows);
        tableCount.textContent = shown < data.rows
            ? `first ${numberFormat.format(shown)} of ${numberFormat.format(data.rows)} · download for all`
            : `${numberFormat.format(data.rows)} rows`;

        rowsBody.innerHTML = data.preview.map((row) => {
            const label = row.label
                ? `<span class="tag" data-tone="${row.label.toLowerCase()}"><span class="dot" aria-hidden="true"></span>${row.label}</span>`
                : '<span class="tag is-skipped">Not classified</span>';
            const comment = row.comment ? escapeHtml(row.comment) : '<span class="blank">Blank</span>';
            return `<tr><td class="col-row" data-cell="Row">${row.row}</td><td class="col-comment" data-cell="Comment">${comment}</td><td class="col-label" data-cell="Sentiment">${label}</td></tr>`;
        }).join('');

        csvText = data.csv;
        results.classList.remove('is-shown');
        setReveal([results], 'pending');
        results.hidden = false;
    }

    async function revealResults() {
        await bringIntoView(results);
        setReveal([results], 'run');
        // Bars grow once the scan line has passed the tally.
        results.classList.add('is-shown');
        resultsHeading.focus({ preventScroll: true });
    }

    fileInput.addEventListener('change', () => useFile(fileInput.files[0] || null));

    columnSelect.addEventListener('change', () => {
        hideResults();
        setBusy(false, 'Mark all rows');
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
        fileInput.value = '';
        useFile(null);
        fileInput.focus();
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
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
        setBusy(true, 'Marking rows');

        try {
            const body = new FormData();
            body.append('file', file);
            body.append('column', columnSelect.value);
            const data = await postForm(form.action, body, controller.signal);
            renderResults(data);
            await revealResults();
        } catch (error) {
            if (error.name === 'AbortError') return;
            setError(error.message === 'Failed to fetch'
                ? 'Could not reach the classifier. Check that the Flask server is running, then try again.'
                : error.message);
        } finally {
            if (pending === controller) {
                pending = null;
                setBusy(false, 'Mark all rows');
            }
        }
    });

    downloadButton.addEventListener('click', () => {
        if (!csvText) return;
        const blob = new Blob(['﻿', csvText], { type: 'text/csv;charset=utf-8' });
        const link = document.createElement('a');
        const base = file ? file.name.replace(/\.csv$/i, '') : 'feedback';
        link.href = URL.createObjectURL(blob);
        link.download = `${base}_classified.csv`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });

    setBusy(false, 'Mark all rows');
})();
