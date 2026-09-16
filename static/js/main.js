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
    const maxChars = Number(counter.dataset.max);
    const items = Array.from(bubbles.querySelectorAll('.bubble-item'));

    let pending = null;

    function setResult(state, html) {
        result.dataset.state = state;
        result.innerHTML = html;
    }

    function escapeHtml(value) {
        const span = document.createElement('span');
        span.textContent = value;
        return span.innerHTML;
    }

    function clearMark() {
        items.forEach((item) => item.classList.remove('is-marked'));
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

            // Next frame so the fill transition runs even when the same bubble is re-marked.
            requestAnimationFrame(() => item.classList.add('is-marked'));

            const note = data.placeholder
                ? '<span class="note">Placeholder rule, not the trained model. Add models/svm_model.pkl and models/tfidf_vectorizer.pkl to use the study model.</span>'
                : '';
            setResult('marked', `Marked <strong>${escapeHtml(data.label)}</strong>.${note}`);
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
