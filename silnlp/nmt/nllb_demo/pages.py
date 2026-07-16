"""HTML pages for the combined demo.

Both demos share one layout (``_layout``) — the same ``<head>``, CSS design tokens,
and header with a **Suggest / Evaluate** nav. Each page contributes its own body
markup, a small amount of page-specific CSS, and a page script. All dynamic values
(language list, defaults) are injected as a single ``window.__NLLB__`` JSON blob so the
CSS/JS below can stay as verbatim static strings.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any, Dict, List

# ── Shared header icon (translate glyph) ──────────────────────────────────────

_HEADER_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="white">'
    '<path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17'
    "C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02"
    'L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33'
    'L19.12 17h-3.24z"/></svg>'
)

_SWAP_ICON = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M6.99 11L3 15l3.99 4v-3H14v-2H6.99v-3zM21 9l-3.99-4v3H10v2h7.01v3L21 9z"/></svg>'
)

# ── Shared CSS (design tokens + header/nav + language bar + panels) ────────────

_SHARED_CSS = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --blue: #1a73e8;
      --blue-light: #e8f0fe;
      --surface: #ffffff;
      --bg: #f0f4f9;
      --border: #e0e0e0;
      --text: #202124;
      --muted: #5f6368;
      --hint: #bdc1c6;
      --radius: 12px;
      --font: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --mono: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
      --shadow: 0 1px 3px rgba(0,0,0,.1), 0 4px 12px rgba(0,0,0,.06);
    }

    body {
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }

    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      height: 60px;
      display: flex;
      align-items: center;
      padding: 0 24px;
      gap: 12px;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .header-icon {
      width: 34px;
      height: 34px;
      background: var(--blue);
      border-radius: 8px;
      display: grid;
      place-items: center;
      flex-shrink: 0;
    }

    header h1 {
      font-size: 16px;
      font-weight: 500;
      letter-spacing: -.01em;
    }

    nav.tabs { display: flex; gap: 4px; margin-left: 10px; }

    nav.tabs a {
      font-size: 13px;
      font-weight: 500;
      color: var(--muted);
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 8px;
      transition: background .12s, color .12s;
    }

    nav.tabs a:hover { background: #f5f5f5; }
    nav.tabs a.active { background: var(--blue-light); color: var(--blue); }

    .spacer { flex: 1; }

    .spinner {
      width: 18px;
      height: 18px;
      border: 2.5px solid var(--border);
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin .65s linear infinite;
      opacity: 0;
      transition: opacity .2s;
      flex-shrink: 0;
    }

    .spinner.active { opacity: 1; }

    @keyframes spin { to { transform: rotate(360deg); } }

    main {
      max-width: 1080px;
      margin: 28px auto;
      padding: 0 20px 28px;
    }

    .card {
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .lang-bar {
      display: grid;
      grid-template-columns: 1fr 52px 1fr;
      align-items: center;
      border-bottom: 1px solid var(--border);
      height: 52px;
    }

    .lang-picker {
      display: flex;
      align-items: center;
      padding: 0 18px;
      gap: 8px;
      height: 100%;
    }

    .lang-badge {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .07em;
      color: var(--muted);
      flex-shrink: 0;
    }

    select.lang-select {
      flex: 1;
      border: none;
      background: transparent;
      font-family: var(--font);
      font-size: 14px;
      font-weight: 500;
      color: var(--text);
      cursor: pointer;
      appearance: none;
      outline: none;
      padding: 6px 8px;
      border-radius: 6px;
      min-width: 0;
      transition: background .12s;
    }

    select.lang-select:hover { background: #f5f5f5; }
    select.lang-select:focus { background: var(--blue-light); color: var(--blue); }

    .swap-btn {
      margin: auto;
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 50%;
      border: 1px solid var(--border);
      background: var(--surface);
      cursor: pointer;
      color: var(--muted);
      transition: background .12s, transform .25s cubic-bezier(.4,0,.2,1);
    }

    .swap-btn:hover { background: #f5f5f5; transform: rotate(180deg); }

    .panels {
      display: grid;
      grid-template-columns: 1fr 1fr;
    }

    .panel {
      display: flex;
      flex-direction: column;
      padding: 20px 20px 14px;
      min-height: 280px;
      position: relative;
    }

    .panel + .panel { border-left: 1px solid var(--border); }

    .editor-font {
      font-family: var(--font);
      font-size: 18px;
      line-height: 1.65;
    }

    textarea.source-area {
      background: transparent;
      border: none;
      outline: none;
      resize: none;
      width: 100%;
      flex: 1;
      padding: 0;
      min-height: 200px;
      color: var(--text);
    }

    textarea.source-area::placeholder { color: var(--hint); }

    .panel-footer {
      display: flex;
      align-items: center;
      margin-top: 10px;
      min-height: 22px;
    }

    @media (max-width: 640px) {
      .panels { grid-template-columns: 1fr; }
      .panel + .panel { border-left: none; border-top: 1px solid var(--border); }
      .lang-bar { grid-template-columns: 1fr 44px 1fr; }
    }
"""

# ── Suggest page ──────────────────────────────────────────────────────────────

_SUGGEST_CSS = """
    main { display: flex; flex-direction: column; gap: 16px; }

    .clear-btn {
      display: none;
      position: absolute;
      top: 18px;
      right: 16px;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #e8eaed;
      border: none;
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
      align-items: center;
      justify-content: center;
      line-height: 1;
      transition: background .12s;
    }

    .clear-btn.visible { display: flex; }
    .clear-btn:hover { background: #dadce0; }

    .char-count { font-size: 12px; color: var(--hint); margin-left: auto; }

    .ghost-wrapper { position: relative; flex: 1; display: flex; }

    .ghost-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
      overflow: hidden;
      z-index: 1;
      color: var(--text);
      white-space: pre-wrap;
      word-wrap: break-word;
    }

    .ghost-layer .hint { color: var(--hint); }
    .ghost-layer .placeholder { color: var(--hint); }

    textarea.target-area {
      position: relative;
      z-index: 2;
      color: transparent;
      caret-color: var(--text);
      overflow: auto;
      flex: 1;
      min-height: 200px;
      width: 100%;
      background: transparent;
      border: none;
      outline: none;
      resize: none;
      padding: 0;
    }

    .tab-hint {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      opacity: 0;
      transition: opacity .18s;
    }

    .tab-hint.visible { opacity: 1; }

    .tab-hint kbd {
      background: #f5f5f5;
      border: 1px solid #d0d0d0;
      border-bottom-width: 2px;
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 11px;
      font-family: var(--mono);
      color: var(--text);
    }

    .settings-card {
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .settings-label { font-size: 13px; font-weight: 500; color: var(--muted); flex-shrink: 0; }

    input[type=range] { flex: 1; accent-color: var(--blue); cursor: pointer; }

    .badge {
      background: var(--blue);
      color: #fff;
      font-size: 12px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 12px;
      min-width: 46px;
      text-align: center;
      flex-shrink: 0;
    }
"""

_SUGGEST_BODY = """
  <div class="card">
    <div class="lang-bar">
      <div class="lang-picker">
        <span class="lang-badge">From</span>
        <select id="srcLang" class="lang-select"></select>
      </div>
      <button id="swapBtn" class="swap-btn" title="Swap languages">__SWAP_ICON__</button>
      <div class="lang-picker">
        <span class="lang-badge">To</span>
        <select id="tgtLang" class="lang-select"></select>
      </div>
    </div>

    <div class="panels">
      <div class="panel">
        <textarea id="sourceText" class="editor-font source-area" placeholder="Enter text to translate…"></textarea>
        <button id="clearBtn" class="clear-btn" title="Clear">&#x2715;</button>
        <div class="panel-footer">
          <span id="charCount" class="char-count"></span>
        </div>
      </div>

      <div class="panel">
        <div class="ghost-wrapper">
          <div id="ghostText" class="editor-font ghost-layer" aria-hidden="true"></div>
          <textarea id="targetText" class="editor-font target-area" spellcheck="false"></textarea>
        </div>
        <div class="panel-footer">
          <div id="tabHint" class="tab-hint">
            Press <kbd>Tab</kbd> to accept suggestion
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="settings-card">
    <span class="settings-label">Confidence threshold</span>
    <input type="range" id="confidenceThreshold" min="0" max="1" step="0.05" value="0.7" />
    <span id="thresholdValue" class="badge">0.70</span>
  </div>
""".replace(
    "__SWAP_ICON__", _SWAP_ICON
)

_SUGGEST_JS = """
  const cfg = window.__NLLB__;
  const sourceText = document.getElementById('sourceText');
  const targetText = document.getElementById('targetText');
  const srcLang    = document.getElementById('srcLang');
  const tgtLang    = document.getElementById('tgtLang');
  const ghostText  = document.getElementById('ghostText');
  const confidenceThreshold = document.getElementById('confidenceThreshold');
  const thresholdValue = document.getElementById('thresholdValue');
  const spinner    = document.getElementById('spinner');
  const clearBtn   = document.getElementById('clearBtn');
  const charCount  = document.getElementById('charCount');
  const tabHint    = document.getElementById('tabHint');
  const swapBtn    = document.getElementById('swapBtn');

  let pendingSuggestion = '';
  let debounceHandle = null;
  let requestSequence = 0;
  let activeRequests = 0;

  function fillLangSelect(select, defaultCode) {
    for (const code of cfg.languages) {
      const option = document.createElement('option');
      option.value = code;
      option.textContent = code;
      if (code === defaultCode) option.selected = true;
      select.appendChild(option);
    }
  }

  fillLangSelect(srcLang, cfg.defaultSrc);
  fillLangSelect(tgtLang, cfg.defaultTgt);

  function setLoading(on) {
    activeRequests = Math.max(0, activeRequests + (on ? 1 : -1));
    spinner.classList.toggle('active', activeRequests > 0);
  }

  function updateClearBtn() {
    clearBtn.classList.toggle('visible', sourceText.value.length > 0);
  }

  function updateCharCount() {
    const n = sourceText.value.length;
    charCount.textContent = n > 0 ? n + ' chars' : '';
  }

  function escapeHtml(value) {
    return value
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;');
  }

  function isCaretAtEnd() {
    return (
      targetText.selectionStart === targetText.value.length &&
      targetText.selectionEnd   === targetText.value.length
    );
  }

  function shouldShowSuggestion() {
    return document.activeElement === targetText && isCaretAtEnd() && !!pendingSuggestion;
  }

  function renderGhostText() {
    const show = shouldShowSuggestion();
    if (!targetText.value && !show) {
      ghostText.innerHTML = '<span class="placeholder">Type translation…</span>';
    } else {
      ghostText.innerHTML =
        escapeHtml(targetText.value) +
        (show ? '<span class="hint">' + escapeHtml(pendingSuggestion) + '</span>' : '');
    }
    tabHint.classList.toggle('visible', show);
    syncGhostScroll();
  }

  function syncGhostScroll() {
    ghostText.scrollTop  = targetText.scrollTop;
    ghostText.scrollLeft = targetText.scrollLeft;
  }

  async function requestSuggestion() {
    const requestId = ++requestSequence;
    const payload = {
      source_text: sourceText.value,
      partial_translation: targetText.value,
      src_lang: srcLang.value,
      tgt_lang: tgtLang.value,
      confidence_threshold: parseFloat(confidenceThreshold.value)
    };

    if (!payload.source_text.trim() || !isCaretAtEnd()) {
      pendingSuggestion = '';
      renderGhostText();
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (requestId !== requestSequence) return;
      if (response.ok) {
        const data = await response.json();
        if (requestId !== requestSequence) return;
        pendingSuggestion = data.suggestion || '';
      } else {
        pendingSuggestion = '';
      }
    } catch (_) {
      if (requestId === requestSequence) pendingSuggestion = '';
    } finally {
      setLoading(false);
    }
    renderGhostText();
  }

  function debounceSuggest() {
    if (debounceHandle) clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {
      requestSuggestion().catch(() => {
        pendingSuggestion = '';
        setLoading(false);
        renderGhostText();
      });
    }, 300);
  }

  function handleTargetInput() {
    pendingSuggestion = '';
    renderGhostText();
    debounceSuggest();
  }

  targetText.addEventListener('keydown', (e) => {
    if (e.key === 'Tab' && pendingSuggestion && isCaretAtEnd()) {
      e.preventDefault();
      targetText.value += pendingSuggestion;
      pendingSuggestion = '';
      renderGhostText();
      debounceSuggest();
    }
  });

  function handleCaretMovement() {
    if (!isCaretAtEnd() && pendingSuggestion) pendingSuggestion = '';
    renderGhostText();
  }

  confidenceThreshold.addEventListener('input', () => {
    thresholdValue.textContent = parseFloat(confidenceThreshold.value).toFixed(2);
    debounceSuggest();
  });

  sourceText.addEventListener('input', () => {
    updateClearBtn();
    updateCharCount();
    debounceSuggest();
  });

  clearBtn.addEventListener('click', () => {
    sourceText.value = '';
    pendingSuggestion = '';
    updateClearBtn();
    updateCharCount();
    renderGhostText();
    sourceText.focus();
  });

  swapBtn.addEventListener('click', () => {
    const tmpLang = srcLang.value;
    srcLang.value = tgtLang.value;
    tgtLang.value = tmpLang;
    const tmpText = sourceText.value;
    sourceText.value = targetText.value;
    targetText.value = tmpText;
    pendingSuggestion = '';
    updateClearBtn();
    updateCharCount();
    renderGhostText();
    debounceSuggest();
  });

  targetText.addEventListener('input',  handleTargetInput);
  targetText.addEventListener('click',  handleCaretMovement);
  targetText.addEventListener('keyup',  handleCaretMovement);
  targetText.addEventListener('select', handleCaretMovement);
  targetText.addEventListener('focus',  renderGhostText);
  targetText.addEventListener('blur',   renderGhostText);
  targetText.addEventListener('scroll', syncGhostScroll);
  srcLang.addEventListener('change', debounceSuggest);
  tgtLang.addEventListener('change', debounceSuggest);

  updateClearBtn();
  updateCharCount();
  renderGhostText();
"""

# ── Evaluate page ─────────────────────────────────────────────────────────────

_EVALUATE_CSS = """
    .card { overflow: visible; }

    .target-editor {
      flex: 1;
      min-height: 200px;
      width: 100%;
      padding: 0;
      outline: none;
      cursor: text;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
    }

    .target-editor:empty::before {
      content: attr(data-placeholder);
      color: var(--hint);
      pointer-events: none;
    }

    .flag {
      background: #fce8e6;
      border-bottom: 2px solid #d93025;
      border-radius: 2px;
      cursor: pointer;
    }

    .flag:hover, .flag.active { background: #fad2cf; }

    .suggestion-menu {
      position: absolute;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      z-index: 200;
      min-width: 200px;
      max-width: 380px;
      max-height: 260px;
      overflow-y: auto;
    }

    .suggestion-menu.hidden { display: none; }

    .suggestion-item {
      display: flex;
      align-items: baseline;
      gap: 8px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 14px;
      border-bottom: 1px solid var(--bg);
    }

    .suggestion-item:last-child { border-bottom: none; }
    .suggestion-item:hover { background: var(--bg); }
    .suggestion-phrase { font-weight: 500; color: var(--text); }
    .suggestion-delta { font-size: 12px; color: var(--muted); white-space: nowrap; }
    .suggestion-empty { padding: 10px 16px; font-size: 14px; color: var(--muted); font-style: italic; }

    .status-text { font-size: 12px; color: var(--muted); }
"""

_EVALUATE_BODY = """
  <div class="card">
    <div class="lang-bar">
      <div class="lang-picker">
        <span class="lang-badge">From</span>
        <select id="source-language" class="lang-select"></select>
      </div>
      <button id="swap-btn" class="swap-btn" title="Swap languages">__SWAP_ICON__</button>
      <div class="lang-picker">
        <span class="lang-badge">To</span>
        <select id="target-language" class="lang-select"></select>
      </div>
    </div>
    <div class="panels">
      <div class="panel">
        <textarea id="source-text"
                  class="editor-font source-area"
                  placeholder="Paste source sentence"></textarea>
        <div class="panel-footer"></div>
      </div>
      <div class="panel">
        <div id="target-editor"
             class="editor-font target-editor"
             contenteditable="true"
             spellcheck="false"
             role="textbox"
             aria-multiline="true"
             data-placeholder="Paste target translation"></div>
        <div id="suggestion-menu" class="suggestion-menu hidden" role="listbox"></div>
        <div class="panel-footer">
          <span id="status" class="status-text">Waiting for input.</span>
        </div>
      </div>
    </div>
  </div>
""".replace(
    "__SWAP_ICON__", _SWAP_ICON
)

_EVALUATE_JS = """
  const cfg = window.__NLLB__;
  const languages = cfg.languages;
  const sourceLanguage = document.getElementById('source-language');
  const targetLanguage = document.getElementById('target-language');
  const sourceText = document.getElementById('source-text');
  const targetEditor = document.getElementById('target-editor');
  const suggestionMenu = document.getElementById('suggestion-menu');
  const spinner = document.getElementById('spinner');
  const status = document.getElementById('status');
  const swapBtn = document.getElementById('swap-btn');

  let lastFlags = [];
  let debounceHandle = null;
  let nextRequestSeq = 0;
  let activeRequests = 0;
  let isUpdatingEditor = false;
  let activeFlagId = null;

  function setLoading(on) {
    activeRequests = Math.max(0, activeRequests + (on ? 1 : -1));
    spinner.classList.toggle('active', activeRequests > 0);
  }

  function addLanguageOptions(selectElement, defaultCode) {
    for (const code of languages) {
      const option = document.createElement('option');
      option.value = code;
      option.textContent = code;
      if (code === defaultCode) option.selected = true;
      selectElement.appendChild(option);
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function getEditorText() {
    return targetEditor.innerText;
  }

  function setEditorWithHighlights(plainText, flags) {
    isUpdatingEditor = true;
    try {
      if (!flags.length) {
        targetEditor.textContent = plainText;
        return;
      }
      const sorted = [...flags].sort((a, b) => a.char_start - b.char_start);
      let html = '';
      let cursor = 0;
      for (const flag of sorted) {
        html += escapeHtml(plainText.slice(cursor, flag.char_start));
        html += `<span class="flag" data-flag-id="${flag.id}">${escapeHtml(plainText.slice(flag.char_start, flag.char_end))}</span>`;
        cursor = flag.char_end;
      }
      html += escapeHtml(plainText.slice(cursor));
      targetEditor.innerHTML = html;
    } finally {
      isUpdatingEditor = false;
    }
  }

  function stripHighlights() {
    for (const span of targetEditor.querySelectorAll('.flag')) {
      span.replaceWith(document.createTextNode(span.textContent));
    }
    targetEditor.normalize();
  }

  function showSuggestionMenu(flag, anchorSpan) {
    activeFlagId = flag.id;

    const panelRect = suggestionMenu.parentElement.getBoundingClientRect();
    const spanRect  = anchorSpan.getBoundingClientRect();
    suggestionMenu.style.left = (spanRect.left - panelRect.left) + 'px';
    suggestionMenu.style.top  = (spanRect.bottom - panelRect.top + 4) + 'px';

    targetEditor.querySelectorAll('.flag').forEach(el => el.classList.remove('active'));
    anchorSpan.classList.add('active');

    suggestionMenu.innerHTML = '';
    if (!flag.suggestions || !flag.suggestions.length) {
      const empty = document.createElement('div');
      empty.className = 'suggestion-empty';
      empty.textContent = 'No suggestions available.';
      suggestionMenu.appendChild(empty);
    } else {
      for (const suggestion of flag.suggestions) {
        const item = document.createElement('div');
        item.className = 'suggestion-item';
        item.setAttribute('role', 'option');

        const phrase = document.createElement('span');
        phrase.className = 'suggestion-phrase';
        phrase.textContent = suggestion.phrase;

        const delta = document.createElement('span');
        delta.className = 'suggestion-delta';
        delta.textContent = `Δ=${suggestion.improvement.toFixed(3)}`;

        item.appendChild(phrase);
        item.appendChild(delta);
        item.addEventListener('mousedown', e => e.preventDefault());
        item.addEventListener('click', e => {
          e.stopPropagation();
          applyReplacement(flag.id, suggestion.phrase);
        });
        suggestionMenu.appendChild(item);
      }
    }

    suggestionMenu.classList.remove('hidden');
  }

  function hideSuggestionMenu() {
    suggestionMenu.classList.add('hidden');
    activeFlagId = null;
    targetEditor.querySelectorAll('.flag.active').forEach(el => el.classList.remove('active'));
  }

  function applyReplacement(flagId, phrase) {
    const span = targetEditor.querySelector(`[data-flag-id="${flagId}"]`);
    if (span) {
      targetEditor.focus();
      const range = document.createRange();
      range.selectNode(span);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand('insertText', false, phrase);
    }
    hideSuggestionMenu();
  }

  function scheduleScoring() {
    clearTimeout(debounceHandle);
    debounceHandle = setTimeout(runScoring, 400);
  }

  async function runScoring() {
    const translation = getEditorText().trim();
    const source = sourceText.value.trim();

    if (!translation || !source) {
      status.textContent = 'Enter both source and target text.';
      lastFlags = [];
      return;
    }

    const mySeq = ++nextRequestSeq;
    status.textContent = 'Scoring…';
    setLoading(true);

    try {
      const response = await fetch('/api/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          translation,
          source_lang: sourceLanguage.value,
          target_lang: targetLanguage.value,
        }),
      });

      if (mySeq !== nextRequestSeq) return;

      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.error || `Request failed (${response.status})`);
      }

      const data = await response.json();
      if (mySeq !== nextRequestSeq) return;

      lastFlags = data.flags || [];
      setEditorWithHighlights(translation, lastFlags);
      status.textContent = `Found ${lastFlags.length} highlighted span(s).`;
    } catch (error) {
      if (mySeq !== nextRequestSeq) return;
      status.textContent = `Error: ${error.message}`;
    } finally {
      if (mySeq === nextRequestSeq) setLoading(false);
    }
  }

  targetEditor.addEventListener('input', () => {
    if (isUpdatingEditor) return;
    nextRequestSeq++;
    stripHighlights();
    hideSuggestionMenu();
    lastFlags = [];
    scheduleScoring();
  });

  targetEditor.addEventListener('paste', event => {
    event.preventDefault();
    const text = event.clipboardData.getData('text/plain');
    const sel = window.getSelection();
    if (!sel.rangeCount) return;
    const range = sel.getRangeAt(0);
    range.deleteContents();
    const node = document.createTextNode(text);
    range.insertNode(node);
    range.setStartAfter(node);
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);
  });

  targetEditor.addEventListener('keydown', event => {
    if (event.key === 'Enter') event.preventDefault();
  });

  targetEditor.addEventListener('click', event => {
    const span = event.target.closest('.flag');
    if (!span) { hideSuggestionMenu(); return; }
    const flagId = span.dataset.flagId;
    if (activeFlagId === flagId) { hideSuggestionMenu(); return; }
    const flag = lastFlags.find(f => f.id === flagId);
    if (flag) showSuggestionMenu(flag, span);
  });

  document.addEventListener('click', event => {
    if (!suggestionMenu.contains(event.target) && !event.target.closest('.flag')) {
      hideSuggestionMenu();
    }
  });

  swapBtn.addEventListener('click', () => {
    const tmpLang = sourceLanguage.value;
    sourceLanguage.value = targetLanguage.value;
    targetLanguage.value = tmpLang;
    const tmpText = sourceText.value;
    sourceText.value = getEditorText();
    isUpdatingEditor = true;
    targetEditor.textContent = tmpText;
    isUpdatingEditor = false;
    nextRequestSeq++;
    hideSuggestionMenu();
    lastFlags = [];
    scheduleScoring();
  });

  sourceLanguage.addEventListener('change', scheduleScoring);
  targetLanguage.addEventListener('change', scheduleScoring);
  sourceText.addEventListener('input', scheduleScoring);

  addLanguageOptions(sourceLanguage, cfg.defaultSrc);
  addLanguageOptions(targetLanguage, cfg.defaultTgt);
"""


def _nav(active: str) -> str:
    def link(href: str, key: str, label: str) -> str:
        cls = ' class="active"' if key == active else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    return (
        '<nav class="tabs">'
        + link("/suggest", "suggest", "Suggest")
        + link("/evaluate", "evaluate", "Evaluate")
        + "</nav>"
    )


def _layout(title: str, active: str, page_css: str, body_html: str, config: Dict[str, Any], page_js: str) -> str:
    """Assemble a full page from the shared layout and page-specific parts.

    Everything is concatenated verbatim (never ``.format``/f-strings over the CSS/JS
    blocks), and dynamic values reach the page only through the ``window.__NLLB__``
    JSON blob, so no brace-escaping is needed in the static assets above.
    """
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        f"<title>{escape(title)}</title>",
        "<style>",
        _SHARED_CSS,
        page_css,
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        f'<div class="header-icon">{_HEADER_ICON}</div>',
        "<h1>NLLB Demo</h1>",
        _nav(active),
        '<span class="spacer"></span>',
        '<div id="spinner" class="spinner"></div>',
        "</header>",
        "<main>",
        body_html,
        "</main>",
        "<script>window.__NLLB__ = " + json.dumps(config) + ";</script>",
        "<script>",
        page_js,
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def suggest_page(languages: List[str], default_src: str, default_tgt: str) -> str:
    config = {"languages": languages, "defaultSrc": default_src, "defaultTgt": default_tgt}
    return _layout("Suggest · NLLB Demo", "suggest", _SUGGEST_CSS, _SUGGEST_BODY, config, _SUGGEST_JS)


def evaluate_page(languages: List[str], default_src: str, default_tgt: str) -> str:
    config = {"languages": languages, "defaultSrc": default_src, "defaultTgt": default_tgt}
    return _layout("Evaluate · NLLB Demo", "evaluate", _EVALUATE_CSS, _EVALUATE_BODY, config, _EVALUATE_JS)
