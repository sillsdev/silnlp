import argparse
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Sequence

import torch

from silnlp.common.iso_info import NLLB_TAGS

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_LOW_PROB_THRESHOLD = -3.0
DEFAULT_TOP_K_SUGGESTIONS = 5
DEFAULT_SOURCE_LANG = "fra_Latn"
DEFAULT_TARGET_LANG = "eng_Latn"
SUPPORTED_LANGUAGES = sorted(set(NLLB_TAGS))


@dataclass(frozen=True)
class _FlagCandidate:
    text: str
    span_start: int
    span_end: int
    mean_token_log_prob: float
    suggestions: List[Dict[str, float | str]]
    kind: str

    @property
    def length(self) -> int:
        return self.span_end - self.span_start


def _word_char_spans(text: str) -> List[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _build_suggestions(score: Any) -> List[Dict[str, float | str]]:
    return [
        {
            "phrase": suggestion.phrase,
            "mean_token_log_prob": suggestion.mean_token_log_prob,
            "improvement": suggestion.improvement,
        }
        for suggestion in score.suggestions
    ]


def _collect_flag_candidates(scored: Any) -> List[_FlagCandidate]:
    candidates: List[_FlagCandidate] = []
    for score in scored.flagged_phrases:
        candidates.append(
            _FlagCandidate(
                text=score.text,
                span_start=score.word_start,
                span_end=score.word_end,
                mean_token_log_prob=score.mean_token_log_prob,
                suggestions=_build_suggestions(score),
                kind="phrase",
            )
        )

    for score in scored.flagged_words:
        candidates.append(
            _FlagCandidate(
                text=score.text,
                span_start=score.word_start,
                span_end=score.word_end,
                mean_token_log_prob=score.mean_token_log_prob,
                suggestions=_build_suggestions(score),
                kind="word",
            )
        )

    return candidates


def _candidate_weight(candidate: _FlagCandidate) -> float:
    """Return improvement * word_length for the top suggestion, or 0 if none."""
    if not candidate.suggestions:
        return 0.0
    return max(0.0, float(candidate.suggestions[0].get("improvement", 0.0)) * candidate.length)


def _select_max_improvement_flags(candidates: Sequence[_FlagCandidate]) -> List[_FlagCandidate]:
    """Select a non-overlapping subset of flag candidates that maximises
    total_improvement = Σ (top_suggestion.improvement × span_word_length).

    Uses the standard weighted interval scheduling DP (sort by end position,
    binary-search for the latest compatible predecessor, then backtrack).
    Candidates with zero weight are excluded — they cannot improve the objective
    but can block candidates that do.
    """
    items = [(c, _candidate_weight(c)) for c in candidates if _candidate_weight(c) > 0]
    if not items:
        return []

    items.sort(key=lambda x: (x[0].span_end, x[0].span_start))
    n = len(items)

    def last_compatible(i: int) -> int:
        """Largest j < i such that items[j].span_end <= items[i].span_start."""
        target = items[i][0].span_start
        lo, hi, result = 0, i - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if items[mid][0].span_end <= target:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # dp[i] = max total improvement from a subset of items[0..i-1]  (1-indexed)
    dp = [0.0] * (n + 1)
    for i in range(1, n + 1):
        _, w = items[i - 1]
        p = last_compatible(i - 1)
        dp[i] = max(dp[i - 1], w + dp[p + 1])

    # Backtrack: at each step re-evaluate the same include/skip condition.
    selected: List[_FlagCandidate] = []
    i = n
    while i > 0:
        c, w = items[i - 1]
        p = last_compatible(i - 1)
        if w + dp[p + 1] >= dp[i - 1]:  # including item i is at least as good
            selected.append(c)
            i = p + 1
        else:
            i -= 1

    selected.reverse()
    return selected


def _format_flags(scored: Any) -> List[Dict[str, Any]]:
    char_spans = _word_char_spans(scored.translation)
    flags: List[Dict[str, Any]] = []

    for index, candidate in enumerate(_select_max_improvement_flags(_collect_flag_candidates(scored))):
        if candidate.span_end <= 0 or candidate.span_end > len(char_spans):
            continue
        char_start = char_spans[candidate.span_start][0]
        char_end = char_spans[candidate.span_end - 1][1]
        flags.append(
            {
                "id": f"flag-{index}",
                "text": candidate.text,
                "kind": candidate.kind,
                "span_start": candidate.span_start,
                "span_end": candidate.span_end,
                "char_start": char_start,
                "char_end": char_end,
                "mean_token_log_prob": candidate.mean_token_log_prob,
                "suggestions": candidate.suggestions,
            }
        )

    return flags


class NllbScoringService:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        low_prob_threshold: float = DEFAULT_LOW_PROB_THRESHOLD,
        top_k_suggestions: int = DEFAULT_TOP_K_SUGGESTIONS,
    ):
        self._model_name = model_name
        self._low_prob_threshold = low_prob_threshold
        self._top_k_suggestions = top_k_suggestions
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        LOGGER.info("Loading model %s", self._model_name)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name, torch_dtype=torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        LOGGER.info("Model loaded on device: %s", device)
        if device.type == "cuda":
            LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(device))
            LOGGER.info("VRAM allocated: %.0f MiB", torch.cuda.memory_allocated(device) / 1024**2)
            # Warm up and time a CUDA synchronization to confirm GPU is responsive.
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            torch.cuda.synchronize(device)
            LOGGER.info("CUDA synchronization latency: %.3f ms", (time.perf_counter() - t0) * 1000)

    def _configure_languages(self, source_lang: str, target_lang: str) -> None:
        assert self._tokenizer is not None
        assert self._model is not None

        def _lang_id(lang: str) -> int:
            return self._tokenizer.convert_tokens_to_ids(lang)  # type: ignore[union-attr]

        if _lang_id(source_lang) == self._tokenizer.unk_token_id:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if _lang_id(target_lang) == self._tokenizer.unk_token_id:
            raise ValueError(f"Unsupported target language: {target_lang}")

        self._tokenizer.src_lang = source_lang
        self._tokenizer.tgt_lang = target_lang
        forced_bos_token_id = _lang_id(target_lang)
        self._model.config.forced_bos_token_id = forced_bos_token_id
        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.forced_bos_token_id = forced_bos_token_id

    def warmup(self) -> None:
        with self._lock:
            self._ensure_model()
            assert self._model is not None
            assert self._tokenizer is not None

            device = next(self._model.parameters()).device
            if device.type != "cuda":
                return

            # The first CUDA forward pass triggers JIT kernel compilation and
            # cuDNN auto-tuning, which takes 15-20 s on this hardware. Running
            # dummy passes here moves that cost to server startup so the first
            # real request is fast.
            LOGGER.info("Running warmup forward passes to pre-compile CUDA kernels...")
            t0 = time.perf_counter()
            self._configure_languages(DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG)

            from transformers.modeling_outputs import BaseModelOutput

            src = self._tokenizer("warmup", return_tensors="pt")
            tgt = self._tokenizer(text_target="warmup", return_tensors="pt")
            input_ids = src["input_ids"].to(device)
            attention_mask = src["attention_mask"].to(device)
            labels = tgt["input_ids"].to(device)

            with torch.no_grad():
                encoder_out = self._model.get_encoder()(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                self._model(
                    encoder_outputs=encoder_out,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                self._model.generate(
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=encoder_out.last_hidden_state
                    ),
                    attention_mask=attention_mask,
                    num_beams=4,
                    max_new_tokens=4,
                )
            torch.cuda.synchronize(device)
            LOGGER.info("Warmup complete in %.2f s", time.perf_counter() - t0)

    def score(self, source: str, translation: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        source = source.strip()
        translation = translation.strip()
        if source == "" or translation == "":
            return {"source": source, "translation": translation, "flags": []}

        with self._lock:
            self._ensure_model()
            self._configure_languages(source_lang, target_lang)
            assert self._model is not None
            assert self._tokenizer is not None
            from silnlp.nmt.translation_scorer import AbsoluteThresholdAnomalyDetector, TranslationScorer

            scorer = TranslationScorer(
                self._model,
                self._tokenizer,
                anomaly_detector=AbsoluteThresholdAnomalyDetector(self._low_prob_threshold),
                top_k_suggestions=self._top_k_suggestions,
            )
            t0 = time.perf_counter()
            scored = scorer.score(source, translation)
            elapsed = time.perf_counter() - t0
            LOGGER.info("score() completed in %.2f s", elapsed)
            device = next(self._model.parameters()).device
            if device.type == "cuda":
                LOGGER.info(
                    "VRAM allocated: %.0f MiB, reserved: %.0f MiB",
                    torch.cuda.memory_allocated(device) / 1024**2,
                    torch.cuda.memory_reserved(device) / 1024**2,
                )
        return {
            "source": source,
            "translation": scored.translation,
            "flags": _format_flags(scored),
        }


def _html_template() -> str:
    languages_json = json.dumps(SUPPORTED_LANGUAGES)
    default_source = escape(DEFAULT_SOURCE_LANG)
    default_target = escape(DEFAULT_TARGET_LANG)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Translation Scorer · NLLB</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
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
      --shadow: 0 1px 3px rgba(0,0,0,.1), 0 4px 12px rgba(0,0,0,.06);
    }}

    body {{
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}

    header {{
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
    }}

    .header-icon {{
      width: 34px;
      height: 34px;
      background: var(--blue);
      border-radius: 8px;
      display: grid;
      place-items: center;
      flex-shrink: 0;
    }}

    header h1 {{
      font-size: 16px;
      font-weight: 500;
      letter-spacing: -.01em;
    }}

    .spacer {{ flex: 1; }}

    .spinner {{
      width: 18px;
      height: 18px;
      border: 2.5px solid var(--border);
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin .65s linear infinite;
      opacity: 0;
      transition: opacity .2s;
      flex-shrink: 0;
    }}

    .spinner.active {{ opacity: 1; }}

    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    main {{
      max-width: 1080px;
      margin: 28px auto;
      padding: 0 20px 28px;
    }}

    .card {{
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: visible;
    }}

    .lang-bar {{
      display: grid;
      grid-template-columns: 1fr 52px 1fr;
      align-items: center;
      border-bottom: 1px solid var(--border);
      height: 52px;
    }}

    .lang-picker {{
      display: flex;
      align-items: center;
      padding: 0 18px;
      gap: 8px;
      height: 100%;
    }}

    .lang-badge {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .07em;
      color: var(--muted);
      flex-shrink: 0;
    }}

    select.lang-select {{
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
    }}

    select.lang-select:hover {{ background: #f5f5f5; }}
    select.lang-select:focus {{ background: var(--blue-light); color: var(--blue); }}

    .swap-btn {{
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
    }}

    .swap-btn:hover {{ background: #f5f5f5; transform: rotate(180deg); }}

    .panels {{
      display: grid;
      grid-template-columns: 1fr 1fr;
    }}

    .panel {{
      display: flex;
      flex-direction: column;
      padding: 20px 20px 14px;
      min-height: 280px;
      position: relative;
    }}

    .panel + .panel {{ border-left: 1px solid var(--border); }}

    .editor-font {{
      font-family: var(--font);
      font-size: 18px;
      line-height: 1.65;
    }}

    textarea.source-area {{
      background: transparent;
      border: none;
      outline: none;
      resize: none;
      width: 100%;
      flex: 1;
      padding: 0;
      min-height: 200px;
      color: var(--text);
    }}

    textarea.source-area::placeholder {{ color: var(--hint); }}

    .target-editor {{
      flex: 1;
      min-height: 200px;
      width: 100%;
      padding: 0;
      outline: none;
      cursor: text;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
    }}

    .target-editor:empty::before {{
      content: attr(data-placeholder);
      color: var(--hint);
      pointer-events: none;
    }}

    .flag {{
      background: #fce8e6;
      border-bottom: 2px solid #d93025;
      border-radius: 2px;
      cursor: pointer;
    }}

    .flag:hover, .flag.active {{ background: #fad2cf; }}

    .suggestion-menu {{
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
    }}

    .suggestion-menu.hidden {{ display: none; }}

    .suggestion-item {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 14px;
      border-bottom: 1px solid var(--bg);
    }}

    .suggestion-item:last-child {{ border-bottom: none; }}
    .suggestion-item:hover {{ background: var(--bg); }}
    .suggestion-phrase {{ font-weight: 500; color: var(--text); }}
    .suggestion-delta {{ font-size: 12px; color: var(--muted); white-space: nowrap; }}
    .suggestion-empty {{ padding: 10px 16px; font-size: 14px; color: var(--muted); font-style: italic; }}

    .panel-footer {{
      display: flex;
      align-items: center;
      margin-top: 10px;
      min-height: 22px;
    }}

    .status-text {{ font-size: 12px; color: var(--muted); }}

    @media (max-width: 640px) {{
      .panels {{ grid-template-columns: 1fr; }}
      .panel + .panel {{ border-left: none; border-top: 1px solid var(--border); }}
      .lang-bar {{ grid-template-columns: 1fr 44px 1fr; }}
    }}
  </style>
</head>
<body>

<header>
  <div class=\"header-icon\">
    <svg width=\"20\" height=\"20\" viewBox=\"0 0 24 24\" fill=\"white\">
      <path d=\"M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z\"/>
    </svg>
  </div>
  <h1>NLLB Translation Scorer</h1>
  <span class=\"spacer\"></span>
  <div id=\"spinner\" class=\"spinner\"></div>
</header>

<main>
  <div class=\"card\">
    <div class=\"lang-bar\">
      <div class=\"lang-picker\">
        <span class=\"lang-badge\">From</span>
        <select id=\"source-language\" class=\"lang-select\"></select>
      </div>
      <button id=\"swap-btn\" class=\"swap-btn\" title=\"Swap languages\">
        <svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"currentColor\">
          <path d=\"M6.99 11L3 15l3.99 4v-3H14v-2H6.99v-3zM21 9l-3.99-4v3H10v2h7.01v3L21 9z\"/>
        </svg>
      </button>
      <div class=\"lang-picker\">
        <span class=\"lang-badge\">To</span>
        <select id=\"target-language\" class=\"lang-select\"></select>
      </div>
    </div>
    <div class=\"panels\">
      <div class=\"panel\">
        <textarea id=\"source-text\"
                  class=\"editor-font source-area\"
                  placeholder=\"Paste source sentence\"></textarea>
        <div class=\"panel-footer\"></div>
      </div>
      <div class=\"panel\">
        <div id=\"target-editor\"
             class=\"editor-font target-editor\"
             contenteditable=\"true\"
             spellcheck=\"false\"
             role=\"textbox\"
             aria-multiline=\"true\"
             data-placeholder=\"Paste target translation\"></div>
        <div id=\"suggestion-menu\" class=\"suggestion-menu hidden\" role=\"listbox\"></div>
        <div class=\"panel-footer\">
          <span id=\"status\" class=\"status-text\">Waiting for input.</span>
        </div>
      </div>
    </div>
  </div>
</main>

  <script>
    const languages = {languages_json};
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

    // ── Spinner ───────────────────────────────────────────────────────────────

    function setLoading(on) {{
      activeRequests = Math.max(0, activeRequests + (on ? 1 : -1));
      spinner.classList.toggle('active', activeRequests > 0);
    }}

    // ── Language selects ──────────────────────────────────────────────────────

    function addLanguageOptions(selectElement, defaultCode) {{
      for (const code of languages) {{
        const option = document.createElement('option');
        option.value = code;
        option.textContent = code;
        if (code === defaultCode) option.selected = true;
        selectElement.appendChild(option);
      }}
    }}

    function escapeHtml(text) {{
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }}

    // ── Editor helpers ────────────────────────────────────────────────────────

    function getEditorText() {{
      return targetEditor.innerText;
    }}

    function setEditorWithHighlights(plainText, flags) {{
      isUpdatingEditor = true;
      try {{
        if (!flags.length) {{
          targetEditor.textContent = plainText;
          return;
        }}
        const sorted = [...flags].sort((a, b) => a.char_start - b.char_start);
        let html = '';
        let cursor = 0;
        for (const flag of sorted) {{
          html += escapeHtml(plainText.slice(cursor, flag.char_start));
          html += `<span class="flag" data-flag-id="${{flag.id}}">${{escapeHtml(plainText.slice(flag.char_start, flag.char_end))}}</span>`;
          cursor = flag.char_end;
        }}
        html += escapeHtml(plainText.slice(cursor));
        targetEditor.innerHTML = html;
      }} finally {{
        isUpdatingEditor = false;
      }}
    }}

    function stripHighlights() {{
      for (const span of targetEditor.querySelectorAll('.flag')) {{
        span.replaceWith(document.createTextNode(span.textContent));
      }}
      targetEditor.normalize();
    }}

    // ── Suggestion menu ───────────────────────────────────────────────────────

    function showSuggestionMenu(flag, anchorSpan) {{
      activeFlagId = flag.id;

      const panelRect = suggestionMenu.parentElement.getBoundingClientRect();
      const spanRect  = anchorSpan.getBoundingClientRect();
      suggestionMenu.style.left = (spanRect.left - panelRect.left) + 'px';
      suggestionMenu.style.top  = (spanRect.bottom - panelRect.top + 4) + 'px';

      targetEditor.querySelectorAll('.flag').forEach(el => el.classList.remove('active'));
      anchorSpan.classList.add('active');

      suggestionMenu.innerHTML = '';
      if (!flag.suggestions || !flag.suggestions.length) {{
        const empty = document.createElement('div');
        empty.className = 'suggestion-empty';
        empty.textContent = 'No suggestions available.';
        suggestionMenu.appendChild(empty);
      }} else {{
        for (const suggestion of flag.suggestions) {{
          const item = document.createElement('div');
          item.className = 'suggestion-item';
          item.setAttribute('role', 'option');

          const phrase = document.createElement('span');
          phrase.className = 'suggestion-phrase';
          phrase.textContent = suggestion.phrase;

          const delta = document.createElement('span');
          delta.className = 'suggestion-delta';
          delta.textContent = `Δ=${{suggestion.improvement.toFixed(3)}}`;

          item.appendChild(phrase);
          item.appendChild(delta);
          item.addEventListener('mousedown', e => e.preventDefault());
          item.addEventListener('click', e => {{
            e.stopPropagation();
            applyReplacement(flag.id, suggestion.phrase);
          }});
          suggestionMenu.appendChild(item);
        }}
      }}

      suggestionMenu.classList.remove('hidden');
    }}

    function hideSuggestionMenu() {{
      suggestionMenu.classList.add('hidden');
      activeFlagId = null;
      targetEditor.querySelectorAll('.flag.active').forEach(el => el.classList.remove('active'));
    }}

    function applyReplacement(flagId, phrase) {{
      const span = targetEditor.querySelector(`[data-flag-id="${{flagId}}"]`);
      if (span) {{
        targetEditor.focus();
        const range = document.createRange();
        range.selectNode(span);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand('insertText', false, phrase);
      }}
      hideSuggestionMenu();
    }}

    // ── Scoring ───────────────────────────────────────────────────────────────

    function scheduleScoring() {{
      clearTimeout(debounceHandle);
      debounceHandle = setTimeout(runScoring, 400);
    }}

    async function runScoring() {{
      const translation = getEditorText().trim();
      const source = sourceText.value.trim();

      if (!translation || !source) {{
        status.textContent = 'Enter both source and target text.';
        lastFlags = [];
        return;
      }}

      const mySeq = ++nextRequestSeq;
      status.textContent = 'Scoring…';
      setLoading(true);

      try {{
        const response = await fetch('/api/score', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            source,
            translation,
            source_lang: sourceLanguage.value,
            target_lang: targetLanguage.value,
          }}),
        }});

        if (mySeq !== nextRequestSeq) return;

        if (!response.ok) {{
          const body = await response.json();
          throw new Error(body.error || `Request failed (${{response.status}})`);
        }}

        const data = await response.json();
        if (mySeq !== nextRequestSeq) return;

        lastFlags = data.flags || [];
        setEditorWithHighlights(translation, lastFlags);
        status.textContent = `Found ${{lastFlags.length}} highlighted span(s).`;
      }} catch (error) {{
        if (mySeq !== nextRequestSeq) return;
        status.textContent = `Error: ${{error.message}}`;
      }} finally {{
        if (mySeq === nextRequestSeq) setLoading(false);
      }}
    }}

    // ── Event listeners ───────────────────────────────────────────────────────

    targetEditor.addEventListener('input', () => {{
      if (isUpdatingEditor) return;
      nextRequestSeq++;
      stripHighlights();
      hideSuggestionMenu();
      lastFlags = [];
      scheduleScoring();
    }});

    targetEditor.addEventListener('paste', event => {{
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
    }});

    targetEditor.addEventListener('keydown', event => {{
      if (event.key === 'Enter') event.preventDefault();
    }});

    targetEditor.addEventListener('click', event => {{
      const span = event.target.closest('.flag');
      if (!span) {{ hideSuggestionMenu(); return; }}
      const flagId = span.dataset.flagId;
      if (activeFlagId === flagId) {{ hideSuggestionMenu(); return; }}
      const flag = lastFlags.find(f => f.id === flagId);
      if (flag) showSuggestionMenu(flag, span);
    }});

    document.addEventListener('click', event => {{
      if (!suggestionMenu.contains(event.target) && !event.target.closest('.flag')) {{
        hideSuggestionMenu();
      }}
    }});

    swapBtn.addEventListener('click', () => {{
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
    }});

    sourceLanguage.addEventListener('change', scheduleScoring);
    targetLanguage.addEventListener('change', scheduleScoring);
    sourceText.addEventListener('input', scheduleScoring);

    addLanguageOptions(sourceLanguage, '{default_source}');
    addLanguageOptions(targetLanguage, '{default_target}');
  </script>
</body>
</html>
"""


class TranslationScorerHttpHandler(BaseHTTPRequestHandler):
    scorer_service: NllbScoringService = NllbScoringService()

    def _send_json(self, body: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        response = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _send_html(self, body: str, status: int = HTTPStatus.OK) -> None:
        response = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_html(_html_template())
            return
        if self.path == "/api/languages":
            self._send_json({"languages": SUPPORTED_LANGUAGES})
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/score":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json({"error": "Request body is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        source = payload.get("source", "")
        translation = payload.get("translation", "")
        source_lang = payload.get("source_lang", DEFAULT_SOURCE_LANG)
        target_lang = payload.get("target_lang", DEFAULT_TARGET_LANG)

        try:
            scored = self.scorer_service.score(source, translation, source_lang, target_lang)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception:
            LOGGER.exception("Failed to score translation")
            self._send_json({"error": "Scoring failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(scored)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a prototype web app for NLLB translation scoring.")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--low-prob-threshold",
        type=float,
        default=DEFAULT_LOW_PROB_THRESHOLD,
        help=f"Contextual log-probability threshold for highlights (default: {DEFAULT_LOW_PROB_THRESHOLD})",
    )
    parser.add_argument(
        "--top-k-suggestions",
        type=int,
        default=DEFAULT_TOP_K_SUGGESTIONS,
        help=f"Number of alternatives to return per highlight (default: {DEFAULT_TOP_K_SUGGESTIONS})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name or path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (DEBUG shows per-request phase timings, default: INFO)",
    )
    args = parser.parse_args()

    # basicConfig is a no-op if handlers already exist (HuggingFace sets up its own
    # at import time). setLevel on the root logger always overrides.
    logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    TranslationScorerHttpHandler.scorer_service = NllbScoringService(
        model_name=args.model_name,
        low_prob_threshold=args.low_prob_threshold,
        top_k_suggestions=args.top_k_suggestions,
    )

    TranslationScorerHttpHandler.scorer_service.warmup()

    server = ThreadingHTTPServer((args.host, args.port), TranslationScorerHttpHandler)
    LOGGER.info("Serving translation scorer prototype at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
