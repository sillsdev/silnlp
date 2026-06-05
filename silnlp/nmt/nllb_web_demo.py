from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from silnlp.common.iso_info import NLLB_TAGS

from .config import TranslationSuggester
from .hugging_face_config import (
    ConstraintIndexes,
    HuggingFaceNMTModel,
    HuggingFaceTranslationSuggester,
    SilTranslationPipeline,
    load_partial_word_constraint_indexes,
)

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "facebook/nllb-200-distilled-600M"
UTF8_ENCODING = "utf-8"


def _ensure_utf8_text(text: str) -> str:
    return text.encode(UTF8_ENCODING, errors="strict").decode(UTF8_ENCODING)


@dataclass
class SuggestionService:
    model: Any
    tokenizer: Any
    device: torch.device
    language_codes: list[str]
    constraint_indexes: ConstraintIndexes
    suggesters: dict[tuple[str, str], TranslationSuggester] = field(default_factory=dict)

    @classmethod
    def create(cls, model_name: str = MODEL_NAME) -> "SuggestionService":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        language_codes = sorted(NLLB_TAGS)
        constraint_indexes = load_partial_word_constraint_indexes(tokenizer)
        LOGGER.info("Loaded %s on %s", model_name, device)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            language_codes=language_codes,
            constraint_indexes=constraint_indexes,
        )

    def _get_suggester(self, src_lang: str, tgt_lang: str) -> TranslationSuggester:
        key = (src_lang, tgt_lang)
        if key in self.suggesters:
            return self.suggesters[key]

        pipeline_device = 0 if self.device.type == "cuda" else -1
        pipeline = SilTranslationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=pipeline_device,
        )
        suggester = HuggingFaceTranslationSuggester(
            _NllbSuggesterModelAdapter(),
            pipeline,
            self.tokenizer,
            confidence_threshold=0.25,
            max_new_tokens=64,
            num_beams=1,
            constraint_indexes=self.constraint_indexes,
        )
        self.suggesters[key] = suggester
        return suggester

    def suggest(
        self, source_text: str, partial_translation: str, src_lang: str, tgt_lang: str, confidence_threshold: float
    ) -> str:
        if not source_text.strip():
            return ""
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        suggester = self._get_suggester(src_lang, tgt_lang)
        suggester._confidence_threshold = confidence_threshold
        suggestion = suggester.suggestion_translation(source_text, partial_translation)
        if suggestion is None:
            return ""
        return _ensure_utf8_text(suggestion)


class _NllbSuggesterModelAdapter:
    # Reuse the production suggestion logic from HuggingFaceNMTModel.
    def _split_partial_translation(self, tokenizer: Any, partial_translation: str) -> tuple[str, str]:
        return HuggingFaceNMTModel._split_partial_translation(self, tokenizer, partial_translation)

    def _build_decoder_input_ids(self, tokenizer: Any, model: Any, prefix: str) -> Any:
        return HuggingFaceNMTModel._build_decoder_input_ids(self, tokenizer, model, prefix)

    def _extract_suggestion_text(
        self,
        tokenizer: Any,
        partial_translation: str,
        translation_token_ids: list[int],
        token_scores: Any,
        confidence_threshold: float,
    ) -> str | None:
        return HuggingFaceNMTModel._extract_suggestion_text(
            self,
            tokenizer,
            partial_translation,
            translation_token_ids,
            token_scores,
            confidence_threshold,
        )


def _html_page(default_src_lang: str, default_tgt_lang: str, language_codes: list[str]) -> str:
    src_options = "\n".join(
        f'<option value="{escape(code)}" {"selected" if code == default_src_lang else ""}>{escape(code)}</option>'
        for code in language_codes
    )
    tgt_options = "\n".join(
        f'<option value="{escape(code)}" {"selected" if code == default_tgt_lang else ""}>{escape(code)}</option>'
        for code in language_codes
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Translation Suggestions · NLLB</title>
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
      --mono: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
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
      padding: 0 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    .card {{
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
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

    .swap-btn:hover {{
      background: #f5f5f5;
      transform: rotate(180deg);
    }}

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

    .clear-btn {{
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
    }}

    .clear-btn.visible {{ display: flex; }}
    .clear-btn:hover {{ background: #dadce0; }}

    .ghost-wrapper {{
      position: relative;
      flex: 1;
      display: flex;
    }}

    .ghost-layer {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      overflow: hidden;
      z-index: 1;
      color: var(--text);
      white-space: pre-wrap;
      word-wrap: break-word;
    }}

    .ghost-layer .hint {{ color: var(--hint); }}
    .ghost-layer .placeholder {{ color: var(--hint); }}

    textarea.target-area {{
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
    }}

    .panel-footer {{
      display: flex;
      align-items: center;
      margin-top: 10px;
      min-height: 22px;
    }}

    .char-count {{
      font-size: 12px;
      color: var(--hint);
      margin-left: auto;
    }}

    .tab-hint {{
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      opacity: 0;
      transition: opacity .18s;
    }}

    .tab-hint.visible {{ opacity: 1; }}

    .tab-hint kbd {{
      background: #f5f5f5;
      border: 1px solid #d0d0d0;
      border-bottom-width: 2px;
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 11px;
      font-family: var(--mono);
      color: var(--text);
    }}

    .settings-card {{
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .settings-label {{
      font-size: 13px;
      font-weight: 500;
      color: var(--muted);
      flex-shrink: 0;
    }}

    input[type=range] {{
      flex: 1;
      accent-color: var(--blue);
      cursor: pointer;
    }}

    .badge {{
      background: var(--blue);
      color: #fff;
      font-size: 12px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 12px;
      min-width: 46px;
      text-align: center;
      flex-shrink: 0;
    }}

    @media (max-width: 640px) {{
      .panels {{ grid-template-columns: 1fr; }}
      .panel + .panel {{ border-left: none; border-top: 1px solid var(--border); }}
      .lang-bar {{ grid-template-columns: 1fr 44px 1fr; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="header-icon">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
      <path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/>
    </svg>
  </div>
  <h1>NLLB Translation Suggestions</h1>
  <span class="spacer"></span>
  <div id="spinner" class="spinner"></div>
</header>

<main>
  <div class="card">
    <div class="lang-bar">
      <div class="lang-picker">
        <span class="lang-badge">From</span>
        <select id="srcLang" class="lang-select">{src_options}</select>
      </div>
      <button id="swapBtn" class="swap-btn" title="Swap languages">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6.99 11L3 15l3.99 4v-3H14v-2H6.99v-3zM21 9l-3.99-4v3H10v2h7.01v3L21 9z"/>
        </svg>
      </button>
      <div class="lang-picker">
        <span class="lang-badge">To</span>
        <select id="tgtLang" class="lang-select">{tgt_options}</select>
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
</main>

<script>
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

  srcLang.value = {json.dumps(default_src_lang)};
  tgtLang.value = {json.dumps(default_tgt_lang)};

  function setLoading(on) {{
    activeRequests = Math.max(0, activeRequests + (on ? 1 : -1));
    spinner.classList.toggle('active', activeRequests > 0);
  }}

  function updateClearBtn() {{
    clearBtn.classList.toggle('visible', sourceText.value.length > 0);
  }}

  function updateCharCount() {{
    const n = sourceText.value.length;
    charCount.textContent = n > 0 ? n + ' chars' : '';
  }}

  function escapeHtml(value) {{
    return value
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;');
  }}

  function isCaretAtEnd() {{
    return (
      targetText.selectionStart === targetText.value.length &&
      targetText.selectionEnd   === targetText.value.length
    );
  }}

  function shouldShowSuggestion() {{
    return document.activeElement === targetText && isCaretAtEnd() && !!pendingSuggestion;
  }}

  function renderGhostText() {{
    const show = shouldShowSuggestion();
    if (!targetText.value && !show) {{
      ghostText.innerHTML = '<span class="placeholder">Type translation…</span>';
    }} else {{
      ghostText.innerHTML =
        escapeHtml(targetText.value) +
        (show ? '<span class="hint">' + escapeHtml(pendingSuggestion) + '</span>' : '');
    }}
    tabHint.classList.toggle('visible', show);
    syncGhostScroll();
  }}

  function syncGhostScroll() {{
    ghostText.scrollTop  = targetText.scrollTop;
    ghostText.scrollLeft = targetText.scrollLeft;
  }}

  async function requestSuggestion() {{
    const requestId = ++requestSequence;
    const payload = {{
      source_text: sourceText.value,
      partial_translation: targetText.value,
      src_lang: srcLang.value,
      tgt_lang: tgtLang.value,
      confidence_threshold: parseFloat(confidenceThreshold.value)
    }};

    if (!payload.source_text.trim() || !isCaretAtEnd()) {{
      pendingSuggestion = '';
      renderGhostText();
      return;
    }}

    setLoading(true);
    try {{
      const response = await fetch('/api/suggest', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      if (requestId !== requestSequence) return;
      if (response.ok) {{
        const data = await response.json();
        if (requestId !== requestSequence) return;
        pendingSuggestion = data.suggestion || '';
      }} else {{
        pendingSuggestion = '';
      }}
    }} catch (_) {{
      if (requestId === requestSequence) pendingSuggestion = '';
    }} finally {{
      setLoading(false);
    }}
    renderGhostText();
  }}

  function debounceSuggest() {{
    if (debounceHandle) clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {{
      requestSuggestion().catch(() => {{
        pendingSuggestion = '';
        setLoading(false);
        renderGhostText();
      }});
    }}, 300);
  }}

  function handleTargetInput() {{
    pendingSuggestion = '';
    renderGhostText();
    debounceSuggest();
  }}

  targetText.addEventListener('keydown', (e) => {{
    if (e.key === 'Tab' && pendingSuggestion && isCaretAtEnd()) {{
      e.preventDefault();
      targetText.value += pendingSuggestion;
      pendingSuggestion = '';
      renderGhostText();
      debounceSuggest();
    }}
  }});

  function handleCaretMovement() {{
    if (!isCaretAtEnd() && pendingSuggestion) pendingSuggestion = '';
    renderGhostText();
  }}

  confidenceThreshold.addEventListener('input', () => {{
    thresholdValue.textContent = parseFloat(confidenceThreshold.value).toFixed(2);
    debounceSuggest();
  }});

  sourceText.addEventListener('input', () => {{
    updateClearBtn();
    updateCharCount();
    debounceSuggest();
  }});

  clearBtn.addEventListener('click', () => {{
    sourceText.value = '';
    pendingSuggestion = '';
    updateClearBtn();
    updateCharCount();
    renderGhostText();
    sourceText.focus();
  }});

  swapBtn.addEventListener('click', () => {{
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
  }});

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
</script>
</body>
</html>
"""


class NllbDemoHandler(BaseHTTPRequestHandler):
    service: SuggestionService
    default_src_lang: str
    default_tgt_lang: str
    language_codes: set[str]

    def _set_headers(self, status: HTTPStatus, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/":
            self._set_headers(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8")
            self.wfile.write(b"Not found")
            return

        html = _html_page(self.default_src_lang, self.default_tgt_lang, sorted(self.language_codes))
        self._set_headers(HTTPStatus.OK, "text/html; charset=utf-8")
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self) -> None:
        if self.path != "/api/suggest":
            self._set_headers(HTTPStatus.NOT_FOUND, "application/json")
            self.wfile.write(b'{"error":"not found"}')
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
            source_text = str(payload.get("source_text", ""))
            partial_translation = str(payload.get("partial_translation", ""))
            src_lang = str(payload.get("src_lang", self.default_src_lang))
            tgt_lang = str(payload.get("tgt_lang", self.default_tgt_lang))
            confidence_threshold = float(payload.get("confidence_threshold", 0.25))
            confidence_threshold = max(0.0, min(1.0, confidence_threshold))

            if src_lang not in self.language_codes or tgt_lang not in self.language_codes:
                raise ValueError("Unsupported language code")

            suggestion = self.service.suggest(
                source_text, partial_translation, src_lang, tgt_lang, confidence_threshold
            )
            response = {"suggestion": suggestion}
            self._set_headers(HTTPStatus.OK, "application/json")
            self.wfile.write(json.dumps(response).encode("utf-8"))
        except Exception as error:
            LOGGER.exception("Suggestion request failed: %s", error)
            self._set_headers(HTTPStatus.BAD_REQUEST, "application/json")
            self.wfile.write(json.dumps({"error": str(error)}).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="NLLB 600M web suggestion demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--src-lang", default="eng_Latn")
    parser.add_argument("--tgt-lang", default="fra_Latn")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    service = SuggestionService.create(MODEL_NAME)
    language_codes = set(service.language_codes)

    if args.src_lang not in language_codes or args.tgt_lang not in language_codes:
        raise ValueError("--src-lang and --tgt-lang must be valid NLLB language tags for the selected model")

    NllbDemoHandler.service = service
    NllbDemoHandler.default_src_lang = args.src_lang
    NllbDemoHandler.default_tgt_lang = args.tgt_lang
    NllbDemoHandler.language_codes = language_codes

    server = ThreadingHTTPServer((args.host, args.port), NllbDemoHandler)
    LOGGER.info("Starting server at http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
