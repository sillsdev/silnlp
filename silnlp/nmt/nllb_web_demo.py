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
        return cls(model=model, tokenizer=tokenizer, device=device, language_codes=language_codes, constraint_indexes=constraint_indexes)

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
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>NLLB Suggestion Prototype</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    .toolbar {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
    .pane {{ display: flex; flex-direction: column; gap: 8px; }}
    .threshold-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }}
    .threshold-row label {{ white-space: nowrap; }}
    .threshold-row input[type=range] {{ flex: 1; }}
    .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    textarea {{ width: 100%; min-height: 240px; font-size: 16px; padding: 10px; box-sizing: border-box; }}
    select {{ width: 100%; padding: 8px; font-size: 14px; }}
    .suggestion-editor {{ position: relative; }}
    #ghostText,
    #targetText {{
      font-family: sans-serif;
      font-size: 16px;
      line-height: 1.4;
      padding: 10px;
      border: 1px solid #767676;
      border-radius: 2px;
      min-height: 240px;
      box-sizing: border-box;
      width: 100%;
      white-space: pre-wrap;
      word-wrap: break-word;
    }}
    #ghostText {{
      position: absolute;
      top: 0;
      left: 0;
      color: #111;
      background: #fff;
      pointer-events: none;
      overflow: hidden;
      z-index: 1;
    }}
    #ghostText .hint {{ color: #a0a0a0; }}
    #ghostText .placeholder {{ color: #9a9a9a; }}
    #targetText {{
      position: relative;
      background: transparent;
      color: transparent;
      caret-color: #111;
      overflow: auto;
      resize: vertical;
      z-index: 2;
    }}
  </style>
</head>
<body>
  <h2>NLLB 600M Translation Suggestion Prototype</h2>
  <div class=\"toolbar\">
    <div class=\"pane\">
      <label for=\"srcLang\">Source language code</label>
      <select id=\"srcLang\">{src_options}</select>
    </div>
    <div class=\"pane\">
      <label for=\"tgtLang\">Target language code</label>
      <select id=\"tgtLang\">{tgt_options}</select>
    </div>
  </div>
  <div class="threshold-row">
    <label for="confidenceThreshold">Confidence threshold: <strong id="thresholdValue">0.25</strong></label>
    <input type="range" id="confidenceThreshold" min="0" max="1" step="0.05" value="0.25" />
  </div>
  <div class=\"columns\">
    <div class=\"pane\">
      <label for=\"sourceText\">Source sentence</label>
      <textarea id=\"sourceText\" placeholder=\"Paste source sentence...\"></textarea>
    </div>
    <div class=\"pane\">
      <label for=\"targetText\">Translation</label>
      <div class="suggestion-editor">
        <div id="ghostText" aria-hidden="true"></div>
        <textarea id=\"targetText\" placeholder=\"Type translation...\"></textarea>
      </div>
    </div>
  </div>

  <script>
    const sourceText = document.getElementById('sourceText');
    const targetText = document.getElementById('targetText');
    const srcLang = document.getElementById('srcLang');
    const tgtLang = document.getElementById('tgtLang');
    const ghostText = document.getElementById('ghostText');
    const confidenceThreshold = document.getElementById('confidenceThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    let pendingSuggestion = '';
    let debounceHandle = null;
    let requestSequence = 0;

    srcLang.value = {json.dumps(default_src_lang)};
    tgtLang.value = {json.dumps(default_tgt_lang)};

    function escapeHtml(value) {{
      return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }}

    function isCaretAtEnd() {{
      return (
        targetText.selectionStart === targetText.value.length &&
        targetText.selectionEnd === targetText.value.length
      );
    }}

    function shouldShowSuggestion() {{
      return document.activeElement === targetText && isCaretAtEnd() && !!pendingSuggestion;
    }}

    function renderGhostText() {{
      if (!targetText.value && !pendingSuggestion) {{
        ghostText.innerHTML = '<span class="placeholder">Type translation...</span>';
      }} else {{
        const suggestionText = shouldShowSuggestion() ? pendingSuggestion : '';
        ghostText.innerHTML = `${{escapeHtml(targetText.value)}}<span class="hint">${{escapeHtml(suggestionText)}}</span>`;
      }}
      syncGhostScroll();
    }}

    function syncGhostScroll() {{
      ghostText.scrollTop = targetText.scrollTop;
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

      const response = await fetch('/api/suggest', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      if (!response.ok) {{
        if (requestId !== requestSequence) {{
          return;
        }}
        pendingSuggestion = '';
        renderGhostText();
        return;
      }}
      const data = await response.json();
      if (requestId !== requestSequence) {{
        return;
      }}
      pendingSuggestion = data.suggestion || '';
      renderGhostText();
    }}

    function debounceSuggest() {{
      if (debounceHandle) clearTimeout(debounceHandle);
      debounceHandle = setTimeout(() => {{
        requestSuggestion().catch(() => {{
          pendingSuggestion = '';
          renderGhostText();
        }});
      }}, 300);
    }}

    function handleTargetInput() {{
      // Invalidate current completion immediately on any manual edit.
      if (pendingSuggestion) {{
        pendingSuggestion = '';
      }}
      renderGhostText();
      debounceSuggest();
    }}

    targetText.addEventListener('keydown', (event) => {{
      if (event.key === 'Tab' && pendingSuggestion && isCaretAtEnd()) {{
        event.preventDefault();
        targetText.value += pendingSuggestion;
        pendingSuggestion = '';
        renderGhostText();
        debounceSuggest();
      }}
    }});

    function handleCaretMovement() {{
      if (!isCaretAtEnd() && pendingSuggestion) {{
        pendingSuggestion = '';
      }}
      renderGhostText();
    }}

    confidenceThreshold.addEventListener('input', () => {{
      thresholdValue.textContent = parseFloat(confidenceThreshold.value).toFixed(2);
      debounceSuggest();
    }});

    sourceText.addEventListener('input', debounceSuggest);
    targetText.addEventListener('input', handleTargetInput);
    targetText.addEventListener('click', handleCaretMovement);
    targetText.addEventListener('keyup', handleCaretMovement);
    targetText.addEventListener('select', handleCaretMovement);
    targetText.addEventListener('focus', renderGhostText);
    targetText.addEventListener('blur', renderGhostText);
    targetText.addEventListener('scroll', syncGhostScroll);
    srcLang.addEventListener('change', debounceSuggest);
    tgtLang.addEventListener('change', debounceSuggest);

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
