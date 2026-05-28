from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from silnlp.common.iso_info import NLLB_TAGS

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "facebook/nllb-200-distilled-600M"


@dataclass
class SuggestionService:
    model: Any
    tokenizer: Any
    device: torch.device

    @classmethod
    def create(cls, model_name: str = MODEL_NAME) -> "SuggestionService":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        LOGGER.info("Loaded %s on %s", model_name, device)
        return cls(model=model, tokenizer=tokenizer, device=device)

    def suggest(self, source_text: str, partial_translation: str, src_lang: str, tgt_lang: str) -> str:
        if not source_text.strip():
            return ""
        self.tokenizer.src_lang = src_lang
        model_inputs = self.tokenizer(source_text, return_tensors="pt", truncation=True)
        model_inputs = {name: tensor.to(self.device) for name, tensor in model_inputs.items()}
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=64,
                num_beams=4,
            )

        full_translation = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return _remaining_completion(partial_translation, full_translation)


def _remaining_completion(partial_translation: str, full_translation: str) -> str:
    if len(full_translation) == 0:
        return ""

    if len(partial_translation) == 0:
        return full_translation

    if full_translation.startswith(partial_translation):
        return full_translation[len(partial_translation) :]

    trimmed_partial = partial_translation.rstrip()
    if full_translation.startswith(trimmed_partial):
        return full_translation[len(trimmed_partial) :]

    return ""


def _html_page(default_src_lang: str, default_tgt_lang: str) -> str:
    options = "\n".join(
        f'<option value="{escape(code)}" {"selected" if code in {default_src_lang, default_tgt_lang} else ""}>{escape(code)}</option>'
        for code in NLLB_TAGS
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
    .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    textarea {{ width: 100%; min-height: 240px; font-size: 16px; padding: 10px; box-sizing: border-box; }}
    select {{ width: 100%; padding: 8px; font-size: 14px; }}
    #suggestion {{ margin-top: 8px; color: #0a6; min-height: 20px; }}
  </style>
</head>
<body>
  <h2>NLLB 600M Translation Suggestion Prototype</h2>
  <div class=\"toolbar\">
    <div class=\"pane\">
      <label for=\"srcLang\">Source language code</label>
      <select id=\"srcLang\">{options}</select>
    </div>
    <div class=\"pane\">
      <label for=\"tgtLang\">Target language code</label>
      <select id=\"tgtLang\">{options}</select>
    </div>
  </div>
  <div class=\"columns\">
    <div class=\"pane\">
      <label for=\"sourceText\">Source sentence</label>
      <textarea id=\"sourceText\" placeholder=\"Paste source sentence...\"></textarea>
    </div>
    <div class=\"pane\">
      <label for=\"targetText\">Translation</label>
      <textarea id=\"targetText\" placeholder=\"Type translation...\"></textarea>
      <div id=\"suggestion\"></div>
    </div>
  </div>

  <script>
    const sourceText = document.getElementById('sourceText');
    const targetText = document.getElementById('targetText');
    const srcLang = document.getElementById('srcLang');
    const tgtLang = document.getElementById('tgtLang');
    const suggestion = document.getElementById('suggestion');
    let pendingSuggestion = '';
    let debounceHandle = null;

    srcLang.value = {json.dumps(default_src_lang)};
    tgtLang.value = {json.dumps(default_tgt_lang)};

    async function requestSuggestion() {{
      const payload = {{
        source_text: sourceText.value,
        partial_translation: targetText.value,
        src_lang: srcLang.value,
        tgt_lang: tgtLang.value
      }};

      if (!payload.source_text.trim()) {{
        pendingSuggestion = '';
        suggestion.textContent = '';
        return;
      }}

      const response = await fetch('/api/suggest', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      if (!response.ok) {{
        pendingSuggestion = '';
        suggestion.textContent = '';
        return;
      }}
      const data = await response.json();
      pendingSuggestion = data.suggestion || '';
      suggestion.textContent = pendingSuggestion ? `Tab to accept: ${{pendingSuggestion}}` : '';
    }}

    function debounceSuggest() {{
      if (debounceHandle) clearTimeout(debounceHandle);
      debounceHandle = setTimeout(() => {{
        requestSuggestion().catch(() => {{
          pendingSuggestion = '';
          suggestion.textContent = '';
        }});
      }}, 300);
    }}

    targetText.addEventListener('keydown', (event) => {{
      if (event.key === 'Tab' && pendingSuggestion) {{
        event.preventDefault();
        targetText.value += pendingSuggestion;
        pendingSuggestion = '';
        suggestion.textContent = '';
        debounceSuggest();
      }}
    }});

    sourceText.addEventListener('input', debounceSuggest);
    targetText.addEventListener('input', debounceSuggest);
    srcLang.addEventListener('change', debounceSuggest);
    tgtLang.addEventListener('change', debounceSuggest);
  </script>
</body>
</html>
"""


class NllbDemoHandler(BaseHTTPRequestHandler):
    service: SuggestionService
    default_src_lang: str
    default_tgt_lang: str

    def _set_headers(self, status: HTTPStatus, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/":
            self._set_headers(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8")
            self.wfile.write(b"Not found")
            return

        html = _html_page(self.default_src_lang, self.default_tgt_lang)
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

            if src_lang not in NLLB_TAGS or tgt_lang not in NLLB_TAGS:
                raise ValueError("Unsupported language code")

            suggestion = self.service.suggest(source_text, partial_translation, src_lang, tgt_lang)
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

    if args.src_lang not in NLLB_TAGS or args.tgt_lang not in NLLB_TAGS:
        raise ValueError("--src-lang and --tgt-lang must be valid NLLB language tags")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    service = SuggestionService.create(MODEL_NAME)

    NllbDemoHandler.service = service
    NllbDemoHandler.default_src_lang = args.src_lang
    NllbDemoHandler.default_tgt_lang = args.tgt_lang

    server = ThreadingHTTPServer((args.host, args.port), NllbDemoHandler)
    LOGGER.info("Starting server at http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
