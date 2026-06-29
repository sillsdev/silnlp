import argparse
import json
import logging
import re
import threading
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Sequence

from silnlp.common.iso_info import NLLB_TAGS

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_LOW_PROB_THRESHOLD = -3.0
DEFAULT_TOP_K_SUGGESTIONS = 5
DEFAULT_SOURCE_LANG = "eng_Latn"
DEFAULT_TARGET_LANG = "fra_Latn"
SUPPORTED_LANGUAGES = sorted(set(NLLB_TAGS))


@dataclass(frozen=True)
class _FlagCandidate:
    text: str
    span_start: int
    span_end: int
    normalized_log_prob: float
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
            "normalized_log_prob": suggestion.normalized_log_prob,
            "improvement": suggestion.improvement,
        }
        for suggestion in score.suggestions
    ]


def _collect_flag_candidates(scored: Any) -> List[_FlagCandidate]:
    candidates: List[_FlagCandidate] = []
    for score in scored.low_probability_phrases:
        candidates.append(
            _FlagCandidate(
                text=score.phrase,
                span_start=score.span_start,
                span_end=score.span_end,
                normalized_log_prob=score.normalized_log_prob,
                suggestions=_build_suggestions(score),
                kind="phrase",
            )
        )

    for score in scored.low_probability_words:
        candidates.append(
            _FlagCandidate(
                text=score.word,
                span_start=score.span_start,
                span_end=score.span_end,
                normalized_log_prob=score.normalized_log_prob,
                suggestions=_build_suggestions(score),
                kind="word",
            )
        )

    return candidates


def _select_non_overlapping_flags(candidates: Sequence[_FlagCandidate]) -> List[_FlagCandidate]:
    selected: List[_FlagCandidate] = []
    for candidate in sorted(candidates, key=lambda c: (c.span_start, -c.length, c.normalized_log_prob)):
        overlaps = any(
            candidate.span_start < selected_candidate.span_end and selected_candidate.span_start < candidate.span_end
            for selected_candidate in selected
        )
        if not overlaps:
            selected.append(candidate)
    return selected


def _format_flags(scored: Any) -> List[Dict[str, Any]]:
    char_spans = _word_char_spans(scored.translation)
    flags: List[Dict[str, Any]] = []

    for index, candidate in enumerate(_select_non_overlapping_flags(_collect_flag_candidates(scored))):
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
                "normalized_log_prob": candidate.normalized_log_prob,
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
        model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        LOGGER.info("Model loaded successfully")

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
            from silnlp.nmt.translation_scorer import TranslationScorer

            scorer = TranslationScorer(
                self._model,
                self._tokenizer,
                low_prob_threshold=self._low_prob_threshold,
                top_k_suggestions=self._top_k_suggestions,
            )
            scored = scorer.score(source, translation)
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
  <meta charset=\"utf-8\">
  <title>NLLB Translation Scorer Prototype</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 1rem; background: #f6f8fa; }}
    .layout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
    .panel {{ background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 0.75rem; }}
    label {{ font-size: 0.85rem; color: #57606a; display: block; margin-bottom: 0.25rem; }}
    select, textarea {{ width: 100%; box-sizing: border-box; margin-bottom: 0.5rem; }}
    textarea {{ min-height: 160px; font-family: inherit; font-size: 1rem; padding: 0.5rem; }}
    .highlight-view {{
      min-height: 160px;
      border: 1px solid #d0d7de;
      border-radius: 6px;
      padding: 0.5rem;
      background: #fff;
      white-space: pre-wrap;
    }}
    .flag {{ background: #ffebe9; border-bottom: 2px solid #cf222e; cursor: pointer; }}
    .status {{ margin-top: 0.5rem; color: #57606a; font-size: 0.9rem; }}
    .suggestions {{ margin-top: 0.75rem; border-top: 1px solid #d0d7de; padding-top: 0.5rem; }}
    .suggestions ul {{ margin: 0.25rem 0 0; padding-left: 1.2rem; }}
  </style>
</head>
<body>
  <h2>NLLB 600M Translation Scorer Prototype</h2>
  <div class=\"layout\">
    <div class=\"panel\">
      <label for=\"source-language\">Source language code</label>
      <select id=\"source-language\"></select>
      <label for=\"source-text\">Source sentence</label>
      <textarea id=\"source-text\" placeholder=\"Paste source sentence\"></textarea>
    </div>
    <div class=\"panel\">
      <label for=\"target-language\">Target language code</label>
      <select id=\"target-language\"></select>
      <label for=\"target-text\">Target translation</label>
      <textarea id=\"target-text\" placeholder=\"Paste target translation\"></textarea>
      <label>Flagged phrases</label>
      <div id=\"highlight-view\" class=\"highlight-view\"></div>
      <div id=\"suggestions\" class=\"suggestions\">Click a highlighted phrase to view alternatives.</div>
    </div>
  </div>
  <div id=\"status\" class=\"status\">Waiting for input.</div>
  <script>
    const languages = {languages_json};
    const sourceLanguage = document.getElementById('source-language');
    const targetLanguage = document.getElementById('target-language');
    const sourceText = document.getElementById('source-text');
    const targetText = document.getElementById('target-text');
    const highlightView = document.getElementById('highlight-view');
    const suggestions = document.getElementById('suggestions');
    const status = document.getElementById('status');

    let lastFlags = [];
    let debounceHandle = null;

    function addLanguageOptions(selectElement, defaultCode) {{
      for (const code of languages) {{
        const option = document.createElement('option');
        option.value = code;
        option.textContent = code;
        if (code === defaultCode) {{
          option.selected = true;
        }}
        selectElement.appendChild(option);
      }}
    }}

    function escapeHtml(text) {{
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }}

    function renderHighlightedTranslation(text, flags) {{
      if (!text) {{
        highlightView.textContent = '';
        return;
      }}
      if (!flags.length) {{
        highlightView.textContent = text;
        return;
      }}

      let cursor = 0;
      let html = '';
      for (const flag of flags.sort((a, b) => a.char_start - b.char_start)) {{
        html += escapeHtml(text.slice(cursor, flag.char_start));
        const flaggedText = escapeHtml(text.slice(flag.char_start, flag.char_end));
        html += `<span class=\"flag\" data-flag-id=\"${{flag.id}}\">${{flaggedText}}</span>`;
        cursor = flag.char_end;
      }}
      html += escapeHtml(text.slice(cursor));
      highlightView.innerHTML = html;
    }}

    function showSuggestions(flag) {{
      if (!flag) {{
        suggestions.textContent = 'Click a highlighted phrase to view alternatives.';
        return;
      }}
      if (!flag.suggestions.length) {{
        suggestions.innerHTML = `<strong>${{escapeHtml(flag.text)}}</strong>: no suggestions available.`;
        return;
      }}
      const items = flag.suggestions
        .map((item) => `<li><strong>${{escapeHtml(item.phrase)}}</strong> (Δ=${{item.improvement.toFixed(3)}})</li>`)
        .join('');
      suggestions.innerHTML = `<strong>${{escapeHtml(flag.text)}}</strong><ul>${{items}}</ul>`;
    }}

    async function runScoring() {{
      const source = sourceText.value.trim();
      const translation = targetText.value.trim();
      showSuggestions(null);

      if (!source || !translation) {{
        status.textContent = 'Enter both source and target text.';
        lastFlags = [];
        renderHighlightedTranslation(translation, []);
        return;
      }}

      status.textContent = 'Scoring…';
      try {{
        const response = await fetch('/api/score', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            source,
            translation,
            source_lang: sourceLanguage.value,
            target_lang: targetLanguage.value
          }})
        }});

        if (!response.ok) {{
          const body = await response.json();
          throw new Error(body.error || `Request failed (${{response.status}})`);
        }}

        const data = await response.json();
        lastFlags = data.flags || [];
        renderHighlightedTranslation(data.translation, lastFlags);
        status.textContent = `Found ${{lastFlags.length}} highlighted span(s).`;
      }} catch (error) {{
        status.textContent = `Error: ${{error.message}}`;
      }}
    }}

    function scheduleScoring() {{
      clearTimeout(debounceHandle);
      debounceHandle = setTimeout(runScoring, 400);
    }}

    highlightView.addEventListener('click', (event) => {{
      const target = event.target;
      if (!target || !target.dataset.flagId) {{
        return;
      }}
      const flag = lastFlags.find((item) => item.id === target.dataset.flagId);
      showSuggestions(flag);
    }});

    addLanguageOptions(sourceLanguage, '{default_source}');
    addLanguageOptions(targetLanguage, '{default_target}');

    sourceLanguage.addEventListener('change', scheduleScoring);
    targetLanguage.addEventListener('change', scheduleScoring);
    sourceText.addEventListener('input', scheduleScoring);
    targetText.addEventListener('input', scheduleScoring);
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
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
