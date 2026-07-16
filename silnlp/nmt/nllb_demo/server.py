"""HTTP server that serves both demos over a single shared model."""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, Optional

from . import pages
from .model import DEFAULT_MODEL_NAME, DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG, NllbModel
from .scoring_service import DEFAULT_LOW_PROB_THRESHOLD, DEFAULT_TOP_K_SUGGESTIONS, ScoringService
from .suggestion_service import SuggestionService

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class NllbDemoHandler(BaseHTTPRequestHandler):
    """Routes every request to the shared suggestion or scoring service."""

    model: ClassVar[Optional[NllbModel]] = None
    suggestion_service: ClassVar[Optional[SuggestionService]] = None
    scoring_service: ClassVar[Optional[ScoringService]] = None
    default_src_lang: ClassVar[str] = DEFAULT_SOURCE_LANG
    default_tgt_lang: ClassVar[str] = DEFAULT_TARGET_LANG

    # ── Response helpers ──────────────────────────────────────────────────────

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, body: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    # ── Routing ───────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        model = self.model
        assert model is not None
        if self.path == "/":
            self._redirect("/suggest")
        elif self.path in ("/suggest", "/suggest.html"):
            self._send_html(pages.suggest_page(model.language_codes, self.default_src_lang, self.default_tgt_lang))
        elif self.path in ("/evaluate", "/evaluate.html"):
            self._send_html(pages.evaluate_page(model.language_codes, self.default_src_lang, self.default_tgt_lang))
        elif self.path == "/api/languages":
            self._send_json({"languages": model.language_codes})
        else:
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/suggest":
            self._handle_suggest()
        elif self.path == "/api/score":
            self._handle_score()
        else:
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_suggest(self) -> None:
        model = self.model
        service = self.suggestion_service
        assert model is not None and service is not None
        try:
            payload = self._read_json_body()
            source_text = str(payload.get("source_text", ""))
            partial_translation = str(payload.get("partial_translation", ""))
            src_lang = str(payload.get("src_lang", self.default_src_lang))
            tgt_lang = str(payload.get("tgt_lang", self.default_tgt_lang))
            confidence_threshold = max(0.0, min(1.0, float(payload.get("confidence_threshold", 0.25))))

            if not model.supports_language(src_lang) or not model.supports_language(tgt_lang):
                raise ValueError("Unsupported language code")

            suggestion = service.suggest(source_text, partial_translation, src_lang, tgt_lang, confidence_threshold)
            self._send_json({"suggestion": suggestion})
        except ValueError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Suggestion request failed")
            self._send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_score(self) -> None:
        model = self.model
        service = self.scoring_service
        assert model is not None and service is not None
        try:
            payload = self._read_json_body()
            source = str(payload.get("source", ""))
            translation = str(payload.get("translation", ""))
            source_lang = str(payload.get("source_lang", self.default_src_lang))
            target_lang = str(payload.get("target_lang", self.default_tgt_lang))
            scored = service.score(source, translation, source_lang, target_lang)
            self._send_json(scored)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Scoring request failed")
            self._send_json({"error": "Scoring failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the combined NLLB suggestion + evaluation demos.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Model name or path (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--src-lang", default=DEFAULT_SOURCE_LANG, help="Default source language tag")
    parser.add_argument("--tgt-lang", default=DEFAULT_TARGET_LANG, help="Default target language tag")
    parser.add_argument(
        "--low-prob-threshold",
        type=float,
        default=DEFAULT_LOW_PROB_THRESHOLD,
        help=f"Evaluate: log-prob threshold for highlights (default: {DEFAULT_LOW_PROB_THRESHOLD})",
    )
    parser.add_argument(
        "--top-k-suggestions",
        type=int,
        default=DEFAULT_TOP_K_SUGGESTIONS,
        help=f"Evaluate: alternatives returned per highlight (default: {DEFAULT_TOP_K_SUGGESTIONS})",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip the CUDA warmup pass at startup")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # The one and only model load. Both services share this instance.
    model = NllbModel.load(args.model_name)
    if args.src_lang not in model.language_codes or args.tgt_lang not in model.language_codes:
        raise ValueError("--src-lang and --tgt-lang must be valid NLLB language tags")

    NllbDemoHandler.model = model
    NllbDemoHandler.suggestion_service = SuggestionService(model)
    NllbDemoHandler.scoring_service = ScoringService(
        model,
        low_prob_threshold=args.low_prob_threshold,
        top_k_suggestions=args.top_k_suggestions,
    )
    NllbDemoHandler.default_src_lang = args.src_lang
    NllbDemoHandler.default_tgt_lang = args.tgt_lang

    if not args.no_warmup:
        model.warmup(args.src_lang, args.tgt_lang)

    server = ThreadingHTTPServer((args.host, args.port), NllbDemoHandler)
    LOGGER.info("Serving combined NLLB demo at http://%s:%d (Suggest: /suggest, Evaluate: /evaluate)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
