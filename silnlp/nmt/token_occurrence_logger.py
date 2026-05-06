import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple, IO

LOGGER = logging.getLogger(__name__)

_MAX_LOGGED_OCCURRENCE_LINES = 10  # Maximum number of line numbers to log per token per corpus file

# SentencePiece word-boundary prefix character (U+2581 LOWER ONE EIGHTH BLOCK)
_SP_WORD_BOUNDARY = "\u2581"


def _get_unicode_details(token: str) -> str:
    """Return a semicolon-separated string of Unicode code-point descriptions for each character in token."""
    parts = []
    for char in token:
        cp = ord(char)
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "UNKNOWN"
        parts.append(f"U+{cp:04X} {name}")
    return "; ".join(parts)


def _build_search_pattern(token: str) -> Optional[re.Pattern]:
    """Build a regex pattern used to find token occurrences in raw corpus text.

    The SentencePiece word-boundary prefix ▁ (U+2581) is stripped from the token
    before searching because it does not appear in raw corpus text.  When the token
    originally carried the prefix, a \\b word-boundary anchor is prepended to the
    pattern so that only whole-word (or word-start) matches are counted, avoiding
    spurious mid-word hits.
    """
    has_word_boundary = token.startswith(_SP_WORD_BOUNDARY)
    search_token = token.replace(_SP_WORD_BOUNDARY, "").strip()
    if not search_token:
        return None
    escaped = re.escape(search_token)
    if has_word_boundary:
        return re.compile(r"\b" + escaped)
    return re.compile(escaped)


def _count_file_occurrences(file_path: Path, pattern: re.Pattern, max_lines: int) -> Tuple[int, List[int], bool]:
    """Scan file_path for matches of pattern and return occurrence statistics.

    Returns:
        total_count: number of matches found (up to and including the line where
            max_lines is reached; scanning stops after that line for efficiency).
        occurrence_lines: 1-based line numbers of the first max_lines matching lines.
        truncated: True when the file contained more matching lines than max_lines.
    """
    occurrence_lines: List[int] = []
    total_count = 0
    truncated = False
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                matches = len(pattern.findall(line))
                if matches > 0:
                    total_count += matches
                    if len(occurrence_lines) < max_lines:
                        occurrence_lines.append(line_num)
                    else:
                        truncated = True
                        break
    except OSError:
        pass
    return total_count, occurrence_lines, truncated


class TokenOccurrenceLogger:
    """Logs details about tokens being added to the tokenizer.

    For each token the logger emits:
    - the token (repr) and the Unicode code point/name for every character it contains
    - for each corpus file: the total occurrence count and up to
      ``max_lines`` 1-based line numbers where the token was found

    Use as a context manager to keep the log file open during logging operations:
        with TokenOccurrenceLogger(file_paths, output_path) as logger:
            logger.log(missing_tokens)
    """

    def __init__(
        self, file_paths: List[Path], output_path: Path, max_lines: int = _MAX_LOGGED_OCCURRENCE_LINES
    ) -> None:
        self._file_paths = file_paths
        self._output_path = output_path / "token_occurrence.log"
        self._max_lines = max_lines
        self._file: Optional[IO] = None

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.touch(exist_ok=True)

    def __enter__(self) -> "TokenOccurrenceLogger":
        """Open the log file for writing when entering the context."""
        self._file = open(self._output_path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the log file when exiting the context."""
        if self._file:
            self._file.close()
            self._file = None

    def log(self, missing_tokens: List[str]) -> None:
        """Log details for each token in missing_tokens."""
        if not missing_tokens:
            return
        self._log_message(f"\nLogging occurrence details for {len(missing_tokens)} missing tokens...")
        for token in missing_tokens:
            self._log_token(token)

    def _log_token(self, token: str) -> None:
        self._log_message(f"\nToken: {repr(token)}\nUnicode: [{_get_unicode_details(token)}]\n")
        pattern = _build_search_pattern(token)
        if pattern is None:
            return
        for file_path in self._file_paths:
            self._log_file_occurrences(file_path, pattern)

    def _log_file_occurrences(self, file_path: Path, pattern: re.Pattern) -> None:
        total_count, occurrence_lines, truncated = _count_file_occurrences(file_path, pattern, self._max_lines)
        if total_count > 0:
            lines_str = str(occurrence_lines)
            if truncated:
                lines_str += f" (showing first {self._max_lines} of more)"
            self._log_message(f"  File: {file_path.name}\n  Occurrences: {total_count}\n  Lines: {lines_str}\n")

    def _log_message(self, message: str) -> None:
        """Log a message to both the logger and the file.

        The log file must be open when this is called (i.e., within a context manager).
        """
        LOGGER.info(message)
        if self._file:
            self._file.write(message + "\n")
