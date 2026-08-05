"""Building and unpacking multi-sentence context windows.

When ``data.context_size`` is greater than zero, the NMT pipeline trains and translates on windows
of sentences rather than on isolated sentences. The window for sentence ``k`` with a context size of
2 looks like::

    source_{k-2} source_{k-1} <start> source_k <end> source_{k+1} source_{k+2}

and the target side has the same shape. Training runs over the whole window, so the model learns to
emit the surrounding target context too. Scoring and drafting only care about the sentence being
translated, which is the part between the markers; :func:`find_central_segment` recovers it.
"""

import logging
import re
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

CONTEXT_START_TOKEN = "<start>"
CONTEXT_END_TOKEN = "<end>"
CONTEXT_TOKENS = [CONTEXT_START_TOKEN, CONTEXT_END_TOKEN]

_CENTRAL_SEGMENT_PATTERN = re.compile(
    re.escape(CONTEXT_START_TOKEN) + r"(.*?)" + re.escape(CONTEXT_END_TOKEN), re.DOTALL
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def build_context_window(sentences: Sequence[str], index: int, context_size: int, group: Tuple[int, int]) -> str:
    """Build the context window centered on ``sentences[index]``.

    ``group`` is the half-open range of indices the window may draw from, which keeps windows from
    spanning unrelated stretches of text (books, for instance).
    """
    if context_size <= 0:
        return sentences[index]
    group_start, group_end = group
    start = max(group_start, index - context_size)
    end = min(group_end, index + context_size + 1)
    parts = list(sentences[start:index])
    parts += [CONTEXT_START_TOKEN, sentences[index], CONTEXT_END_TOKEN]
    parts += list(sentences[index + 1 : end])
    return _collapse_whitespace(" ".join(parts))


def build_context_windows(
    sentences: Iterable[str], context_size: int, group_ids: Optional[Sequence[object]] = None
) -> List[str]:
    """Build a context window for every sentence, one window per input sentence.

    ``group_ids`` optionally assigns each sentence to a group; a window never reaches across a group
    boundary. Groups are taken to be contiguous runs, so a group id that reappears later starts a
    new group rather than rejoining the earlier one.
    """
    sentence_list = list(sentences)
    if context_size <= 0:
        return sentence_list
    if group_ids is None:
        groups = [(0, len(sentence_list))] * len(sentence_list)
    else:
        groups = _group_ranges(group_ids)
        if len(groups) != len(sentence_list):
            raise ValueError(
                f"Received {len(sentence_list)} sentences but {len(groups)} group ids; they must correspond."
            )
    return [build_context_window(sentence_list, i, context_size, groups[i]) for i in range(len(sentence_list))]


def _group_ranges(group_ids: Sequence[object]) -> List[Tuple[int, int]]:
    """Map each index to the half-open range of the contiguous run of equal group ids it belongs to."""
    ranges: List[Tuple[int, int]] = [(0, 0)] * len(group_ids)
    run_start = 0
    for i in range(1, len(group_ids) + 1):
        if i == len(group_ids) or group_ids[i] != group_ids[run_start]:
            for j in range(run_start, i):
                ranges[j] = (run_start, i)
            run_start = i
    return ranges


def iterate_context_windows(
    rows: Iterable[Sequence[str]], context_size: int
) -> Iterator[Tuple[Sequence[str], Tuple[str, ...]]]:
    """Stream context windows over parallel rows of sentences.

    Each row holds one sentence per side (source, target, and any variants); the yielded windows
    correspond position for position. Only ``2 * context_size + 1`` rows are buffered at a time, so
    this works on corpora that are too large to hold in memory. All rows are treated as one group,
    which suits a single line-aligned corpus file.
    """
    if context_size <= 0:
        for row in rows:
            yield row, tuple(row)
        return

    buffer: List[Sequence[str]] = []
    center = 0
    for row in rows:
        buffer.append(row)
        # The window for buffer[center] is only complete once context_size later rows have arrived.
        if center + context_size + 1 < len(buffer):
            yield buffer[center], _build_row_windows(buffer, center, context_size)
            center += 1
            # Drop rows that no remaining window can reach.
            if center > context_size:
                del buffer[: center - context_size]
                center = context_size
    while center < len(buffer):
        yield buffer[center], _build_row_windows(buffer, center, context_size)
        center += 1


def _build_row_windows(buffer: Sequence[Sequence[str]], center: int, context_size: int) -> Tuple[str, ...]:
    group = (0, len(buffer))
    return tuple(
        build_context_window([row[side] for row in buffer], center, context_size, group)
        for side in range(len(buffer[center]))
    )


def find_central_segment(text: str) -> Optional[str]:
    """Return the text between the context markers, or None if there is no well-formed span."""
    match = _CENTRAL_SEGMENT_PATTERN.search(text)
    if match is None:
        return None
    return _collapse_whitespace(match.group(1))


def extract_central_segment(text: str) -> str:
    """Return the text between the context markers, falling back to the whole text."""
    segment = find_central_segment(text)
    return _collapse_whitespace(text) if segment is None else segment


def find_central_token_span(tokens: List[str]) -> Optional[Tuple[int, int]]:
    """Return the half-open range of ``tokens`` between the context markers, or None if unmarked.

    Used to slice token-level confidence scores down to the same span as the extracted text.
    """
    try:
        start = tokens.index(CONTEXT_START_TOKEN)
        end = tokens.index(CONTEXT_END_TOKEN, start + 1)
    except ValueError:
        return None
    return start + 1, end


class CentralSegmentExtractor:
    """Extracts central segments, reporting how many inputs arrived without a well-formed span.

    A model can always emit malformed output, so a missing span is recoverable - the whole text is
    used instead - but it is worth telling the user how often it happened rather than silently
    scoring or drafting the surrounding context as if it were the translation.
    """

    def __init__(self, enabled: bool = True, description: str = "sequences") -> None:
        self._enabled = enabled
        self._description = description
        self._num_total = 0
        self._num_missing = 0

    def extract(self, text: str) -> str:
        if not self._enabled or text.strip() == "":
            # Blank lines are expected - references are padded out for projects with no data there -
            # so they are not worth reporting as malformed.
            return text
        self._num_total += 1
        segment = find_central_segment(text)
        if segment is None:
            self._num_missing += 1
            return _collapse_whitespace(text)
        return segment

    @property
    def num_missing(self) -> int:
        return self._num_missing

    def report(self) -> None:
        if self._num_missing == 0:
            return
        LOGGER.warning(
            f"{self._num_missing} of {self._num_total} {self._description} did not contain a well-formed "
            f"{CONTEXT_START_TOKEN} ... {CONTEXT_END_TOKEN} span. The whole output was used for those."
        )
