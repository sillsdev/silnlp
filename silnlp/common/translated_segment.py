from dataclasses import dataclass

from machine.corpora import ScriptureRef


@dataclass(frozen=True)
class TranslatedSegment:
    """What the translation step produced for one segment: where it belongs, what it was translated
    from, and what it says."""

    ref: ScriptureRef
    source: str
    translation: str
