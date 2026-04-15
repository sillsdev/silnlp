from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .word_alignments import WordAlignments


@dataclass
class SubPassage:
    source_verse_token_offsets: List[int]
    source_tokens: List[str]
    target_tokens: List[str]
    word_alignments: Optional[WordAlignments] = None

    def get_token_separated_source_text_for_alignment(self) -> str:
        return " ".join(self.source_tokens)

    def get_token_separated_target_text_for_alignment(self) -> str:
        return " ".join(self.target_tokens)

    def to_json(self) -> Dict[str, Any]:
        return {
            "source_verse_token_offsets": self.source_verse_token_offsets,
            "source_tokens": " ".join(self.source_tokens),
            "target_tokens": " ".join(self.target_tokens),
            "word_alignments": self.word_alignments.to_json() if self.word_alignments is not None else None,
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "SubPassage":
        return SubPassage(
            source_verse_token_offsets=data["source_verse_token_offsets"],
            source_tokens=data["source_tokens"].split(),
            target_tokens=data["target_tokens"].split(),
            word_alignments=(
                WordAlignments.from_json(data["word_alignments"]) if data["word_alignments"] is not None else None
            ),
        )
