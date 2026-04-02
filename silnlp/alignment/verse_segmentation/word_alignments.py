from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Dict, List

from machine.corpora import AlignedWordPair


class WordAlignments:
    def __init__(self, src_length: int, trg_length: int, aligned_pairs: Collection[AlignedWordPair]):
        self._src_length = src_length
        self._trg_length = trg_length
        self._aligned_pairs = aligned_pairs
        self._target_tokens_by_source_token: Dict[int, List[int]] = self._create_source_to_target_alignment_lookup(
            aligned_pairs
        )
        self._source_tokens_by_target_token: Dict[int, List[int]] = self._create_target_to_source_alignment_lookup(
            aligned_pairs
        )
        self._cached_crossed_alignments: Dict[tuple[int, int], float] = {}

    def _create_source_to_target_alignment_lookup(
        self, aligned_pairs: Collection[AlignedWordPair]
    ) -> Dict[int, List[int]]:
        target_tokens_by_source_token: Dict[int, List[int]] = defaultdict(list)
        for aligned_word_pair in aligned_pairs:
            target_tokens_by_source_token[aligned_word_pair.source_index].append(aligned_word_pair.target_index)
        return target_tokens_by_source_token

    def _create_target_to_source_alignment_lookup(
        self, aligned_pairs: Collection[AlignedWordPair]
    ) -> Dict[int, List[int]]:
        source_tokens_by_target_token: Dict[int, List[int]] = defaultdict(list)
        for aligned_word_pair in aligned_pairs:
            source_tokens_by_target_token[aligned_word_pair.target_index].append(aligned_word_pair.source_index)
        return source_tokens_by_target_token

    def get_target_aligned_words(self, source_word_index: int) -> List[int]:
        return self._target_tokens_by_source_token.get(source_word_index) or []

    def get_source_aligned_words(self, target_word_index: int) -> List[int]:
        return self._source_tokens_by_target_token.get(target_word_index) or []

    def get_num_crossed_alignments(self, src_word_index: int, trg_word_index: int) -> float:
        if (src_word_index, trg_word_index) in self._cached_crossed_alignments:
            return self._cached_crossed_alignments[(src_word_index, trg_word_index)]
        num_crossings = 0
        for aligned_word_pair in self._aligned_pairs:
            # By convention, a break at "word_index" is placed immediately before
            # the word at words[word_index]
            if (
                aligned_word_pair.source_index < src_word_index and aligned_word_pair.target_index >= trg_word_index
            ) or (aligned_word_pair.source_index >= src_word_index and aligned_word_pair.target_index < trg_word_index):
                num_crossings += 1

        self._cached_crossed_alignments[(src_word_index, trg_word_index)] = num_crossings
        return num_crossings

    def remove_links_crossing_n(self, n: int, other_alignment: "WordAlignments") -> "WordAlignments":
        pairs_to_retain: List[AlignedWordPair] = []
        for aligned_pair in self._aligned_pairs:
            if other_alignment.get_num_crossed_alignments(aligned_pair.source_index, aligned_pair.target_index) < n:
                pairs_to_retain.append(aligned_pair)
        return WordAlignments(self._src_length, self._trg_length, pairs_to_retain)

    def append_to_file(self, output_file: Path) -> None:
        with output_file.open("w") as f:
            for aligned_pair in self._aligned_pairs:
                f.write(f"{aligned_pair.source_index}-{aligned_pair.target_index} ")
            f.write("\n")

    def to_json(self) -> Dict[str, Any]:
        return {
            "src_length": self._src_length,
            "trg_length": self._trg_length,
            "aligned_pairs": " ".join(
                [f"{aligned_pair.source_index}-{aligned_pair.target_index}" for aligned_pair in self._aligned_pairs]
            ),
        }

    @classmethod
    def from_json(cls, word_alignment_json: Dict[str, Any]) -> "WordAlignments":
        src_length = word_alignment_json["src_length"]
        trg_length = word_alignment_json["trg_length"]
        aligned_pairs = []
        for pair_str in word_alignment_json["aligned_pairs"].split():
            source_index, target_index = map(int, pair_str.split("-"))
            aligned_pairs.append(AlignedWordPair(source_index, target_index))
        return cls(src_length=src_length, trg_length=trg_length, aligned_pairs=aligned_pairs)


class WordAlignmentsBuilder:
    def __init__(self, src_length: int, trg_length: int):
        self._src_length = src_length
        self._trg_length = trg_length
        self._aligned_pairs: List[AlignedWordPair] = []

    def add_aligments(self, aligned_pairs: Collection[AlignedWordPair]) -> None:
        self._aligned_pairs.extend(aligned_pairs)

    def build(self) -> WordAlignments:
        return WordAlignments(self._src_length, self._trg_length, self._aligned_pairs)
