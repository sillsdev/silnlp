from typing import Dict, Iterable, Iterator, Set, Tuple
from pathlib import Path
from tqdm import tqdm

from ..common.corpus import load_corpus

SPECIAL_TOKENS: Set[str] = {"NULL", "UNKNOWN_WORD", "<UNUSED_WORD>"}


class Lexicon:
    @classmethod
    def load(cls, file_path: Path, include_special_tokens: bool = False) -> "Lexicon":
        lexicon = Lexicon()
        for line in load_corpus(file_path):
            if line.startswith("#"):
                continue
            src_word, trg_word, prob_str = line.strip().split("\t")
            if include_special_tokens or (src_word not in SPECIAL_TOKENS and trg_word not in SPECIAL_TOKENS):
                lexicon[src_word, trg_word] = float(prob_str)
        return lexicon

    @classmethod
    def symmetrize(cls, direct_lexicon: "Lexicon", inverse_lexicon: "Lexicon") -> "Lexicon":
        lexicon = Lexicon()
        # Add src/trg word pairs from the direct lexicon as long as the inverse prob is non-0
        for src_word, trg_word, direct_prob in tqdm(direct_lexicon):
            inverse_prob = inverse_lexicon[trg_word, src_word]
            prob = direct_prob * inverse_prob
            if prob > 0.0:
                lexicon[src_word, trg_word] = prob

        # Add src/trg word pairs from the inverse lexicon if the src word is not already there
        # This isn't really necessary; the direct probability will always be 0 in this case.
        for trg_word, src_word, inverse_prob in tqdm(inverse_lexicon):
            if lexicon[src_word, trg_word] == 0.0:
                direct_prob = direct_lexicon[src_word, trg_word]
                prob = direct_prob * inverse_prob
                if prob > 0.0:
                    lexicon[src_word, trg_word] = prob

        lexicon.normalize()
        return lexicon

    def __init__(self) -> None:
        self._table: Dict[str, Dict[str, float]] = {}

    def __getitem__(self, indices: Tuple[str, str]) -> float:
        src_word, trg_word = indices
        src_entry = self._table.get(src_word)
        if src_entry is None:
            return 0
        return src_entry.get(trg_word, 0)

    def __setitem__(self, indices: Tuple[str, str], value: float) -> None:
        if value == 0:
            return
        src_word, trg_word = indices
        src_entry = self._table.get(src_word)
        if src_entry is None:
            src_entry = {}
            self._table[src_word] = src_entry
        src_entry[trg_word] = value

    def __iter__(self) -> Iterator[Tuple[str, str, float]]:
        return (
            (src_word, trg_word, prob)
            for (src_word, trg_words) in self._table.items()
            for (trg_word, prob) in trg_words.items()
        )

    @property
    def source_words(self) -> Iterable[str]:
        return self._table.keys()

    @property
    def target_words(self) -> Iterable[str]:
        trg_words: Set[str] = set()
        for src_entry in self._table.values():
            trg_words.update(src_entry.keys())
        return trg_words

    def get_target_words(self, src_word: str) -> Iterable[str]:
        for trg_word, _ in self.get_target_word_probs(src_word):
            yield trg_word

    def get_target_word_probs(self, src_word: str) -> Iterable[Tuple[str, float]]:
        src_entry = self._table.get(src_word)
        if src_entry is not None:
            for trg_word, prob in sorted(src_entry.items(), key=lambda t: t[1], reverse=True):
                yield (trg_word, prob)

    def increment(self, src_word: str, trg_word: str, n: float = 1) -> None:
        if n == 0:
            return
        src_entry = self._table.get(src_word)
        if src_entry is None:
            src_entry = {}
            self._table[src_word] = src_entry
        if trg_word in src_entry:
            src_entry[trg_word] += n
        else:
            src_entry[trg_word] = n

    def normalize(self) -> None:
        for src_entry in self._table.values():
            src_entry_sum = sum(src_entry.values())
            for trg_word in src_entry.keys():
                src_entry[trg_word] /= src_entry_sum

    def add(self, lexicon: "Lexicon") -> None:
        self._table.update(lexicon._table)

    def write(self, file_path: Path) -> None:
        with file_path.open("w", encoding="utf-8", newline="\n") as file:
            for src_word, trg_word, prob in sorted(self, key=lambda t: (t[0], -t[2], t[1])):
                file.write(f"{src_word}\t{trg_word}\t{round(prob, 8)}\n")
