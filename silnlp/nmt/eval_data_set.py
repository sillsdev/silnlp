from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from ..common.utils import add_tags_to_dataframe


class EvalDataSet:
    """Validation or test data, gathered per language pair as each corpus pair contributes to it."""

    def __init__(self, selected_verses: Optional[Dict[Tuple[str, str], Set[int]]] = None) -> None:
        self._corpora: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._selected_verses: Dict[Tuple[str, str], Set[int]] = (
            {} if selected_verses is None else selected_verses
        )

    def add(
        self, src_iso: str, trg_iso: str, target_project: str, tags: List[str], corpus: pd.DataFrame
    ) -> None:
        if len(corpus) == 0:
            return
        add_tags_to_dataframe(tags, corpus)

        language_pair = (src_iso, trg_iso)
        gathered = self._corpora.get(language_pair)
        # The first project to arrive fixes which verses the rest have to cover.
        if language_pair not in self._selected_verses:
            self._selected_verses[language_pair] = set(corpus.index)

        corpus.rename(columns={"target": f"target_{target_project}"}, inplace=True)
        if gathered is None:
            gathered = corpus
        else:
            gathered = gathered.combine_first(corpus)
            gathered.fillna("", inplace=True)
        self._corpora[language_pair] = gathered

    def indices_for(self, src_iso: str, trg_iso: str, default: Optional[Set[int]]) -> Optional[Set[int]]:
        return self._selected_verses.get((src_iso, trg_iso), default)

    def is_empty(self) -> bool:
        return len(self._corpora) == 0

    def total_size(self) -> int:
        return sum(len(corpus) for corpus in self._corpora.values())

    def language_pairs(self) -> Iterable[Tuple[Tuple[str, str], pd.DataFrame]]:
        return self._corpora.items()
