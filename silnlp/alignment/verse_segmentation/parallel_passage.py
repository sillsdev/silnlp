import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional

from machine.scripture import VerseRef
from machine.tokenization import LatinWordTokenizer
from tqdm import tqdm

from .alignment_generators import AbstractAlignmentGeneratorFactory, AlignmentGenerator
from .break_scorers import AbstractBreakScorerFactory
from .paratext_project_reader import ParatextProjectReader
from .passage import Passage, PassageReader, SegmentedPassage, Verse, VerseCollector, VerseRange
from .passage_splitter import PassageSplitter, PassageSplitterFactory
from .sub_passage import SubPassage
from .verse_offset_predictors import AbstractVerseOffsetPredictorFactory
from .verse_segmenter import VerseSegmenter
from .word_alignments import WordAlignments


class ParallelPassage(VerseRange):

    def __init__(
        self,
        start_ref: VerseRef,
        end_ref: VerseRef,
        source_verses: List[Verse],
        target_text: str,
        word_alignments: WordAlignments,
        tokenized_source_verses: Optional[List[List[str]]] = None,
        tokenized_source_text: Optional[List[str]] = None,
        tokenized_target_text: Optional[List[str]] = None,
        verse_token_offsets: Optional[List[int]] = None,
        sub_passages: Optional[List["SubPassage"]] = None,
    ):
        super().__init__(start_ref, end_ref)
        self._start_ref = start_ref
        self._end_ref = end_ref
        self._source_verses = source_verses
        self._target_text = target_text
        self._word_alignments = word_alignments

        if tokenized_source_verses is not None:
            self._tokenized_source_verses = tokenized_source_verses
        else:
            self._tokenized_source_verses: List[List[str]] = []

        if tokenized_source_text is not None:
            self._tokenized_source_text = tokenized_source_text
        else:
            self._tokenized_source_text = []

        if tokenized_target_text is not None:
            self._tokenized_target_text = tokenized_target_text
        else:
            self._tokenized_target_text = []

        if verse_token_offsets is not None:
            self._source_verse_token_offsets = verse_token_offsets
        else:
            self._source_verse_token_offsets = []

        if sub_passages is not None:
            self._sub_passages = sub_passages
        else:
            self._sub_passages = []

    def segment_target_passage(
        self, verse_offset_predictor_factory: AbstractVerseOffsetPredictorFactory
    ) -> SegmentedPassage:
        verse_segmenter: VerseSegmenter = VerseSegmenter(
            verse_offset_predictor_factory,
            self._tokenized_source_text,
            self._tokenized_target_text,
            self._target_text,
            self._source_verse_token_offsets,
            self._word_alignments,
            self._sub_passages,
        )
        target_verses: List[Verse] = verse_segmenter.segment_verses(
            [verse.reference for verse in self._source_verses],
        )
        return SegmentedPassage(self._start_ref, self._end_ref, self._target_text, target_verses)

    def get_source_passage(self) -> SegmentedPassage:
        return SegmentedPassage(
            self._start_ref, self._end_ref, " ".join([v.text for v in self._source_verses]), self._source_verses
        )

    def to_json(self) -> Dict[str, Any]:

        return {
            "start_ref": str(self._start_ref),
            "end_ref": str(self._end_ref),
            "source_verses": [{"reference": str(v.reference), "text": v.text} for v in self._source_verses],
            "target_text": self._target_text,
            "word_alignments": self._word_alignments.to_json(),
            "tokenized_source_verses": [" ".join(tokens) for tokens in self._tokenized_source_verses],
            "tokenized_source_text": " ".join(self._tokenized_source_text),
            "tokenized_target_text": " ".join(self._tokenized_target_text),
            "verse_token_offsets": self._source_verse_token_offsets,
            "sub_passages": [sub_passage.to_json() for sub_passage in self._sub_passages],
        }

    @classmethod
    def from_json(cls, parallel_passage_json: Dict[str, Any]) -> "ParallelPassage":
        return cls(
            start_ref=VerseRef.from_string(parallel_passage_json["start_ref"]),
            end_ref=VerseRef.from_string(parallel_passage_json["end_ref"]),
            source_verses=[
                Verse(VerseRef.from_string(v["reference"]), v["text"]) for v in parallel_passage_json["source_verses"]
            ],
            target_text=parallel_passage_json["target_text"],
            word_alignments=WordAlignments.from_json(parallel_passage_json["word_alignments"]),
            tokenized_source_verses=[v.split() for v in parallel_passage_json.get("tokenized_source_verses", [])],
            tokenized_source_text=parallel_passage_json.get("tokenized_source_text", "").split(),
            tokenized_target_text=parallel_passage_json.get("tokenized_target_text", "").split(),
            verse_token_offsets=parallel_passage_json.get("verse_token_offsets", []),
            sub_passages=[SubPassage.from_json(sp) for sp in parallel_passage_json.get("sub_passages", [])],
        )


class ParallelPassageBuilder(VerseCollector):
    tokenizer = LatinWordTokenizer()

    def __init__(self, start_ref: VerseRef, end_ref: VerseRef, target_text: str):
        super().__init__(start_ref, end_ref)
        self._target_text = target_text
        self._tokenized_target_text = list(self.tokenizer.tokenize(self._target_text))
        self._source_verses: List[Verse] = []
        self._word_alignments: WordAlignments
        self._tokenized_source_verses: List[List[str]] = []
        self._tokenized_source_text: List[str] = []
        self._verse_token_offsets: List[int] = []
        self._sub_passages: Optional[List["SubPassage"]] = None

    def add_verse(self, verse: Verse) -> None:
        self._source_verses.append(verse)
        tokens = list(self.tokenizer.tokenize(verse.text))
        self._tokenized_source_verses.append(tokens)
        self._tokenized_source_text.extend(tokens)
        self._verse_token_offsets.append(len(self._tokenized_source_text))

    def get_token_separated_source_text_for_alignment(self) -> str:
        return " ".join(self._tokenized_source_text)

    def get_token_separated_target_text_for_alignment(self) -> str:
        return " ".join(self._tokenized_target_text)

    def set_word_alignments(self, word_alignments: WordAlignments) -> None:
        self._word_alignments = word_alignments

    def split_into_subpassages(
        self, passage_splitter_factory: "PassageSplitterFactory", break_scorer_factory: AbstractBreakScorerFactory
    ) -> None:
        passage_splitter: "PassageSplitter" = passage_splitter_factory.create(
            self._tokenized_source_text,
            self._tokenized_target_text,
            self._verse_token_offsets,
            self._word_alignments,
            break_scorer_factory,
        )
        self._sub_passages = passage_splitter.split_into_minimal_segments()

    def get_sub_passages(self) -> List["SubPassage"]:
        if self._sub_passages is None:
            return [SubPassage(self._verse_token_offsets, self._tokenized_source_text, self._tokenized_target_text)]
        return self._sub_passages

    def build(self) -> ParallelPassage:
        if self._word_alignments is None:
            raise ValueError("Word alignments not loaded")
        if (
            self._sub_passages is not None
            and len(self._sub_passages) > 0
            and any(sub_passage.word_alignments is None for sub_passage in self._sub_passages)
        ):
            raise ValueError("Sub-passage word alignments not loaded")
        return ParallelPassage(
            self.start_ref,
            self.end_ref,
            self._source_verses,
            self._target_text,
            self._word_alignments,
            self._tokenized_source_verses,
            self._tokenized_source_text,
            self._tokenized_target_text,
            self._verse_token_offsets,
            self._sub_passages,
        )


class ParallelPassageCollection:
    def __init__(self, parallel_passages: List[ParallelPassage]):
        self._parallel_passages = parallel_passages

    def segment_target_passages(
        self, verse_offset_predictor_factory: AbstractVerseOffsetPredictorFactory
    ) -> Iterable[SegmentedPassage]:
        for passage in self._parallel_passages:
            yield passage.segment_target_passage(verse_offset_predictor_factory)

    def get_source_segmented_passages(self) -> List[SegmentedPassage]:
        return [parallel_passage.get_source_passage() for parallel_passage in self._parallel_passages]

    def to_json(self) -> List[Dict[str, Any]]:
        return [parallel_passage.to_json() for parallel_passage in self._parallel_passages]


class ParallelPassageCollectionCreator(ABC):
    @abstractmethod
    def create(self, source_project_name: str, target_passage_file: Path) -> "ParallelPassageCollection": ...


class ParatextParallelPassageCollectionCreator(ParallelPassageCollectionCreator):
    def __init__(
        self,
        alignment_generator_factory: AbstractAlignmentGeneratorFactory,
        break_scorer_factory: AbstractBreakScorerFactory,
        save_alignments: bool = False,
        subdivide_passages: bool = False,
    ):
        self._save_alignments = save_alignments
        self._subdivide_passages = subdivide_passages
        self._alignment_generator_factory = alignment_generator_factory
        self._break_scorer_factory = break_scorer_factory

    def create(self, source_project_name: str, target_passage_file: Path) -> "ParallelPassageCollection":
        target_passages = PassageReader(target_passage_file).get_passages()
        parallel_passage_builders = self._collect_parallel_passages(source_project_name, target_passages)
        alignment_generator = self._alignment_generator_factory.create(target_passage_file)
        self._create_word_alignment_matrix(parallel_passage_builders, alignment_generator)

        if self._subdivide_passages:
            self._split_into_subpassages(parallel_passage_builders, alignment_generator)

        parallel_passages = [parallel_passage_builder.build() for parallel_passage_builder in parallel_passage_builders]
        parallel_passage_collection = ParallelPassageCollection(parallel_passages)

        if self._save_alignments:
            with open(target_passage_file.with_suffix(".saved.json"), "w", encoding="utf-8") as f:
                json.dump(parallel_passage_collection.to_json(), f, ensure_ascii=False, indent=2)
        return parallel_passage_collection

    def _collect_parallel_passages(
        self, source_project_name: str, target_passages: List[Passage]
    ) -> List[ParallelPassageBuilder]:
        paratext_project_reader: ParatextProjectReader = ParatextProjectReader(source_project_name)
        parallel_passage_builders = [ParallelPassageBuilder(t.start_ref, t.end_ref, t.text) for t in target_passages]
        return paratext_project_reader.collect_verses(parallel_passage_builders)

    def _create_word_alignment_matrix(
        self, parallel_passage_builders: List[ParallelPassageBuilder], alignment_generator: "AlignmentGenerator"
    ) -> None:
        for index, word_alignments in enumerate(
            alignment_generator.generate(
                [
                    passage_builder.get_token_separated_source_text_for_alignment()
                    for passage_builder in parallel_passage_builders
                ],
                [
                    passage_builder.get_token_separated_target_text_for_alignment()
                    for passage_builder in parallel_passage_builders
                ],
            )
        ):
            parallel_passage_builders[index].set_word_alignments(word_alignments)

    def _split_into_subpassages(
        self, parallel_passage_builders: List[ParallelPassageBuilder], alignment_generator: "AlignmentGenerator"
    ) -> None:
        passage_splitter_factory = PassageSplitterFactory()
        sub_passages: List[SubPassage] = []

        for parallel_passage in tqdm(parallel_passage_builders, desc="Splitting passages into smaller chunks"):
            parallel_passage.split_into_subpassages(passage_splitter_factory, self._break_scorer_factory)
            sub_passages.extend(parallel_passage.get_sub_passages())

        for index, word_alignments in enumerate(
            alignment_generator.generate(
                [sub_passage.get_token_separated_source_text_for_alignment() for sub_passage in sub_passages],
                [sub_passage.get_token_separated_target_text_for_alignment() for sub_passage in sub_passages],
            )
        ):
            sub_passages[index].word_alignments = word_alignments


class SavedParallelPassageCollectionCreator(ParallelPassageCollectionCreator):

    def create(self, source_project_name: str, target_passage_file: Path) -> "ParallelPassageCollection":
        with open(target_passage_file.with_suffix(".saved.json"), "r", encoding="utf-8") as f:
            saved_passages_json = json.load(f)
            saved_passages = [ParallelPassage.from_json(p) for p in saved_passages_json]
            return ParallelPassageCollection(saved_passages)
        with open(target_passage_file.with_suffix(".saved.json"), "r", encoding="utf-8") as f:
            saved_passages_json = json.load(f)
            saved_passages = [ParallelPassage.from_json(p) for p in saved_passages_json]
            return ParallelPassageCollection(saved_passages)
