from pathlib import Path
from typing import List

from .paratext_project_reader import ParatextProjectReader
from .passage import Passage, PassageReader, SegmentedPassage, SegmentedPassageBuilder


class ReferenceVerseSegmentationReader:
    def read_passages(self, target_project_name: str, target_passage_file: Path):
        passages = PassageReader(target_passage_file).get_passages()
        return self._read_segmented_passages(passages, target_project_name)

    def _read_segmented_passages(self, passages: List[Passage], target_project_name: str) -> List[SegmentedPassage]:
        paratext_project_reader = ParatextProjectReader(target_project_name)
        segmented_passage_builders: List[SegmentedPassageBuilder] = [
            SegmentedPassageBuilder(p.start_ref, p.end_ref) for p in passages
        ]
        paratext_project_reader.collect_verses(segmented_passage_builders)
        return [segmented_passage_builder.build() for segmented_passage_builder in segmented_passage_builders]


class SegmentationEvaluation:
    def __init__(self):
        self._num_breaks = 0
        self._num_correct_breaks = 0
        self._total_distance = 0

    def record_verse_break(self, distance_from_reference: int) -> None:
        self._num_breaks += 1
        if distance_from_reference == 0:
            self._num_correct_breaks += 1
        self._total_distance += distance_from_reference

    def get_accuracy(self) -> float:
        return self._num_correct_breaks / self._num_breaks if self._num_breaks > 0 else 0.0

    def get_average_distance(self) -> float:
        return self._total_distance / self._num_breaks if self._num_breaks > 0 else 0.0


class SegmentationEvaluator:
    def __init__(self, reference_segmentations: List[SegmentedPassage]):
        self._reference_segmentations = reference_segmentations

    def evaluate_segmentation(self, segmented_passages: List[SegmentedPassage]) -> SegmentationEvaluation:
        if len(self._reference_segmentations) != len(segmented_passages):
            raise ValueError("Number of segmented passages does not match number of reference segmentations")

        segmentation_evaluation = SegmentationEvaluation()
        for reference_passage, segmented_passage in zip(self._reference_segmentations, segmented_passages):
            if (
                reference_passage.start_ref != segmented_passage.start_ref
                or reference_passage.end_ref != segmented_passage.end_ref
            ):
                raise ValueError(
                    f"Passage references do not match: {reference_passage.start_ref}-{reference_passage.end_ref} vs {segmented_passage.start_ref}-{segmented_passage.end_ref}"
                )
            self._check_passage_lengths(reference_passage, segmented_passage)
            self._evaluate_single_passage(reference_passage, segmented_passage, segmentation_evaluation)
        return segmentation_evaluation

    def _check_passage_lengths(self, reference_passage: SegmentedPassage, segmented_passage: SegmentedPassage) -> None:
        reference_length = sum(len(self._normalize_verse_text(verse.text)) for verse in reference_passage.verses)
        segmented_length = sum(
            len(self._normalize_verse_text(verse.text)) for verse in segmented_passage.verses if len(verse.text) > 0
        )
        if reference_length != segmented_length:
            raise ValueError(
                f"Passage lengths do not match for {reference_passage.start_ref}-{reference_passage.end_ref}: {reference_length} vs {segmented_length}"
            )

    def _normalize_verse_text(self, text: str) -> str:
        return text.replace(" ", "")

    def _evaluate_single_passage(
        self,
        reference_passage: SegmentedPassage,
        segmented_passage: SegmentedPassage,
        segmentation_evaluation: SegmentationEvaluation,
    ) -> None:
        reference_running_character_offset = 0
        segmented_running_character_offset = 0
        for reference_verse, segmented_verse in zip(reference_passage.verses[:-1], segmented_passage.verses[:-1]):
            reference_running_character_offset += len(self._normalize_verse_text(reference_verse.text))
            segmented_running_character_offset += len(self._normalize_verse_text(segmented_verse.text))
            distance = abs(reference_running_character_offset - segmented_running_character_offset)
            segmentation_evaluation.record_verse_break(distance)
