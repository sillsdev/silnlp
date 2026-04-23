import argparse
from pathlib import Path
from typing import Dict, List

from machine.scripture import ORIGINAL_VERSIFICATION

from silnlp.common.environment import SIL_NLP_ENV

from .alignment_generators import (
    EflomalAlignmentGeneratorFactory,
    FastAlignAlignmentGeneratorFactory,
    FastAlignConstrainedEflomalAlignmentGeneratorFactory,
)
from .break_scorers import ManualBreakScorerFactory
from .parallel_passage import ParatextParallelPassageCollectionCreator, SavedParallelPassageCollectionCreator
from .passage import SegmentedPassage
from .segmentation_evaluator import ReferenceVerseSegmentationReader, SegmentationEvaluator
from .verse_offset_predictors import ScoringFunctionVerseOffsetPredictorFactory


def evaluate_against_reference(
    project_to_compare_against: str, target_passages_path: Path, trg_segmented_passages: List[SegmentedPassage]
) -> None:
    reference_segmentations = ReferenceVerseSegmentationReader().read_passages(
        project_to_compare_against, target_passages_path
    )

    trg_gold_path = target_passages_path.with_suffix(".trg.gold.txt")
    with open(trg_gold_path, "w", encoding="utf-8") as trg_gold_output:
        for reference_passage in reference_segmentations:
            reference_passage.write_to_file(trg_gold_output)

    segmentation_evaluator = SegmentationEvaluator(reference_segmentations)
    segmentation_evaluation = segmentation_evaluator.evaluate_segmentation(trg_segmented_passages)
    print(
        f"Segmentation accuracy = {segmentation_evaluation.get_accuracy() * 100}%, average distance = {segmentation_evaluation.get_average_distance()} characters"
    )


def write_results_to_file(
    target_passages_path: Path,
    src_segmented_passages: List[SegmentedPassage],
    trg_segmented_passages: List[SegmentedPassage],
    write_vrefs_file: bool,
) -> None:
    src_path = target_passages_path.with_suffix(".src.txt")
    trg_path = target_passages_path.with_suffix(".trg.txt")
    with open(src_path, "w", encoding="utf-8") as src_output, open(trg_path, "w", encoding="utf-8") as trg_output:
        for src_passage, trg_passage in zip(src_segmented_passages, trg_segmented_passages):
            src_passage.write_to_file(src_output)
            trg_passage.write_to_file(trg_output)

    if write_vrefs_file:
        vref_path = target_passages_path.with_suffix(".vref.txt")
        template_vref_path = SIL_NLP_ENV.assets_dir / "vref.txt"

        verse_map: Dict[str, str] = {
            str(verse.reference.to_versification(ORIGINAL_VERSIFICATION)): verse.text
            for trg_passage in trg_segmented_passages
            for verse in trg_passage.verses
        }

        with (
            open(template_vref_path, "r", encoding="utf-8") as template_file,
            open(vref_path, "w", encoding="utf-8") as vref_output,
        ):
            for line in template_file:
                template_ref = line.rstrip("\n")
                vref_output.write(f"{verse_map.get(template_ref, '')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verse counts and compute alignment scores")
    parser.add_argument("--source-project", help="Name of source Paratext project", required=True, type=str)
    parser.add_argument("--target-passages", help=".tsv file with target passages", required=True, type=str)
    parser.add_argument(
        "--compare-against", help="Name of Paratext project where target passages come from", default=None, type=str
    )
    parser.add_argument(
        "--save-alignments", help="Save the computed alignments for future use", default=None, action="store_true"
    )
    parser.add_argument(
        "--use-saved-alignments",
        help="Use pre-computed alignments from a previous run",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--recursive",
        help="Recursively split passages (can improve segmentation accuracy, but increases run time)",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--alignment-method",
        help="Method to use for computing alignments",
        default="combined",
        choices=["fast_align", "eflomal", "combined"],
    )
    parser.add_argument(
        "--alignment-runs",
        help="Number of times to run alignment and average the results (must be >= 1)",
        default=1,
        type=int,
    )
    parser.add_argument("--vref", help="Output vref file for target verses", default=None, action="store_true")
    args = parser.parse_args()

    # There is currently only one option for a break scorer, but more may be added in the future
    break_scorer_factory = ManualBreakScorerFactory()

    if args.use_saved_alignments:
        parallel_passages = SavedParallelPassageCollectionCreator().create(
            args.source_project, Path(args.target_passages)
        )
    else:
        if args.alignment_method == "fast_align":
            alignment_generator_factory = FastAlignAlignmentGeneratorFactory()
        elif args.alignment_method == "eflomal":
            alignment_generator_factory = EflomalAlignmentGeneratorFactory(args.alignment_runs)
        else:
            alignment_generator_factory = FastAlignConstrainedEflomalAlignmentGeneratorFactory(args.alignment_runs)

        parallel_passages = ParatextParallelPassageCollectionCreator(
            alignment_generator_factory,
            break_scorer_factory,
            save_alignments=args.save_alignments,
            subdivide_passages=args.recursive,
        ).create(args.source_project, Path(args.target_passages))
    src_segmented_passages = parallel_passages.get_source_segmented_passages()

    verse_offset_predictor_factory = ScoringFunctionVerseOffsetPredictorFactory(
        break_scorer_factory=break_scorer_factory
    )
    trg_segmented_passages: List[SegmentedPassage] = list(
        parallel_passages.segment_target_passages(verse_offset_predictor_factory)
    )

    write_results_to_file(
        Path(args.target_passages),
        src_segmented_passages,
        trg_segmented_passages,
        write_vrefs_file=args.vref is not None,
    )

    if args.compare_against is not None:
        evaluate_against_reference(args.compare_against, Path(args.target_passages), trg_segmented_passages)


if __name__ == "__main__":
    main()
