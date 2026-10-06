import math
from pathlib import Path
from typing import List

import pytest

from silnlp.common import translator
from silnlp.common.translation_data_structures import (
    SentenceTranslation,
    SequenceConfidence,
    TokenWeightedConfidence,
    TranslatedDraft,
)


def create_translation(num_tokens: int, mean_log_prob: float) -> SentenceTranslation:
    return SentenceTranslation("text", ["text"], [mean_log_prob] * num_tokens, mean_log_prob)


def create_unscored_translation() -> SentenceTranslation:
    return SentenceTranslation("text", ["text"], [], None)


def test_token_weighted_confidence_weights_sequences_by_token_count() -> None:
    confidence = TokenWeightedConfidence()
    confidence.add_sequence(math.log(0.9), 1)
    confidence.add_sequence(math.log(0.5), 3)

    assert confidence.compute() == pytest.approx((0.9 * 0.5**3) ** (1 / 4))


def test_token_weighted_confidence_with_no_tokens_is_nan() -> None:
    assert math.isnan(TokenWeightedConfidence().compute())


def test_token_weighted_confidence_from_sequence_confidences() -> None:
    confidence = TokenWeightedConfidence.from_sequence_confidences(
        [SequenceConfidence(0.9, 2), SequenceConfidence(0.6, 6)]
    )

    assert confidence.compute() == pytest.approx((0.9**2 * 0.6**6) ** (1 / 8))


def test_compute_confidence_weights_sentences_by_token_count() -> None:
    draft = TranslatedDraft([create_translation(2, math.log(0.9)), create_translation(6, math.log(0.6))])

    assert draft.compute_confidence([0, 1]) == pytest.approx((0.9**2 * 0.6**6) ** (1 / 8))


def test_compute_confidence_only_includes_requested_sentences() -> None:
    draft = TranslatedDraft([create_translation(2, math.log(0.9)), create_translation(6, math.log(0.6))])

    assert draft.compute_confidence([1]) == pytest.approx(0.6)


def test_compute_overall_confidence_skips_unscored_sentences() -> None:
    draft = TranslatedDraft(
        [create_translation(2, math.log(0.9)), create_unscored_translation(), create_translation(6, math.log(0.6))]
    )

    assert draft.compute_overall_confidence() == pytest.approx((0.9**2 * 0.6**6) ** (1 / 8))


def test_combine_weights_sequence_score_by_token_count() -> None:
    combined = SentenceTranslation.combine([create_translation(2, math.log(0.9)), create_translation(6, math.log(0.6))])

    assert combined.get_sequence_confidence_score() == pytest.approx((0.9**2 * 0.6**6) ** (1 / 8))


def test_combined_translation_has_the_token_count_of_its_parts() -> None:
    combined = SentenceTranslation.combine([create_translation(2, math.log(0.9)), create_translation(6, math.log(0.6))])
    draft = TranslatedDraft([combined, create_translation(4, math.log(0.5))])

    assert draft.compute_overall_confidence() == pytest.approx((0.9**2 * 0.6**6 * 0.5**4) ** (1 / 12))


def test_combine_without_a_score_for_every_part_has_no_sequence_score() -> None:
    combined = SentenceTranslation.combine([create_translation(2, math.log(0.9)), create_unscored_translation()])

    assert not combined.has_sequence_confidence_score()


def test_test_confidence_file_reads_back_token_counts(tmp_path: Path) -> None:
    confidence_file = translator.TestConfidenceFile(tmp_path / "test.trg-predictions.txt.8.confidences.tsv")
    draft = TranslatedDraft(
        [
            SentenceTranslation("a b", ["▁a", "▁b", "</s>"], [-0.1, -0.2, -0.3], -0.2),
            create_translation(5, math.log(0.6)),
        ]
    )
    confidence_file.generate_confidence_files(draft)

    sequence_confidences: List[SequenceConfidence] = confidence_file.get_sequence_confidences()

    assert [sc.num_tokens for sc in sequence_confidences] == [3, 5]
    assert sequence_confidences[0].confidence == pytest.approx(math.exp(-0.2))
    assert sequence_confidences[1].confidence == pytest.approx(0.6)
