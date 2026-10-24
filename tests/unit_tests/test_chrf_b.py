import random
from pathlib import Path
from typing import List

import pytest
import sacrebleu
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef

from silnlp.common.chrf_b import ChrFB
from silnlp.common.corpus import Term
from silnlp.common.hypothesis_coverage import HypothesisCoverage
from silnlp.common.key_term_consistency import AlignedVerses, KeyTermConsistency, infer_renderings
from silnlp.nmt.test import training_corpus


def _chrf3(hypotheses, references) -> float:
    return sacrebleu.corpus_chrf(hypotheses, references, char_order=6, beta=3, remove_whitespace=True).score


def _verse_refs(vrefs: List[str]) -> List[VerseRef]:
    return [VerseRef.from_string(vref, ORIGINAL_VERSIFICATION) for vref in vrefs]


def _term(glosses: List[str], vrefs: List[str]) -> Term:
    return Term("Major:term", "KT", "", glosses, [], set(_verse_refs(vrefs)))


def test_without_coverage_or_key_terms_chrf_b_is_chrf3() -> None:
    hypotheses = ["in the beginning god made the heavens", "and the earth was empty"]
    references = [["in the beginning god created the heavens", "the earth was formless and empty"]]

    score = ChrFB(None, None).corpus_score(["s1", "s2"], hypotheses, references)

    assert score == pytest.approx(_chrf3(hypotheses, references))


def test_key_term_consistency_adds_twenty_points_per_share_of_reference_renderings_kept() -> None:
    key_terms = KeyTermConsistency({"yesu", "kristo"})
    hypotheses = ["yesu alisema", "kristo alikuja", "hakuna"]
    references = [["yesu kristo alisema", "kristo alikuja", "hakuna neno"]]

    score = ChrFB(None, key_terms).corpus_score(["s1", "s2", "s3"], hypotheses, references)

    # The third reference has no key term, so only the first two count: half kept, then all kept.
    assert score == pytest.approx(_chrf3(hypotheses, references) + 20 * (0.5 + 1.0) / 2)


def test_key_term_renderings_match_inflected_forms_but_not_short_prefixes() -> None:
    key_terms = KeyTermConsistency({"mtume", "aka"})

    scores = key_terms.verse_scores(["mitume wake", "akasema"], ["mtume wake", "aka"])

    assert scores[1] == 0.0
    assert scores[0] == 0.0
    assert key_terms.verse_scores(["mtumeni"], ["mtume"])[0] == 1.0


def test_a_confidently_aligned_rendering_is_kept() -> None:
    vrefs = [f"GEN 1:{i}" for i in range(1, 11)]
    # "covenant" aligns with "agano" in eight of its ten verses and with "na" in two.
    alignments = [[(1, 0)]] * 2 + [[(1, 1)]] * 8
    verses = AlignedVerses(
        _verse_refs(vrefs), [["the", "covenant", "is", "here"]] * 10, [["na", "agano", "iko"]] * 10, alignments
    )

    assert infer_renderings([_term(["covenant"], vrefs)], verses) == {"agano"}


def test_a_rendering_found_mostly_outside_its_terms_verses_is_dropped() -> None:
    vrefs = [f"GEN 1:{i}" for i in range(1, 11)]
    # "here" occurs in one verse, where it aligns with "na", a word in all ten verses.
    alignments = [[(3, 0)]] + [[]] * 9
    verses = AlignedVerses(
        _verse_refs(vrefs), [["the", "covenant", "is", "here"]] * 10, [["na", "agano", "iko"]] * 10, alignments
    )

    assert infer_renderings([_term(["here"], vrefs[:1])], verses) == set()


def test_a_term_verse_inside_a_verse_range_is_found_in_the_range() -> None:
    # One translation combines GEN 1:1-2, so both verses are one row, and the term occurs in GEN 1:2.
    verses = AlignedVerses(
        _verse_refs(["GEN 1:1-2", "GEN 1:3"]),
        [["the", "covenant"], ["nothing"]],
        [["agano"], ["hakuna"]],
        [[(1, 0)], []],
    )

    assert infer_renderings([_term(["covenant"], ["GEN 1:2"])], verses) == {"agano"}


def test_coverage_scores_added_words_below_the_reference() -> None:
    rng = random.Random(0)
    words = [f"w{i}" for i in range(40)]
    sources = [" ".join(rng.sample(words, 6)) for _ in range(600)]
    targets = [source.replace("w", "t") for source in sources]
    coverage = HypothesisCoverage.train(sources, targets)
    assert coverage is not None

    reference = targets[0]
    try:
        scores = coverage.verse_scores(
            [sources[0], sources[0]], [reference, reference + " zz yy xx qq"], [reference, reference]
        )
    finally:
        coverage.close()

    assert scores[0] == pytest.approx(0.0)
    assert scores[1] < -0.2


def test_coverage_needs_enough_verse_pairs_to_train_on() -> None:
    assert HypothesisCoverage.train(["a b"] * 10, ["c d"] * 10) is None


def test_training_corpus_leaves_out_verses_overlapping_the_excluded(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("one\ntwo\n<range>\nfour\nfive\n", encoding="utf-8")
    (tmp_path / "trg.txt").write_text("uno\ndos\ntres\n\ncinco\n", encoding="utf-8")

    corpus = training_corpus(tmp_path / "src.txt", tmp_path / "trg.txt", {"GEN 1:3", "GEN 1:5"})

    assert corpus["target"].tolist() == ["uno"]
