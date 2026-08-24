import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from silnlp.nmt import score as score_mod
from silnlp.nmt import test as legacy_test_mod
from silnlp.nmt.test import score_pair as legacy_score_pair
from silnlp.nmt.test import write_pair_verse_scores as legacy_write_pair_verse_scores

HYP = [
    "The Quick Brown Fox.",
    "",
    "a b c d e f g h",
    "Bonjour, le monde!",
]
REF_A = [
    "The quick brown fox!",
    "something here",
    "a b c d e f g h",
    "Bonjour le monde.",
]
REF_B = [
    "A quick brown fox.",
    "",
    "a b c d",
    "Bonjour, le Monde !",
]

ALL_SCORERS_EXCEPT_CONFIDENCE = [s for s in score_mod.SUPPORTED_SCORERS if s not in ("confidence", "spbleu")]


def _stub_config(exp_dir: Path, sacrebleu_tokenize: str = "13a") -> SimpleNamespace:
    return SimpleNamespace(data={"sacrebleu_tokenize": sacrebleu_tokenize}, exp_dir=exp_dir)


def _spbleu_model_available() -> bool:
    import os

    sacrebleu_dir = os.environ.get("SACREBLEU", str(Path.home() / ".sacrebleu"))
    return (Path(sacrebleu_dir) / "models" / "flores200sacrebleuspm").exists()


# --- Drift tests: silnlp/nmt/score.py must produce identical numbers to silnlp/nmt/test.py ---


def test_supported_scorers_match_legacy():
    # A scorer added to test.py's lists without a matching update here would otherwise be
    # silently unavailable through this CLI - the other drift tests only exercise scorers
    # that are already in both lists, so they can't catch that on their own.
    assert score_mod.SUPPORTED_SCORERS == legacy_test_mod.SUPPORTED_SCORERS
    assert score_mod.SUPPORTED_SENTENCE_SCORERS == legacy_test_mod.SUPPORTED_SENTENCE_SCORERS


@pytest.mark.parametrize("sacrebleu_tokenize", ["13a", "intl", "none"])
@pytest.mark.parametrize(
    "scorers",
    [
        {"bleu"},
        {"chrf3"},
        {"chrf3+"},
        {"chrf3++"},
        {"m-bleu"},
        {"m-chrf3"},
        {"m-chrf3+"},
        {"m-chrf3++"},
        {"ter"},
        set(ALL_SCORERS_EXCEPT_CONFIDENCE),
    ],
)
def test_compute_scores_matches_legacy_score_pair(tmp_path, sacrebleu_tokenize, scorers):
    config = _stub_config(tmp_path, sacrebleu_tokenize)
    pair_refs = [REF_A, REF_B]

    new_bleu, new_other = score_mod.compute_scores(HYP, pair_refs, scorers, sacrebleu_tokenize=sacrebleu_tokenize)

    # book="NOTALL" so the legacy call doesn't also try to write a verse-scores file
    old_pair_score = legacy_score_pair(
        HYP, pair_refs, "NOTALL", "en", "xx", "unused.detok.txt", "unused.conf.tsv", scorers, config, set()
    )

    if "bleu" in scorers:
        assert old_pair_score.bleu is not None
        assert new_bleu.score == pytest.approx(old_pair_score.bleu.score)
        assert new_bleu.precisions == old_pair_score.bleu.precisions
        assert new_bleu.bp == pytest.approx(old_pair_score.bleu.bp)
        assert new_bleu.sys_len == old_pair_score.bleu.sys_len
        assert new_bleu.ref_len == old_pair_score.bleu.ref_len
    else:
        assert new_bleu is None
        assert old_pair_score.bleu is None

    assert new_other.keys() == old_pair_score.other_scores.keys()
    for key in new_other:
        assert new_other[key] == pytest.approx(old_pair_score.other_scores[key])


def test_confidence_scorer_matches_legacy(tmp_path):
    config = _stub_config(tmp_path)
    confidences = [0.9, 0.5, 0.8, 0.65]
    scorers = {"confidence"}

    _, new_other = score_mod.compute_scores(HYP, [REF_A], scorers, confidences=confidences)
    old_pair_score = legacy_score_pair(
        HYP,
        [REF_A],
        "NOTALL",
        "en",
        "xx",
        "unused.detok.txt",
        "unused.conf.tsv",
        scorers,
        config,
        set(),
        pair_confs=confidences,
    )

    assert new_other["Confidence"] == pytest.approx(old_pair_score.other_scores["Confidence"])


@pytest.mark.skipif(not _spbleu_model_available(), reason="FLORES-200 SPM model not downloaded")
def test_spbleu_matches_legacy(tmp_path):
    config = _stub_config(tmp_path)
    scorers = {"spbleu"}

    _, new_other = score_mod.compute_scores(HYP, [REF_A], scorers)
    old_pair_score = legacy_score_pair(
        HYP, [REF_A], "NOTALL", "en", "xx", "unused.detok.txt", "unused.conf.tsv", scorers, config, set()
    )

    assert new_other["spBLEU"] == pytest.approx(old_pair_score.other_scores["spBLEU"])


def test_write_verse_scores_matches_legacy_byte_for_byte(tmp_path):
    config = _stub_config(tmp_path)
    scorers = {"bleu", "chrf3", "chrf3+", "chrf3++", "ter"}
    pair_refs = [REF_A, REF_B]
    _, other_scores = score_mod.compute_scores(HYP, pair_refs, scorers)

    new_path = tmp_path / "new.scores.tsv"
    score_mod.write_verse_scores(new_path, HYP, pair_refs, scorers, other_scores, "13a", None)

    old_detok_name = "old.detok.txt"
    legacy_write_pair_verse_scores(HYP, pair_refs, "xx", old_detok_name, scorers, other_scores, config, None, None)
    old_path = tmp_path / (old_detok_name + score_mod.VERSE_SCORES_SUFFIX)

    assert new_path.read_text(encoding="utf-8") == old_path.read_text(encoding="utf-8")


# --- Unit tests for behavior that is easy to get subtly wrong ---


def test_chrf3_is_case_sensitive_but_bleu_is_not():
    lower_bleu, lower_other = score_mod.compute_scores(["hello world"], [["hello world"]], {"bleu", "chrf3"})
    upper_bleu, upper_other = score_mod.compute_scores(["Hello World"], [["hello world"]], {"bleu", "chrf3"})

    assert lower_bleu.score == pytest.approx(upper_bleu.score)
    assert lower_other["chrF3"] != pytest.approx(upper_other["chrF3"])


def test_chrf3_plus_differs_from_chrf3():
    _, other = score_mod.compute_scores([HYP[0]], [[REF_A[0]]], {"chrf3", "chrf3+"})
    assert other["chrF3"] != pytest.approx(other["chrF3+"])


def test_blank_hypothesis_line_still_counts_toward_sent_len():
    bleu, _ = score_mod.compute_scores(["", "hello"], [["", "hello"]], {"bleu"})
    assert bleu.sys_len >= 0
    # sacrebleu scores the empty string, it does not drop the pair
    assert bleu is not None


def test_confidence_scorer_requires_confidences():
    with pytest.raises(ValueError):
        score_mod.compute_scores(["hello"], [["hello"]], {"confidence"})


def test_version_guard_raises_on_mismatch(monkeypatch):
    monkeypatch.setattr(score_mod.sacrebleu, "__version__", "9.9.9")
    with pytest.raises(SystemExit):
        score_mod.check_versions(allow_mismatch=False)


def test_version_guard_warns_under_allow_mismatch(monkeypatch):
    monkeypatch.setattr(score_mod.sacrebleu, "__version__", "9.9.9")
    score_mod.check_versions(allow_mismatch=True)  # must not raise


def _write_lines(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_text_mode_and_table_mode_agree(tmp_path, monkeypatch):
    hyp_path = tmp_path / "hyp.detok.txt"
    ref_path = tmp_path / "ref.detok.txt"
    _write_lines(hyp_path, HYP)
    _write_lines(ref_path, REF_A)

    text_out = tmp_path / "text-scores.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "silnlp-nmt-score",
            "--hyp",
            str(hyp_path),
            "--ref",
            str(ref_path),
            "--scorers",
            "bleu",
            "chrf3",
            "--out",
            str(text_out),
        ],
    )
    score_mod.main()

    table_path = tmp_path / "table.csv"
    with open(table_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hyp", "ref"])
        for h, r in zip(HYP, REF_A):
            writer.writerow([h, r])

    table_out = tmp_path / "table-scores.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "silnlp-nmt-score",
            "--hyp",
            str(table_path),
            "--hyp-col",
            "hyp",
            "--ref-col",
            "ref",
            "--scorers",
            "bleu",
            "chrf3",
            "--out",
            str(table_out),
        ],
    )
    score_mod.main()

    assert text_out.read_text(encoding="utf-8") == table_out.read_text(encoding="utf-8")


def test_by_book_sent_len_sums_to_all(tmp_path, monkeypatch):
    hyp_lines = ["h1", "h2", "h3", "h4"]
    ref_lines = ["r1", "r2", "r3", "r4"]
    vref_lines = ["GEN 1:1", "GEN 1:2", "EXO 1:1", "EXO 1:2"]

    hyp_path = tmp_path / "hyp.detok.txt"
    ref_path = tmp_path / "ref.detok.txt"
    vref_path = tmp_path / "vref.txt"
    _write_lines(hyp_path, hyp_lines)
    _write_lines(ref_path, ref_lines)
    _write_lines(vref_path, vref_lines)

    out_path = tmp_path / "scores.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "silnlp-nmt-score",
            "--hyp",
            str(hyp_path),
            "--ref",
            str(ref_path),
            "--vref",
            str(vref_path),
            "--by-book",
            "--scorers",
            "bleu",
            "--out",
            str(out_path),
        ],
    )
    score_mod.main()

    with open(out_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    all_row = next(r for r in rows if r["book"] == "ALL")
    book_rows = [r for r in rows if r["book"] != "ALL"]
    assert len(book_rows) == 2
    assert sum(int(r["sent_len"]) for r in book_rows) == int(all_row["sent_len"])
