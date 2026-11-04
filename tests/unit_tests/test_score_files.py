from pathlib import Path

import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.score_files import ScoreFiles

AVERAGED_CHECKPOINT = -1


def score_files_for(pairs, environment, exp_dir: Path) -> ScoreFiles:
    exp_dir.mkdir(parents=True, exist_ok=True)
    return ScoreFiles(exp_dir, CorpusInventory(pairs, include_glosses=False, environment=environment))


@pytest.fixture
def exp_dir(tmp_path) -> Path:
    return tmp_path / "exp"


@pytest.fixture
def one_iso_pair(corpora, environment, exp_dir):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    return score_files_for(pairs, environment, exp_dir)


@pytest.fixture
def three_iso_pairs(corpora, environment, exp_dir):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("fr", "LSG")]),
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("de", "LUT")]),
    ]
    return score_files_for(pairs, environment, exp_dir)


@pytest.fixture
def two_iso_pairs(corpora, environment, exp_dir):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("fr", "LSG")]),
    ]
    return score_files_for(pairs, environment, exp_dir)


@pytest.fixture
def two_test_projects(corpora, environment, exp_dir):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "RVR")]),
    ]
    return score_files_for(pairs, environment, exp_dir)


def only_test_set(files, step=2, books=None):
    test_sets = files.test_sets(step, books if books is not None else {})
    assert len(test_sets) == 1
    return test_sets[0]


def test_no_test_data_until_a_test_set_is_written(one_iso_pair, exp_dir):
    assert not one_iso_pair.has_test_data()

    (exp_dir / "test.src.txt").touch()

    assert one_iso_pair.has_test_data()


def test_a_combined_test_set_is_named_without_the_isos(one_iso_pair, exp_dir):
    (exp_dir / "test.src.txt").touch()

    test_set = only_test_set(one_iso_pair)

    assert test_set.source() == exp_dir / "test.src.txt"
    assert test_set.vref() == exp_dir / "test.vref.txt"
    assert test_set.predictions() == exp_dir / "test.trg-predictions.txt.2"
    assert test_set.predictions_detokenized() == exp_dir / "test.trg-predictions.detok.txt.2"
    assert test_set.confidences() == exp_dir / "test.trg-predictions.txt.2.confidences.tsv"
    assert test_set.verse_scores() == exp_dir / "test.trg-predictions.detok.txt.2.scores.tsv"
    assert test_set.linregress() == exp_dir / "linregress.2.json"


def test_a_combined_test_set_takes_its_isos_from_the_corpora(one_iso_pair, exp_dir):
    (exp_dir / "test.src.txt").touch()

    test_set = only_test_set(one_iso_pair)

    assert test_set.source_iso() == "en"
    assert test_set.target_iso() == "es"


def test_a_test_set_per_iso_pair_is_named_after_the_pair(two_iso_pairs, exp_dir):
    (exp_dir / "test.en.es.src.txt").touch()
    (exp_dir / "test.en.fr.src.txt").touch()

    test_sets = two_iso_pairs.test_sets(2, {})

    assert [test_set.source().name for test_set in test_sets] == ["test.en.es.src.txt", "test.en.fr.src.txt"]
    assert [test_set.predictions().name for test_set in test_sets] == [
        "test.en.es.trg-predictions.txt.2",
        "test.en.fr.trg-predictions.txt.2",
    ]
    assert [(test_set.source_iso(), test_set.target_iso()) for test_set in test_sets] == [("en", "es"), ("en", "fr")]


def test_a_test_set_per_iso_pair_puts_the_isos_in_the_linear_regression_name(two_iso_pairs, exp_dir):
    (exp_dir / "test.en.es.src.txt").touch()

    assert only_test_set(two_iso_pairs).linregress() == exp_dir / "linregress.en.es.2.json"


def test_iso_pairs_without_a_test_set_are_skipped(two_iso_pairs, exp_dir):
    (exp_dir / "test.en.fr.src.txt").touch()

    test_sets = two_iso_pairs.test_sets(2, {})

    assert [test_set.source().name for test_set in test_sets] == ["test.en.fr.src.txt"]


def test_the_combined_test_set_wins_when_both_are_present(two_iso_pairs, exp_dir):
    (exp_dir / "test.src.txt").touch()
    (exp_dir / "test.en.es.src.txt").touch()

    assert only_test_set(two_iso_pairs).source() == exp_dir / "test.src.txt"


def test_the_averaged_checkpoint_is_named_avg_rather_than_by_step(one_iso_pair, exp_dir):
    (exp_dir / "test.src.txt").touch()

    test_set = only_test_set(one_iso_pair, step=AVERAGED_CHECKPOINT)

    assert test_set.predictions() == exp_dir / "test.trg-predictions.txt.avg"
    assert test_set.linregress() == exp_dir / "linregress.avg.json"


def test_scoring_only_some_books_names_them_before_the_step(one_iso_pair, exp_dir):
    (exp_dir / "test.src.txt").touch()

    test_set = only_test_set(one_iso_pair, books={41: [], 40: []})

    assert test_set.predictions() == exp_dir / "test.trg-predictions.txt.MAT_MRK-2"
    # The linear regression file is named by step alone, so the books do not narrow it
    assert test_set.linregress() == exp_dir / "linregress.2.json"


def test_the_scores_file_is_named_after_the_step(one_iso_pair, exp_dir):
    assert one_iso_pair.scores(2, {}, set()) == exp_dir / "scores-2.csv"
    assert one_iso_pair.scores(AVERAGED_CHECKPOINT, {}, set()) == exp_dir / "scores-avg.csv"
    assert one_iso_pair.scores(2, {40: []}, set()) == exp_dir / "scores-MAT-2.csv"


def test_the_scores_file_names_the_reference_projects_it_scored_against(one_iso_pair, exp_dir):
    assert one_iso_pair.scores(2, {}, {"RVR", "LBLA"}) == exp_dir / "scores-2-LBLA_RVR.csv"


def test_each_draft_of_a_test_set_gets_its_own_prediction_files(one_iso_pair, exp_dir):
    (exp_dir / "test.src.txt").touch()

    test_sets = one_iso_pair.test_sets(2, {}, produce_multiple_translations=True, num_drafts=2)

    assert [test_set.predictions().name for test_set in test_sets] == [
        "test.trg-predictions.1.txt.2",
        "test.trg-predictions.2.txt.2",
    ]
    assert [test_set.confidences().name for test_set in test_sets] == [
        "test.trg-predictions.1.txt.2.confidences.tsv",
        "test.trg-predictions.2.txt.2.confidences.tsv",
    ]
    assert [test_set.draft_index() for test_set in test_sets] == [1, 2]
    assert [test_set.linregress().name for test_set in test_sets] == ["linregress.2.1.json", "linregress.2.2.json"]


def test_each_test_set_reports_the_draft_it_came_from(two_iso_pairs, exp_dir):
    (exp_dir / "test.en.es.src.txt").touch()
    (exp_dir / "test.en.fr.src.txt").touch()

    test_sets = two_iso_pairs.test_sets(2, {}, produce_multiple_translations=True, num_drafts=2)

    assert [(test_set.predictions().name, test_set.draft_index()) for test_set in test_sets] == [
        ("test.en.es.trg-predictions.1.txt.2", 1),
        ("test.en.fr.trg-predictions.1.txt.2", 1),
        ("test.en.es.trg-predictions.2.txt.2", 2),
        ("test.en.fr.trg-predictions.2.txt.2", 2),
    ]


def test_no_two_test_sets_share_a_linear_regression_file(two_iso_pairs, exp_dir):
    (exp_dir / "test.en.es.src.txt").touch()
    (exp_dir / "test.en.fr.src.txt").touch()

    test_sets = two_iso_pairs.test_sets(2, {}, produce_multiple_translations=True, num_drafts=2)

    names = [test_set.linregress().name for test_set in test_sets]
    assert names == [
        "linregress.en.es.2.1.json",
        "linregress.en.fr.2.1.json",
        "linregress.en.es.2.2.json",
        "linregress.en.fr.2.2.json",
    ]
    assert len(set(names)) == len(names)


def test_every_test_set_is_scored_for_every_draft(three_iso_pairs, exp_dir):
    for iso in ("de", "es", "fr"):
        (exp_dir / f"test.en.{iso}.src.txt").touch()

    test_sets = three_iso_pairs.test_sets(2, {}, produce_multiple_translations=True, num_drafts=2)

    assert [test_set.predictions().name for test_set in test_sets] == [
        "test.en.de.trg-predictions.1.txt.2",
        "test.en.es.trg-predictions.1.txt.2",
        "test.en.fr.trg-predictions.1.txt.2",
        "test.en.de.trg-predictions.2.txt.2",
        "test.en.es.trg-predictions.2.txt.2",
        "test.en.fr.trg-predictions.2.txt.2",
    ]


def test_a_single_draft_still_covers_every_test_set(two_iso_pairs, exp_dir):
    (exp_dir / "test.en.es.src.txt").touch()
    (exp_dir / "test.en.fr.src.txt").touch()

    test_sets = two_iso_pairs.test_sets(2, {}, produce_multiple_translations=True, num_drafts=1)

    assert [test_set.predictions().name for test_set in test_sets] == [
        "test.en.es.trg-predictions.1.txt.2",
        "test.en.fr.trg-predictions.1.txt.2",
    ]


def test_a_single_reference_is_used_as_it_is(one_iso_pair, exp_dir):
    (exp_dir / "test.src.txt").touch()
    (exp_dir / "test.trg.detok.txt").touch()

    references = only_test_set(one_iso_pair).references(set())

    assert [path.name for path in references.paths] == ["test.trg.detok.txt"]
    assert not references.select_random_line


def test_unnamed_references_are_drawn_from_at_random(two_test_projects, exp_dir):
    (exp_dir / "test.src.txt").touch()
    (exp_dir / "test.trg.detok.LBLA.txt").touch()
    (exp_dir / "test.trg.detok.RVR.txt").touch()

    references = only_test_set(two_test_projects).references(set())

    assert sorted(path.name for path in references.paths) == [
        "test.trg.detok.LBLA.txt",
        "test.trg.detok.RVR.txt",
    ]
    assert references.select_random_line


def test_named_reference_projects_are_the_only_ones_scored_against(two_test_projects, exp_dir):
    (exp_dir / "test.src.txt").touch()
    (exp_dir / "test.trg.detok.LBLA.txt").touch()
    (exp_dir / "test.trg.detok.RVR.txt").touch()

    references = only_test_set(two_test_projects).references({"RVR"})

    assert [path.name for path in references.paths] == ["test.trg.detok.RVR.txt"]
    assert not references.select_random_line
