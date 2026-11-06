import logging

from silnlp.common.postprocessor import TrainingTargetProjects


def scripture_pair(corpora, src_project: str, trg_project: str, **settings):
    return corpora.pair(
        [corpora.scripture_file("en", src_project)], [corpora.scripture_file("es", trg_project)], **settings
    )


def basic_pair(corpora, name: str):
    return corpora.pair([corpora.basic_file("en", name)], [corpora.basic_file("es", name)])


def test_the_target_project_of_every_scripture_pair_is_a_candidate(corpora):
    pairs = [scripture_pair(corpora, "BSB", "LBLA"), scripture_pair(corpora, "BSB", "RVR")]

    candidates = TrainingTargetProjects(pairs).candidates()

    assert [project for project, _ in candidates] == ["LBLA", "RVR"]


def test_a_pairs_corpus_books_travel_with_its_project(corpora):
    pairs = [scripture_pair(corpora, "BSB", "LBLA", corpus_books={1: [1, 2]})]

    assert TrainingTargetProjects(pairs).candidates() == [("LBLA", {1: [1, 2]})]


def test_basic_pairs_are_never_candidates(corpora):
    pairs = [basic_pair(corpora, "extra"), scripture_pair(corpora, "BSB", "LBLA")]

    assert [project for project, _ in TrainingTargetProjects(pairs).candidates()] == ["LBLA"]


def test_with_no_scripture_pair_there_is_nothing_to_detect_from(corpora):
    assert TrainingTargetProjects([basic_pair(corpora, "extra")]).candidates() == [(None, None)]


def test_with_no_pairs_at_all_there_is_nothing_to_detect_from():
    assert TrainingTargetProjects([]).candidates() == [(None, None)]


def test_several_scripture_pairs_are_warned_about(corpora, caplog):
    pairs = [scripture_pair(corpora, "BSB", "LBLA"), scripture_pair(corpora, "BSB", "RVR")]

    with caplog.at_level(logging.WARNING):
        TrainingTargetProjects(pairs).candidates()

    assert any("multiple corpus pairs" in record.message for record in caplog.records)


def test_a_basic_pair_does_not_count_towards_the_warning(corpora, caplog):
    pairs = [basic_pair(corpora, "extra"), scripture_pair(corpora, "BSB", "LBLA")]

    with caplog.at_level(logging.WARNING):
        TrainingTargetProjects(pairs).candidates()

    assert not any("multiple corpus pairs" in record.message for record in caplog.records)


def test_several_source_projects_in_a_pair_are_warned_about(corpora, caplog):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB"), corpora.scripture_file("en", "KJV")],
        [corpora.scripture_file("es", "LBLA"), corpora.scripture_file("es", "RVR")],
    )

    with caplog.at_level(logging.WARNING):
        TrainingTargetProjects([pair]).candidates()

    messages = [record.message for record in caplog.records]
    assert any("multiple source projects" in message for message in messages)
    assert any("multiple target projects" in message for message in messages)
