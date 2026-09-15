import pandas as pd

from silnlp.nmt.eval_data_set import EvalDataSet


def corpus(sources, targets, index=None, vrefs=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vref": vrefs if vrefs is not None else [f"GEN 1:{i + 1}" for i in range(len(sources))],
            "source": sources,
            "target": targets,
        },
        index=index if index is not None else range(len(sources)),
    )


def test_an_empty_corpus_is_ignored():
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", [], corpus([], []))
    assert data_set.is_empty()


def test_the_target_column_is_named_after_its_project():
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))

    (_, frame), = data_set.language_pairs()
    assert "target_LBLA" in frame.columns
    assert "target" not in frame.columns


def test_tags_are_prefixed_to_the_source():
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", ["formal"], corpus(["one"], ["uno"]))

    (_, frame), = data_set.language_pairs()
    assert list(frame["source"]) == ["<formal> one"]


def test_several_projects_become_columns_of_one_frame():
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))
    data_set.add("en", "es", "RVR", [], corpus(["one"], ["uno rvr"]))

    (_, frame), = data_set.language_pairs()
    assert list(frame["target_LBLA"]) == ["uno"]
    assert list(frame["target_RVR"]) == ["uno rvr"]


def test_a_verse_missing_from_one_project_is_left_blank_rather_than_absent():
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", [], corpus(["one"], ["uno"], index=[0]))
    data_set.add("en", "es", "RVR", [], corpus(["two"], ["dos"], index=[1]))

    (_, frame), = data_set.language_pairs()
    assert len(frame) == 2
    assert sorted(frame["target_LBLA"]) == ["", "uno"]
    assert sorted(frame["target_RVR"]) == ["", "dos"]


def test_language_pairs_are_kept_apart():
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))
    data_set.add("fr", "de", "LUT", [], corpus(["un"], ["eins"]))

    assert sorted(key for key, _ in data_set.language_pairs()) == [("en", "es"), ("fr", "de")]


def test_the_total_size_counts_every_language_pair():
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", [], corpus(["one", "two"], ["uno", "dos"]))
    data_set.add("fr", "de", "LUT", [], corpus(["un"], ["eins"]))

    assert data_set.total_size() == 3


def test_the_verses_of_the_first_project_fix_the_selection_for_the_rest():
    # Later projects have to cover the same verses, so the first one to arrive decides them.
    data_set = EvalDataSet()
    data_set.add("en", "es", "LBLA", [], corpus(["one"], ["uno"], index=[3]))
    data_set.add("en", "es", "RVR", [], corpus(["two"], ["dos"], index=[7]))

    assert data_set.indices_for("en", "es", default=None) == {3}


def test_a_language_pair_with_no_data_falls_back_to_the_given_selection():
    data_set = EvalDataSet()
    assert data_set.indices_for("en", "es", default={1, 2}) == {1, 2}


def test_a_preselected_set_of_verses_is_used_instead_of_the_first_project():
    data_set = EvalDataSet({("en", "es"): {5}})
    data_set.add("en", "es", "LBLA", [], corpus(["one"], ["uno"], index=[3]))

    assert data_set.indices_for("en", "es", default=None) == {5}
