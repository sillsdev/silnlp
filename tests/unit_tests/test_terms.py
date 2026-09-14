import logging

import pytest

from silnlp.nmt.terms import GlossLanguage, TermCategories


def test_no_configured_categories_selects_every_term():
    categories = TermCategories(None)
    assert not categories.excludes_everything()
    assert categories.includes("PN")
    assert categories.includes("anything")
    assert categories.as_set() is None


def test_a_single_configured_category():
    categories = TermCategories("PN")
    assert categories.as_set() == {"PN"}
    assert categories.includes("PN")
    assert not categories.includes("FL")


def test_a_comma_separated_list_is_split_and_stripped():
    assert TermCategories("PN, FL ,RE").as_set() == {"PN", "FL", "RE"}


def test_categories_may_be_given_as_a_list():
    assert TermCategories(["PN", "FL"]).as_set() == {"PN", "FL"}


def test_an_empty_list_excludes_every_term():
    assert TermCategories([]).excludes_everything()


def test_an_empty_string_is_one_unnamed_category_rather_than_none():
    # "".split(",") yields a single empty name, which is not the same as configuring no categories.
    categories = TermCategories("")
    assert not categories.excludes_everything()
    assert categories.as_set() == {""}


def gloss(include_glosses, source_isos=(), target_isos=()) -> GlossLanguage:
    return GlossLanguage(include_glosses, set(source_isos), set(target_isos))


def test_glosses_turned_off_have_no_language():
    language = gloss(False, source_isos=["en"], target_isos=["es"])
    assert not language.is_available()
    assert language.iso() is None


def test_a_named_gloss_language_is_used_as_given():
    assert gloss("fr", source_isos=["en"], target_isos=["es"]).iso() == "fr"


def test_a_named_gloss_language_is_lowercased():
    assert gloss("FR", source_isos=["en"]).iso() == "fr"


def test_an_unsupported_gloss_language_is_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        language = gloss("sw", source_isos=["en"], target_isos=["es"])
    assert not language.is_available()
    assert "sw" in caplog.text


def test_true_picks_a_supported_source_language(caplog):
    assert gloss(True, source_isos=["en"], target_isos=["sw"]).iso() == "en"


def test_true_falls_back_to_a_supported_target_language():
    assert gloss(True, source_isos=["sw"], target_isos=["es"]).iso() == "es"


def test_true_prefers_the_source_side():
    assert gloss(True, source_isos=["en"], target_isos=["es"]).iso() == "en"


def test_true_with_no_supported_language_warns_and_gives_up(caplog):
    with caplog.at_level(logging.WARNING):
        language = gloss(True, source_isos=["sw"], target_isos=["zu"])
    assert not language.is_available()
    assert "supported gloss language codes" in caplog.text


def test_the_gloss_language_can_serve_as_the_target_when_it_is_a_target_language():
    assert gloss(True, source_isos=["sw"], target_isos=["es"]).can_serve_as_target()
    assert not gloss("fr", source_isos=["fr"], target_isos=["es"]).can_serve_as_target()


def test_the_gloss_language_can_serve_as_the_source_when_it_is_a_source_language():
    assert gloss(True, source_isos=["en"], target_isos=["sw"]).can_serve_as_source()


def test_a_named_gloss_language_can_serve_as_the_source_even_when_absent_from_the_experiment():
    assert gloss("fr", source_isos=["en"], target_isos=["es"]).can_serve_as_source()


def test_an_unavailable_gloss_language_can_serve_as_neither_side():
    language = gloss(False, source_isos=["en"], target_isos=["es"])
    assert not language.can_serve_as_source()
    assert not language.can_serve_as_target()


@pytest.mark.parametrize("iso", ["fr", "en", "id", "es", "pt"])
def test_every_supported_gloss_language_is_accepted(iso):
    assert gloss(iso, source_isos=["de"]).iso() == iso
