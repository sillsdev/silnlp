from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_languages import ExperimentLanguages


def languages_for(corpora, environment, lang_codes=None, pairs=None) -> ExperimentLanguages:
    if pairs is None:
        pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    return ExperimentLanguages(lang_codes if lang_codes is not None else {}, inventory)


def test_a_configured_code_is_the_name_the_model_knows(corpora, environment):
    languages = languages_for(corpora, environment, {"en": "eng_Latn"})

    assert languages.name_of("en") == "eng_Latn"


def test_an_unconfigured_language_keeps_its_iso_code(corpora, environment):
    assert languages_for(corpora, environment, {"en": "eng_Latn"}).name_of("sdl") == "sdl"


def test_the_test_languages_are_named_the_way_the_model_expects(corpora, environment):
    languages = languages_for(corpora, environment, {"en": "eng_Latn", "es": "spa_Latn"})

    assert languages.test_source() == "eng_Latn"
    assert languages.test_target() == "spa_Latn"


def test_the_validation_languages_are_named_the_same_way(corpora, environment):
    languages = languages_for(corpora, environment, {"en": "eng_Latn", "es": "spa_Latn"})

    assert languages.validation_source() == "eng_Latn"
    assert languages.validation_target() == "spa_Latn"


def test_a_language_carries_both_its_code_and_its_name(corpora, environment):
    language = languages_for(corpora, environment, {"en": "English"}).of("en")

    assert (language.iso, language.name) == ("en", "English")


def test_training_uses_the_test_languages_when_there_are_any(corpora, environment):
    languages = languages_for(corpora, environment)

    assert languages.training_source_iso() == "en"
    assert languages.training_target_iso() == "es"


def test_training_falls_back_to_any_language_of_the_corpora(corpora, environment):
    from silnlp.nmt.corpora import DataFileType

    pairs = [
        corpora.pair(
            [corpora.scripture_file("fr", "LSG")],
            [corpora.scripture_file("de", "LUT")],
            type=DataFileType.TRAIN,
        )
    ]
    languages = languages_for(corpora, environment, pairs=pairs)

    assert languages.training_source_iso() == "fr"
    assert languages.training_target_iso() == "de"


def test_an_experiment_with_no_corpora_names_no_training_languages(corpora, environment):
    languages = languages_for(corpora, environment, pairs=[])

    assert languages.training_source_iso() == ""
    assert languages.training_target_iso() == ""
