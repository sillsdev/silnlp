from silnlp.common.folder_naming import FolderNaming


def naming(countries=None, languages=None) -> FolderNaming:
    return FolderNaming({"countries": countries or {}, "languages": languages or {}})


def test_spaces_become_underscores():
    assert naming().country_segment("Burkina Faso") == "Burkina_Faso"


def test_hyphens_become_underscores():
    assert naming().country_segment("Guinea-Bissau") == "Guinea_Bissau"


def test_capitalisation_is_kept():
    assert naming().country_segment("Isle of Man") == "Isle_of_Man"


def test_a_leading_click_letter_keeps_its_case():
    assert naming().language_segment("ǁGana") == "ǁGana"


def test_accents_and_apostrophes_are_kept():
    assert naming().country_segment("Côte d'Ivoire") == "Côte_d'Ivoire"


def test_bracketed_text_is_dropped():
    assert naming().country_segment("Cocos (Keeling) Islands") == "Cocos_Islands"


def test_repeated_underscores_are_collapsed():
    assert naming().language_segment("Abenaki _ Eastern") == "Abenaki_Eastern"


def test_the_qualifier_moves_to_the_front():
    assert naming().language_segment("Adi, Galo") == "Galo_Adi"


def test_a_direction_qualifier_moves_to_the_front():
    assert naming().language_segment("Abenaki, Eastern") == "Eastern_Abenaki"


def test_a_multi_word_qualifier_moves_to_the_front():
    assert naming().language_segment("Huave, San Mateo del Mar") == "San_Mateo_del_Mar_Huave"


def test_a_direction_inside_the_qualifier_needs_no_special_handling():
    assert naming().language_segment("Karen, Pwo Eastern") == "Pwo_Eastern_Karen"


def test_a_hyphenated_qualifier_has_its_hyphens_joined():
    assert naming().language_segment("Arabic, Judeo-Iraqi") == "Judeo_Iraqi_Arabic"


def test_a_name_without_a_comma_is_left_in_order():
    assert naming().language_segment("Saudi Arabian Sign Language") == "Saudi_Arabian_Sign_Language"


def test_standard_is_dropped_leaving_the_name():
    assert naming().language_segment("German, Standard") == "German"


def test_standard_is_dropped_from_a_longer_qualifier():
    assert naming().language_segment("Tamazight, Standard Moroccan") == "Moroccan_Tamazight"


def test_spoken_is_dropped_and_the_qualifier_moves_to_the_front():
    assert naming().language_segment("Arabic, Chadian Spoken") == "Chadian_Arabic"


def test_a_multi_word_spoken_qualifier_moves_to_the_front():
    assert naming().language_segment("Arabic, Eastern Egyptian Bedawi Spoken") == "Eastern_Egyptian_Bedawi_Arabic"


def test_an_override_is_used_verbatim():
    assert naming(countries={"Russian Federation": "Russia"}).country_segment("Russian Federation") == "Russia"


def test_an_override_keeps_its_capitals():
    assert naming(countries={"Papua New Guinea": "PNG"}).country_segment("Papua New Guinea") == "PNG"


def test_an_override_is_not_reordered():
    assert naming(languages={"Arabic, Standard": "Standard Arabic"}).language_segment("Arabic, Standard") == (
        "Standard_Arabic"
    )


def test_an_override_has_its_spaces_joined():
    assert naming(countries={"Virgin Islands, U.S.": "US Virgin Islands"}).country_segment("Virgin Islands, U.S.") == (
        "US_Virgin_Islands"
    )


def test_a_blank_override_is_ignored():
    assert naming(countries={"Peru": "  "}).country_segment("Peru") == "Peru"


def test_a_language_override_does_not_apply_to_countries():
    assert naming(languages={"Peru": "Wrong"}).country_segment("Peru") == "Peru"


def test_previous_segment_capitalises_each_word_as_before():
    assert naming().previous_segment("Abenaki, Eastern") == "Abenaki_Eastern"


def test_previous_segment_lowercases_the_rest_of_each_word():
    assert naming().previous_segment("Côte d'Ivoire") == "Côte_D'ivoire"
