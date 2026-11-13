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


def test_a_direction_qualifier_moves_to_the_front():
    assert naming().language_segment("Abenaki, Eastern") == "Eastern_Abenaki"


def test_a_compound_direction_moves_to_the_front():
    assert naming().language_segment("Kolami, Northwestern") == "Northwestern_Kolami"


def test_a_two_word_direction_moves_to_the_front():
    assert naming().language_segment("Mixe, West Central") == "West_Central_Mixe"


def test_upper_counts_as_a_direction():
    assert naming().language_segment("Chinantec, Upper") == "Upper_Chinantec"


def test_a_qualifier_that_is_not_a_direction_stays_after_the_name():
    assert naming().language_segment("Adi, Galo") == "Adi_Galo"


def test_a_qualifier_that_starts_with_a_direction_moves_to_the_front():
    assert naming().language_segment("Mixtec, Southern Puebla") == "Southern_Puebla_Mixtec"


def test_a_qualifier_that_starts_with_a_compound_direction_moves_to_the_front():
    assert naming().language_segment("Miao, Southwestern Guiyang") == "Southwestern_Guiyang_Miao"


def test_a_direction_after_another_word_moves_the_whole_qualifier_to_the_front():
    assert naming().language_segment("Karen, Pwo Eastern") == "Pwo_Eastern_Karen"


def test_a_direction_preceded_by_a_modifier_moves_the_whole_qualifier_to_the_front():
    assert naming().language_segment("Muria, Far Western") == "Far_Western_Muria"


def test_a_direction_in_the_middle_of_the_qualifier_moves_it_to_the_front():
    assert naming().language_segment("Quechua, Cajatambo North Lima") == "Cajatambo_North_Lima_Quechua"


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
