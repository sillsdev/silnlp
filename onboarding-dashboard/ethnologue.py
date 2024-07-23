GDRIVE_DATA_FOLDER_ID = "1Cetjb4qRlA1bwY6iJujrlwDSgTWUxMMx"
DATA_FOLDER_PATH = ".eth_data"

import os
from pathlib import Path

import numpy as np
import pandas as pd

from clowder import functions


class EthnologPermissionsError(Exception):
    """Missing permissions for Ethnologue folder."""


class MissingFileException(Exception):
    """Could not find requested file"""


with functions._lock:
    if not Path.exists(Path(DATA_FOLDER_PATH)):
        os.mkdir(DATA_FOLDER_PATH)
        env = functions.get_env()
        env._copy_gdrive_folder_to_storage(GDRIVE_DATA_FOLDER_ID, Path(DATA_FOLDER_PATH))

_files = {x.name: x for x in Path(DATA_FOLDER_PATH).iterdir() if x.is_file()}

_df_cache = {xl_name: None for xl_name in _files.keys()}


def _get_cached_xl_df(xl_name, cleanup_action=lambda _: _) -> pd.DataFrame:
    try:
        if _df_cache[xl_name] is None:
            language_path = _files[xl_name]
            _df_cache[xl_name] = cleanup_action(pd.read_excel(language_path, 0))
        return _df_cache[xl_name]
    except KeyError as exc:
        if len(_files.keys()) == 0:
            raise EthnologPermissionsError from None
        raise MissingFileException from exc


def language_df():
    """Get Ethnologue Language data dataframe"""
    return _get_cached_xl_df("Language.xlsx")


def language_in_country_df():
    """Get Ethnologue LanguageInCountry data dataframe"""

    def cleanup(l_c_df):
        l_c_df["UnitCode"] = l_c_df["UnitCode"].str.strip()
        l_c_df["RegionCode"] = l_c_df["RegionCode"].str.strip()
        return l_c_df

    return _get_cached_xl_df("LanguageInCountry.xlsx", cleanup_action=cleanup)


def language_in_country_add_data_df():
    """Get Ethnologue LanguageInCountryAdditionalData data dataframe"""
    return _get_cached_xl_df("LanguageInCountryAddionalData.xlsx")


def language_add_data_df():
    """Get Ethnologue LanguageEthnologAdditionalData data dataframe"""
    return _get_cached_xl_df("LanguageEthnologAdditionalData.xlsx")


def language_alt_name_df():
    """Get Ethnologue LanguageAlternateName data dataframe"""
    return _get_cached_xl_df("LanguageAlternateName.xlsx")


def lang_info_dict(lang_code: str):
    l_df = language_df()
    row = l_df[l_df["UnitCode"] == lang_code]
    return row.to_dict("records")[0]


def lang_classifications(lang_code: str):
    d = lang_info_dict(lang_code)
    return [_.strip() for _ in d["Classification"].split(",")] if isinstance(d["Classification"], str) else []


def find_class_langs(classfication: str):
    l_df = language_df()
    return list(set(l_df[l_df["Classification"].astype(str).str.contains(classfication)]["UnitCode"].to_list()))


def lang_country_dicts(lang_code: str):
    l_c_df = language_in_country_df()
    row = l_c_df.loc[l_c_df["UnitCode"] == lang_code]
    return row.to_dict("records")


def lang_country_family(lang_code: str):
    return lang_country_dicts(lang_code)[0]["Family"]


def find_country_family_langs(family: str):
    l_c_df = language_in_country_df()
    return list(set(l_c_df[l_c_df["Family"] == family]["UnitCode"].to_list()))


def find_country_country_langs(country_code: str):
    l_c_df = language_in_country_df()
    return list(set(l_c_df[l_c_df["CountryCode"] == country_code]["UnitCode"].to_list()))


def find_country_region_langs(region_code: str):
    l_c_df = language_in_country_df()
    return l_c_df[l_c_df["RegionCode"] == region_code]["UnitCode"].to_list()


def lang_ad_dict(lang_code: str):
    l_ad_df = language_add_data_df()
    row = l_ad_df[l_ad_df["LanguageCode"] == lang_code]
    return row.to_dict("records")[0]


def lang_ad_family(lang_code: str):
    return lang_ad_dict(lang_code)["LanguageFamily"]


def find_ad_family_langs(family: str):
    l_ad_df = language_add_data_df()
    return list(set(l_ad_df[l_ad_df["LanguageFamily"] == family]["UnitCode"].to_list()))
