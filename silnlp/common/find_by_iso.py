import argparse
import json
import os
from pathlib import Path

from ..common.environment import SIL_NLP_ENV

NLLB_ISOS = [
        "ace",
        "acm",
        "acq",
        "aeb",
        "afr",
        "ajp",
        "aka",
        "als",
        "amh",
        "apc",
        "arb",
        "ars",
        "ary",
        "arz",
        "asm",
        "ast",
        "awa",
        "ayr",
        "azb",
        "azj",
        "bak",
        "bam",
        "ban",
        "bel",
        "bem",
        "ben",
        "bho",
        "bjn",
        "bod",
        "bos",
        "bug",
        "bul",
        "cat",
        "ceb",
        "ces",
        "cjk",
        "ckb",
        "crh",
        "cym",
        "dan",
        "deu",
        "dik",
        "dyu",
        "dzo",
        "ell",
        "eng",
        "epo",
        "est",
        "eus",
        "ewe",
        "fao",
        "fij",
        "fin",
        "fon",
        "fra",
        "fur",
        "fuv",
        "gaz",
        "gla",
        "gle",
        "glg",
        "grn",
        "guj",
        "hat",
        "hau",
        "heb",
        "hin",
        "hne",
        "hrv",
        "hun",
        "hye",
        "ibo",
        "ilo",
        "ind",
        "isl",
        "ita",
        "jav",
        "jpn",
        "kab",
        "kac",
        "kam",
        "kan",
        "kas",
        "kat",
        "kaz",
        "kbp",
        "kea",
        "khk",
        "khm",
        "kik",
        "kin",
        "kir",
        "kmb",
        "kmr",
        "knc",
        "kon",
        "kor",
        "lao",
        "lij",
        "lim",
        "lin",
        "lit",
        "lmo",
        "ltg",
        "ltz",
        "lua",
        "lug",
        "luo",
        "lus",
        "lvs",
        "mag",
        "mai",
        "mal",
        "mar",
        "min",
        "mkd",
        "mlt",
        "mni",
        "mos",
        "mri",
        "mya",
        "nld",
        "nno",
        "nob",
        "npi",
        "nso",
        "nus",
        "nya",
        "oci",
        "ory",
        "pag",
        "pan",
        "pap",
        "pbt",
        "pes",
        "plt",
        "pol",
        "por",
        "prs",
        "quy",
        "ron",
        "run",
        "rus",
        "sag",
        "san",
        "sat",
        "scn",
        "shn",
        "sin",
        "slk",
        "slv",
        "smo",
        "sna",
        "snd",
        "som",
        "sot",
        "spa",
        "srd",
        "srp",
        "ssw",
        "sun",
        "swe",
        "swh",
        "szl",
        "tam",
        "taq",
        "tat",
        "tel",
        "tgk",
        "tgl",
        "tha",
        "tir",
        "tpi",
        "tsn",
        "tso",
        "tuk",
        "tum",
        "tur",
        "twi",
        "tzm",
        "uig",
        "ukr",
        "umb",
        "urd",
        "uzn",
        "vec",
        "vie",
        "war",
        "wol",
        "xho",
        "ydd",
        "yor",
        "yue",
        "zho",
        "zsm",
        "zul",
    ]

def load_language_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    # Restructure the data for faster lookups
    language_data = {}
    country_data = {}
    family_data = {}

    for lang in raw_data:
        iso = lang["isoCode"]
        country = lang["langCountry"]
        family = lang["languageFamily"]

        language_data[iso] = {
            "Name": lang["language"],
            "Country": country,
            "Family": family,
        }

        country_data.setdefault(country, []).append(iso)
        family_data.setdefault(family, []).append(iso)

    return language_data, country_data, family_data
   

def process_iso_codes(iso_codes, language_data, country_data, family_data, nllb_isos):
    iso_set = set(iso_codes)

    for iso in iso_codes:
        if iso in language_data:

            lang_info = language_data[iso]
            print(
                f"{iso}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}"
            )

            # Add iso codes from the same country
            iso_set.update(country_data.get(lang_info["Country"], []))

            # Add iso codes from the same family
            iso_set.update(family_data.get(lang_info["Family"], []))

    return iso_set


def main():
    parser = argparse.ArgumentParser(description="Find data in NLLB languages given a list of ISO codes.")
    parser.add_argument("--directory", type=str, default=f"{SIL_NLP_ENV.mt_scripture_dir}", help=f"Directory to search. The default is {SIL_NLP_ENV.mt_scripture_dir}")
    parser.add_argument("iso_codes", type=str, nargs="+", help="List of ISO codes to search for")
    #parser.add_argument("--no_related", action='store_true', help="Only list specified languages and not related iso codes that are part of NLLB")

    args = parser.parse_args()
    iso_codes = args.iso_codes
    projects_folder = SIL_NLP_ENV.pt_projects_dir
    scripture_dir = Path(args.directory)

    print("Finding related languages and those spoken in the same country.")
    file_path = SIL_NLP_ENV.assets_dir / "languageFamilies.json"

    language_data, country_data, family_data = load_language_data(file_path)
    nllb_set = set(NLLB_ISOS)

    related_isos = process_iso_codes(iso_codes, language_data, country_data, family_data, NLLB_ISOS)

    if related_isos:
        # Remove iso codes not in NLLB    
        related_isos_in_nllb = sorted(related_isos.intersection(nllb_set))
        if related_isos_in_nllb:

            # Look for scriptures in these languages too.
            iso_codes.extend(related_isos_in_nllb)
            
            print(f"Found {len(related_isos_in_nllb)} languages that from the same country or language family in NLLB.")
            for related_iso_in_nllb in related_isos_in_nllb:
                lang_info = language_data[related_iso_in_nllb]
                print(
                    f"{related_iso_in_nllb}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}"
                )
    else:
        print(f"Didn't find any language that is related or spoken in the same country in NLLB.")
    

    matching_files = []
    for filepath in scripture_dir.iterdir():
        if filepath.suffix == ".txt":
            iso_code = filepath.stem.split("-")[0]
            if iso_code in args.iso_codes:
                matching_files.append(filepath.stem)  # Remove .txt extension

    if matching_files:
        print("Matching files:")
        for file in matching_files:
            print(f"      - {file}")

        for file in matching_files:
            parts = file.split("-", maxsplit=1)
            if len(parts) > 1:
                iso = parts[0]
                project = parts[1]
                project_dir = projects_folder / project
                print(f"{project} exists: {project_dir.is_dir()}")
            else:
                print(f"Couldn't split {file} on '-'")
    else:
        print("No matching files found.")


if __name__ == "__main__":
    main()


    # projects_folder = Path("S:\Paratext\projects")
    # scripture_dir = args.directory
    # matching_files = []
    # for filename in os.listdir(scripture_dir):
    #     if filename.endswith(".txt"):
    #         iso_code = filename.split("-")[0]
    #         if iso_code in args.iso_codes:
    #             matching_files.append(os.path.splitext(filename)[0])  # Remove .txt extension

    # if matching_files:
    #     print("Matching files:")
    #     for file in matching_files:
    #         print(f"      - {file}")

    #     for file in matching_files:
    #         parts = file.split("-", maxsplit=1)
    #         if len(parts) > 1:
    #             iso = parts[0]
    #             project = parts[1]
    #             project_dir = projects_folder / project
    #             print(f"{project} exists: {project_dir.is_dir()}")
    #         else:
    #             print(f"Couldn't split {file} on '-'")
    # else:
    #     print("No matching files found.")

