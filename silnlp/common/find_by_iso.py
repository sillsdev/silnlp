import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from .environment import SIL_NLP_ENV
from .iso_info import NLLB_TAG_FROM_ISO, NLLB_ISO_SET, ALT_ISO

IsoCode = str
IsoCodeSet = Set[IsoCode]


def load_language_data(file_path: Path) -> Tuple[Dict, Dict, Dict]:
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


def find_related_isocodes(iso_codes, language_data, country_data, family_data) -> IsoCodeSet:
    iso_set = set(iso_codes)

    for iso_code in iso_codes:
        if iso_code in language_data:

            lang_info = language_data[iso_code]
            print(f"{iso_code}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}")

            # Add iso codes from the same country
            iso_set.update(country_data.get(lang_info["Country"], []))

            # Add iso codes from the same family
            iso_set.update(family_data.get(lang_info["Family"], []))

    return iso_set


def main():
    parser = argparse.ArgumentParser(description="Find data in NLLB languages given a list of ISO codes.")
    parser.add_argument(
        "--directory",
        type=str,
        default=f"{SIL_NLP_ENV.mt_scripture_dir}",
        help=f"Directory to search. The default is {SIL_NLP_ENV.mt_scripture_dir}",
    )
    parser.add_argument("iso_codes", type=str, nargs="+", help="List of ISO codes to search for")
    # parser.add_argument("--no_related", action='store_true', help="Only list specified languages and not related iso codes that are part of NLLB")

    args = parser.parse_args()
    iso_codes = args.iso_codes
    iso_codes = args.iso_codes
    equivalent_iso_codes = [code for iso_code in iso_codes for code in (iso_code, ALT_ISO.get_alternative(iso_code)) if code]
    
    projects_folder = SIL_NLP_ENV.pt_projects_dir
    scripture_dir = Path(args.directory)
    file_path = SIL_NLP_ENV.assets_dir / "languageFamilies.json"
    language_data, country_data, family_data = load_language_data(file_path)

    print("\nThe iso codes given represent these languages:")
    related_isos = find_related_isocodes(iso_codes, language_data, country_data, family_data)
    if related_isos:
        print(f"\nThere are {len(related_isos)} languages from the same language family or country.")

    if related_isos:
        # Remove iso codes not in NLLB
        related_isos_in_nllb = sorted(related_isos.intersection(NLLB_ISO_SET))
        if related_isos_in_nllb:

            # Look for scriptures in these languages too.
            iso_codes.extend(related_isos_in_nllb)

            print(f"\nOf these, {len(related_isos_in_nllb)} are in NLLB.")
            if len(related_isos_in_nllb) < 21:
                for related_iso_in_nllb in related_isos_in_nllb:
                    lang_info = language_data[related_iso_in_nllb]
                    print(f"{related_iso_in_nllb}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}")
            else:
                print(f"{' '.join(related_isos_in_nllb)}")

            matching_files = []
            for filepath in scripture_dir.iterdir():
                if filepath.suffix == ".txt":
                    iso_code = filepath.stem.split("-")[0]
                    if iso_code in equivalent_iso_codes:
                        matching_files.append(filepath.stem)  # Remove .txt extension

            if matching_files:
                projects = []
                missing_projects = []

                print("\nThe following Scripture files were found in these languages.")
                for file in matching_files:
                    print(f"    - {file}")
                    if '-' in file:
                        iso, project = file.split("-", maxsplit=1)
                        project_dir = projects_folder / project
                        if project_dir.is_dir():
                            projects.append(project)
                        else:
                            missing_projects.append(project)
                    else:
                        print(f"Couldn't split {file} on '-' to find iso code and project name.")
            else:
                print("\nCouldn't find any Scripture files in these languages.")

    else:
        print(f"\nDidn't find any language that is related or spoken in the same country in NLLB.")



if __name__ == "__main__":
    main()