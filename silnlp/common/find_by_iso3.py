import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from .environment import SIL_NLP_ENV
from .iso_info import NLLB_ISO_SET, ALT_ISO

IsoCode = str
IsoCodeList = List[IsoCode]
IsoCodeSet = Set[IsoCode]

LANGUAGE_FAMILY_FILE = SIL_NLP_ENV.assets_dir / "languageFamilies.json"

def load_language_data(file_path: Path) -> Tuple[Dict, Dict, Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}, {}, {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        return {}, {}, {}

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


def find_related_isocodes(
    iso_codes: IsoCodeList, language_data: Dict, country_data: Dict, family_data: Dict
) -> IsoCodeList:
    iso_set = set(iso_codes)

    for iso_code in iso_codes:
        if iso_code in language_data:
            lang_info = language_data[iso_code]
#            logging.info(f"{iso_code}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}")

            iso_set.update(country_data.get(lang_info["Country"], []))
            iso_set.update(family_data.get(lang_info["Family"], []))

    return sorted(iso_set)


def get_files_by_iso(isocodes: IsoCodeList, scripture_dir: Path) -> List[Path]:
    return [
        file for file in scripture_dir.glob('*.txt')
        if any(file.stem.startswith(isocode + '-') for isocode in isocodes)
    ]

def split_files_by_projects(files: List[Path], projects_dir: Path) -> Tuple[Dict[Path, Path], Dict[Path, Path]]:
    existing_projects = {}
    missing_projects = {}

    for file in files:
        project = projects_dir / file.stem.split("-")[1]
        if project.is_dir():
            existing_projects[file] = project
        else:
            missing_projects[file] = project

    return existing_projects, missing_projects


def get_equivalent_isocodes(iso_codes: List[str]) -> Set[str]:
    return {code for iso_code in iso_codes for code in (iso_code, ALT_ISO.get_alternative(iso_code)) if code}

def main():
    parser = argparse.ArgumentParser(description="Find related ISO language codes.")
    parser.add_argument("iso_codes", nargs="+", help="ISO codes to find related languages for")
    parser.add_argument("--scripture-dir", type=Path, default=Path(SIL_NLP_ENV.mt_scripture_dir), help="Directory containing scripture files")
    parser.add_argument("--all-related", action='store_true', help="List all related scriptures without filtering to those that are part of NLLB")
    parser.add_argument("--no-related", action='store_true', help="Only list scriptures in the specified languages and not in related languages")
    parser.add_argument("--country-related", action='store_true', help="Only list scriptures from the same country.")
    
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args = parser.parse_args()

    language_data, country_data, family_data = load_language_data(LANGUAGE_FAMILY_FILE)
    projects_dir = SIL_NLP_ENV.pt_projects_dir
    scripture_dir = Path(args.scripture_dir)
    country_only = args.country_related
    
    if not language_data:
        logging.error("Failed to load language data.")
        return
    
    # Get equivalent ISO codes for input
    iso_codes = get_equivalent_isocodes(args.iso_codes)

    if args.no_related:
        
        # Option 2: No files in related languages, only equivalent ISO codes
        related_isocodes = list(iso_codes)
        logging.info(f"\nConsidering only the {len(related_isocodes)} specified.")
        
    else:
        
        # Find related ISO codes
        related_isocodes = find_related_isocodes(list(iso_codes), language_data, country_data, family_data)
        logging.info(f"\nFound {len(related_isocodes)} related languages.")
        if not args.all_related:
            
            # Option 3 (default): Filter to NLLB languages
            related_isocodes = [iso for iso in related_isocodes if iso in NLLB_ISO_SET]
            logging.info(f"\nFound {len(related_isocodes)} related or specified languages in NLLB.")
        # Option 1: All related files (no filtering) is handled by not applying the NLLB filter
        else:
            logging.info(f"\nFound {len(related_isocodes)} related or specified languages.")

    # Get all possible 2 and 3 letter codes for the related languages
    all_possible_codes = get_equivalent_isocodes(related_isocodes)
    
    # Find files matching the codes
    files = get_files_by_iso(all_possible_codes, scripture_dir)
    existing_projects, missing_projects = split_files_by_projects(files, projects_dir)

    # Display results
    if existing_projects:
        print(f"\nThese {len(existing_projects)} files have a corresponding project folder:")
        for file, project in existing_projects.items():
            print(file.stem, project)
    
    if missing_projects:
        print(f"\nThese {len(missing_projects)} files don't have a corresponding project folder:")
        for file, _ in missing_projects.items():
            print(f"{file.stem}")
    print(f"All the files:")
    for file in files:
        print(f"    - {file.stem}")

    if not files:
        logging.info("\nCouldn't find any Scripture files in these languages.")

if __name__ == "__main__":
    main()