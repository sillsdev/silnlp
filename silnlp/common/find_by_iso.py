import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import regex as re

from .environment import SIL_NLP_ENV
from .iso_info import ALT_ISO, NLLB_ISO_SET

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
            iso_set.update(country_data.get(lang_info["Country"], []))
            iso_set.update(family_data.get(lang_info["Family"], []))

    return sorted(iso_set)


def get_files_by_iso(isocodes: IsoCodeList, scripture_dir: Path) -> List[Path]:
    return [
        file for file in scripture_dir.glob("*.txt") if any(file.stem.startswith(isocode + "-") for isocode in isocodes)
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


def filter_files(files: List[Path], excluded_patterns: List[str]) -> List[Path]:
    filtered = []

    today = datetime.now()
    today_pattern = re.compile(f"{today.strftime('_%Y_%m_%d')}|{today.strftime('_%d_%m_%Y')}")
    date_pattern = re.compile(r"_\d{4}_\d{1,2}_\d{1,2}|_\d{1,2}_\d{1,2}_\d{4}")

    for file in files:
        parts = file.stem.split("-", 1)
        if len(parts) != 2:
            continue
        iso, name = parts
        if today_pattern.search(name):
            filtered.append(file)
            continue
        if date_pattern.search(name):
            continue
        if len(iso) not in (2, 3):
            continue
        if any(pattern.lower() in name.lower() for pattern in excluded_patterns):
            continue
        if file.is_file() and file.stat().st_size < 100_000:
            continue
        filtered.append(file)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Find related ISO language codes.")
    parser.add_argument("iso_codes", nargs="+", help="ISO codes to find related languages for")
    parser.add_argument(
        "--scripture-dir",
        type=Path,
        default=Path(SIL_NLP_ENV.mt_scripture_dir),
        help="Directory containing scripture files",
    )
    parser.add_argument(
        "--all-related",
        action="store_true",
        help="List all related scriptures without filtering to those that are part of NLLB",
    )
    parser.add_argument(
        "--no-related",
        action="store_true",
        help="Only list scriptures in the specified languages and not in related languages",
    )
    parser.add_argument("--output", type=Path, help="Output to the specified file.")

    args = parser.parse_args()

    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the global logging level
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")

    if args.output:
        # Create handler for the file output.
        file_handler = logging.FileHandler(args.output)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Create handler for the console output.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    language_data, country_data, family_data = load_language_data(LANGUAGE_FAMILY_FILE)
    projects_dir = SIL_NLP_ENV.pt_projects_dir
    scripture_dir = Path(args.scripture_dir)

    if not language_data:
        logging.error("Failed to load language data.")
        return

    # Get equivalent ISO codes for input
    iso_codes = get_equivalent_isocodes(args.iso_codes)

    if args.no_related:

        # Option 2: No files in related languages, only equivalent ISO codes
        codes_to_find = list(iso_codes)
        logger.info(f"\nConsidering only the specified iso codes and their equivalents. {codes_to_find}")

    else:
        # Find related ISO codes
        codes_to_find = find_related_isocodes(list(iso_codes), language_data, country_data, family_data)
        logger.info(f"\nFound {len(codes_to_find)} related languages:\n{codes_to_find}.")

        if not args.all_related:
            # Option 3 (default): Filter to NLLB languages
            codes_to_find = [iso for iso in codes_to_find if iso in NLLB_ISO_SET]
            logger.info(f"\nFound {len(codes_to_find)} specified or related languages in NLLB:\n{codes_to_find}")
        # Option 1: All related files (no filtering) is handled by not applying the NLLB filter
        else:
            logger.info(f"\nFound {len(codes_to_find)} specified or related languages:\n{codes_to_find}")

    # Get all possible 2 and 3 letter codes for the related languages
    all_possible_codes = get_equivalent_isocodes(codes_to_find)

    # Find files matching the codes
    files = get_files_by_iso(all_possible_codes, scripture_dir)

    # Filter out AI and XRI files, and others.
    excluded_patterns = [
        "XRI",
        "600M",
        "3.3B",
        "1.3B",
        "words",
        "name",
        "clean",
        "transcription",
        "matthew",
        "mark",
        "mrk",
        "luk",
    ]
    filtered_files = filter_files(files, excluded_patterns)
    print(f"There are {len(files)} files and {len(files)-len(filtered_files)} were filtered out.")

    existing_projects, missing_projects = split_files_by_projects(filtered_files, projects_dir)

    # Display results
    if existing_projects:
        logger.info(f"\nThese {len(existing_projects)} files have a corresponding project folder:")
        for file, project in existing_projects.items():
            logger.info(f"{file.stem}, {project}")
        logger.info("")
    if missing_projects:
        logger.info(f"\nThese {len(missing_projects)} files don't have a corresponding project folder:")
        for file, _ in missing_projects.items():
            logger.info(f"{file.stem}")
    logger.info("\nFiltered files:")
    for file in filtered_files:
        logger.info(f"    - {file.stem}")

    if not files:
        logger.info("\nCouldn't find any Scripture files in these languages.")


if __name__ == "__main__":
    main()
