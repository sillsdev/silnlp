import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from .environment import SIL_NLP_ENV
from .iso_info import ALT_ISO, NLLB_ISO_SET

IsoCode = str
IsoCodeList = List[IsoCode]
IsoCodeSet = Set[IsoCode]

LANGUAGE_FAMILY_FILE = SIL_NLP_ENV.assets_dir / "languageFamilies.json"


def is_file_pattern(input_str: str) -> bool:
    """Check if the input string contains a hyphen, indicating it's a filename pattern."""
    return "-" in input_str


def split_input_list(input_list: List[str]) -> Tuple[List[str], List[str]]:
    """Split input list into ISO codes and file patterns."""
    iso_codes = []
    files = []
    for item in input_list:
        if is_file_pattern(item):
            files.append(item)
        else:
            iso_codes.append(item)
    return iso_codes, files


def get_stem_name(file_path: Path) -> str:
    """Get the stem name without path or extension."""
    return file_path.stem


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
            #            logger.info(f"{iso_code}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}")

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


def create_alignment_config(source_files: List[Path], target_files: List[str]) -> dict:
    """Create the alignment configuration dictionary."""
    config = {
        "data": {
            "aligner": "eflomal",
            "corpus_pairs": [
                {
                    "type": "train",
                    "src": source_files,
                    "trg": target_files,
                    "mapping": "many_to_many",
                    "test_size": 0,
                    "val_size": 0,
                }
            ],
            "tokenize": False,
        }
    }
    return config


def resolve_config_file(output_path: str) -> Path:
    "Resolve config folder/file path, creating the config folder if needed."
    path = Path(output_path)
    if path.suffix:
        config_file_name = path.name
        folder_part = path.parent
        if config_file_name != 'config.yml':
            resp = input(f"Warning: filename '{config_file_name}' is not 'config.yml'. Use 'config.yml' instead? [Y/n] ")
            if resp.strip().lower() != 'n': config_file_name = 'config.yml'
    else:
        config_file_name = 'config.yml'
        folder_part = path

    if folder_part.is_absolute():
        config_folder = folder_part
        if not config_folder.parent.is_dir():
            raise argparse.ArgumentTypeError(
                f"Parent directory does not exist: {config_folder.parent}")
        try: config_folder.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise argparse.ArgumentTypeError(f"Permission denied creating directory: {config_folder}")
    else:
        if len(folder_part.parts) < 2:
            raise argparse.ArgumentTypeError(
                f"Relative path '{folder_part}' must include a subfolder inside Experiments (e.g. 'country/analyze').")
        config_folder = (SIL_NLP_ENV.mt_experiments_dir / folder_part).resolve()
        try: config_folder.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise argparse.ArgumentTypeError(f"Permission denied creating directory: {config_folder}")

    return config_folder / config_file_name


def main():
    parser = argparse.ArgumentParser(description="Find related ISO language codes and create alignment config.")
    parser.add_argument("inputs", nargs="+", help="ISO codes or file patterns (e.g., 'fra' or 'en-NIV')")
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
    parser.add_argument("--targets", nargs="+", help="List of target files in format <iso_code>-<project_name>")
    parser.add_argument("--config-folder", type=resolve_config_file, help=f"Existing folder, or folder relative to the Experiments folder: {SIL_NLP_ENV.mt_experiments_dir}")

    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Split inputs into ISO codes and file patterns
    iso_codes, file_patterns = split_input_list(args.inputs)

    source_files = []
    if iso_codes:
        # Load language data and process ISO codes
        language_data, country_data, family_data = load_language_data(LANGUAGE_FAMILY_FILE)
        if not language_data:
            logging.error("Failed to load language data.")
            return

        iso_codes = get_equivalent_isocodes(iso_codes)

        if args.no_related:
            codes_to_find = list(iso_codes)
            logger.info(f"\nConsidering only the specified iso codes and their equivalents: {codes_to_find}")
        else:
            codes_to_find = find_related_isocodes(list(iso_codes), language_data, country_data, family_data)
            logger.info(f"\nFound {len(codes_to_find)} related languages:\n{codes_to_find}.")

            if not args.all_related:
                codes_to_find = [iso for iso in codes_to_find if iso in NLLB_ISO_SET]
                logger.info(f"\nFound {len(codes_to_find)} specified or related languages in NLLB:\n{codes_to_find}")
            else:
                logger.info(f"\nFound {len(codes_to_find)} specified or related languages:\n{codes_to_find}")

        # Get all possible codes and find matching files
        all_possible_codes = get_equivalent_isocodes(codes_to_find)
        source_files.extend(get_files_by_iso(all_possible_codes, args.scripture_dir))

    # Add files from file patterns
    if file_patterns:
        pattern_files = [args.scripture_dir / f"{pattern}.txt" for pattern in file_patterns]
        existing_files = [f for f in pattern_files if f.exists()]
        source_files.extend(existing_files)
        if len(existing_files) < len(pattern_files):
            missing = set(file_patterns) - set(get_stem_name(f) for f in existing_files)
            logger.warning(f"Could not find these files: {missing}")

    if not source_files:
        logger.error("\nCouldn't find any Scripture files.")
        return

    # Use target files from command line or file patterns from inputs 
    target_files = args.targets if args.targets else file_patterns
    targets = sorted([target_file for target_file in set(target_files)])

    # Filter out targets from the source list keep only unique source and targets.
    sources = [get_stem_name(f) for f in source_files if get_stem_name(f) not in target_files]
    sources = sorted([source for source in set(sources)])

    # Create and output configuration
    config = create_alignment_config(sources, targets)

    if args.config_folder:
        config_file = args.config_folder
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"\nCreated alignment configuration in: {config_file}")
    else:
        logger.info("\nAlignment configuration:")
        logger.info(yaml.dump(config, default_flow_style=False, sort_keys=False))

    logger.info(f"\nSource files found: {len(sources)}")
    for source in sources:
        logger.info(f"    - {source}")
    logger.info(f"\nTarget files: {len(targets)}")
    for target in targets:
        logger.info(f"    - {target}")


if __name__ == "__main__":
    main()
