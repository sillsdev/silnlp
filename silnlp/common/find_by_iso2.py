import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
import sys
import yaml

from .environment import SIL_NLP_ENV
from .iso_info import NLLB_ISO_SET, ALT_ISO

IsoCode = str
IsoCodeList = List[IsoCode]
IsoCodeSet = Set[IsoCode]

LANGUAGE_FAMILY_FILE = SIL_NLP_ENV.assets_dir / "languageFamilies.json"

def is_file_pattern(input_str: str) -> bool:
    """Check if the input string contains a hyphen, indicating it's a filename pattern."""
    return '-' in input_str

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

def resolve_config_path(config_folder: Path) -> Path:
    """Resolve config folder path relative to experiments directory if not absolute."""
    if not config_folder.is_absolute():
        return SIL_NLP_ENV.mt_experiments_dir / config_folder
    return config_folder

def create_alignment_config(source_files: List[Path], target_files: List[str]) -> dict:
    """Create the alignment configuration dictionary."""
    config = {
        'data': {
            'aligner': 'fast_align',
            'corpus_pairs': [{
                'type': 'train',
                'src': [get_stem_name(f) for f in source_files],
                'trg': target_files,
                'mapping': 'many_to_many',
                'test_size': 0,
                'val_size': 0
            }],
            'tokenize': False
        }
    }
    return config

def write_or_print_config(config: dict, config_folder: Path = None):
    """Write config to file or print to terminal."""
    if config_folder:
        config_folder = Path(config_folder)
        if not config_folder.is_absolute():
            config_folder = SIL_NLP_ENV.mt_experiments_dir / config_folder
        config_folder.mkdir(parents=True, exist_ok=True)
        config_path = config_folder / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return str(config_path)
    else:
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

def main():
    parser = argparse.ArgumentParser(description="Find related ISO language codes and create alignment config.")
    parser.add_argument("inputs", nargs="+", 
                       help="ISO codes or file patterns (e.g., 'fra' or 'en-NIV')")
    parser.add_argument("--scripture-dir", type=Path, 
                       default=Path(SIL_NLP_ENV.mt_scripture_dir), 
                       help="Directory containing scripture files")
    parser.add_argument("--all-related", action='store_true', 
                       help="List all related scriptures without filtering to those that are part of NLLB")
    parser.add_argument("--no-related", action='store_true', 
                       help="Only list scriptures in the specified languages and not in related languages")
    parser.add_argument("--output", type=Path, help="Output to the specified file.")
    parser.add_argument("--target-files", nargs="+",
                       help="List of target files in format <iso_code>-<project_name>")
    parser.add_argument("--config-folder", type=Path,
                       help="Folder to write the config.yml file (absolute or relative to mt_experiments_dir)")

    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    if args.output:
        file_handler = logging.FileHandler(args.output)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
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
    target_files = args.target_files if args.target_files else file_patterns

    # Create and output configuration
    config = create_alignment_config(source_files, target_files)
    result = write_or_print_config(config, args.config_folder)
    
    if args.config_folder:
        logger.info(f"\nCreated alignment configuration in: {result}")
    else:
        logger.info("\nAlignment configuration:")
        logger.info(result)

    logger.info(f"\nSource files found: {len(source_files)}")
    for file in source_files:
        logger.info(f"    - {get_stem_name(file)}")
    logger.info(f"\nTarget files: {len(target_files)}")
    for file in target_files:
        logger.info(f"    - {file}")

if __name__ == "__main__":
    main()