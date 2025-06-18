import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
import re
import sys
import yaml

from .environment import SIL_NLP_ENV
from .iso_info import NLLB_ISO_SET, ALT_ISO

IsoCode = str
IsoCodeList = List[IsoCode]
IsoCodeSet = Set[IsoCode]

# Patterns for filtering out files. Case-insensitive.
EXCLUDE_PATTERNS_FOR_FILE_FILTERING = ["_AI", "3.3B", "600M", "1.3B", "xri", "term", "name", "words"]

LANGUAGE_FAMILY_FILE = SIL_NLP_ENV.assets_dir / "languageFamilies.json"

# --- Filtering Toggles for Target Files (Module Level for Testability) ---
APPLY_TARGET_ISO_FILTER = False  # Set to True to enable ISO code-based filtering for targets
APPLY_TARGET_DATE_FILTER = False # Set to True to enable date-based filtering for targets
# Exclusion filtering for targets is already removed, but a flag could be added if needed.

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

def should_exclude_file(file_stem_or_pattern: str) -> bool:
    """Checks if a file stem or pattern should be excluded based on hardcoded patterns."""
    lower_stem = file_stem_or_pattern.lower()
    for pattern in EXCLUDE_PATTERNS_FOR_FILE_FILTERING:
        if pattern.lower() in lower_stem:
            return True
    return False

# Regex to find _YYYY_MM_DD pattern in a filename stem
DATE_PATTERN_REGEX = re.compile(r"_(\d{4})_(\d{2})_(\d{2})")

def filter_latest_dated_stems(stems: List[str]) -> List[str]:
    """
    Filters a list of stems, keeping only the latest dated version for stems
    that match the _YYYY_MM_DD pattern. Stems without the pattern are kept.
    """
    latest_stems: Dict[str, Tuple[str, str]] = {} # prefix -> (full_stem, date_string)
    other_stems: List[str] = []

    for stem in stems:
        match = DATE_PATTERN_REGEX.search(stem)
        if match:
            prefix = stem[:match.start()]
            date_str = match.group(0) # e.g., _2025_01_01

            if prefix not in latest_stems:
                latest_stems[prefix] = (stem, date_str)
            else:
                # Compare dates lexicographically (YYYY_MM_DD format makes this easy)
                if date_str > latest_stems[prefix][1]:
                    latest_stems[prefix] = (stem, date_str)
        else:
            other_stems.append(stem)
    # Return the full stems of the latest dated files, plus all other non-dated stems
    return [data[0] for data in latest_stems.values()] + other_stems

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
        file
        for iso_code in isocodes
        for file in scripture_dir.glob(f'{iso_code}-*.txt')  # Glob with f-string
        if file.is_file()
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


    original_cmd_line_iso_codes, file_patterns = split_input_list(args.inputs)
    
    source_files_collected = [] # Temporary list to hold all initially found files
    
    # Use a working copy for finding source files, preserving original_cmd_line_iso_codes
    iso_codes_for_source_search = list(original_cmd_line_iso_codes)

    if iso_codes_for_source_search:
        # Load language data and process ISO codes
        language_data, country_data, family_data = load_language_data(LANGUAGE_FAMILY_FILE)
        projects_dir = SIL_NLP_ENV.pt_projects_dir # Define for project splitting logic
        if not language_data:
            logging.error("Failed to load language data.")
            return
        
        current_equivalent_codes = get_equivalent_isocodes(iso_codes_for_source_search)
        
        if args.no_related:
            codes_to_find = list(current_equivalent_codes)
            logger.info(f"\nConsidering only the specified iso codes and their equivalents: {codes_to_find}")
        else:
            codes_to_find = find_related_isocodes(list(current_equivalent_codes), language_data, country_data, family_data)
            logger.info(f"\nFound {len(codes_to_find)} related languages:\n{codes_to_find}.")

            if not args.all_related:
                codes_to_find = [iso for iso in codes_to_find if iso in NLLB_ISO_SET]
                logger.info(f"\nFound {len(codes_to_find)} specified or related languages in NLLB:\n{codes_to_find}")
            else:
                logger.info(f"\nFound {len(codes_to_find)} specified or related languages:\n{codes_to_find}")

        # Get all possible codes and find matching files
        all_possible_codes = get_equivalent_isocodes(codes_to_find)
        source_files_collected.extend(get_files_by_iso(list(all_possible_codes), args.scripture_dir))

    # Add files from file patterns
    if file_patterns:
        pattern_files = [args.scripture_dir / f"{pattern}.txt" for pattern in file_patterns]
        existing_files = [f for f in pattern_files if f.exists()]
        source_files_collected.extend(existing_files)
        if len(existing_files) < len(pattern_files):
            missing = set(file_patterns) - set(get_stem_name(f) for f in existing_files)
            logger.warning(f"Could not find these files: {missing}")

    # --- Source File Filtering ---
    # Apply date filtering to source_files_collected
    initial_source_stems = [get_stem_name(f) for f in source_files_collected]
    dated_filtered_source_stems = filter_latest_dated_stems(initial_source_stems)

    # Log how many source files were filtered by date
    if len(initial_source_stems) != len(dated_filtered_source_stems):
        logger.info(
            f"\nFiltered {len(initial_source_stems) - len(dated_filtered_source_stems)} source files "
            f"based on date patterns (keeping latest for each base name)."
        )
    elif source_files_collected: # Only log if there were files to begin with
        logger.info(
            f"\nNo source files were filtered out by date patterns (or no dated files found)."
        )

    # Reconstruct Path objects for source_files after date filtering
    # This ensures we keep the correct Path objects from the original collection
    source_files_after_date_filter = []
    stems_to_keep_set = set(dated_filtered_source_stems)
    for f_path in source_files_collected:
        if f_path.stem in stems_to_keep_set:
            source_files_after_date_filter.append(f_path)
            stems_to_keep_set.remove(f_path.stem) # Avoid adding duplicates if stems were somehow repeated

    # Filter the date-filtered source_files based on EXCLUDE_PATTERNS_FOR_FILE_FILTERING
    source_files = [f for f in source_files_after_date_filter if not should_exclude_file(f.stem)]
    
    # Log how many source files were filtered by exclusion patterns
    if len(source_files_after_date_filter) != len(source_files):
        logger.info(
            f"Filtered {len(source_files_after_date_filter) - len(source_files)} source files "
            f"based on exclusion name patterns (AI, 3.3B, etc.)."
        )
    elif source_files_after_date_filter : # Only log if there were files to begin with
        logger.info(
            f"No source files were filtered out by exclusion name patterns."
        )

    if not source_files:
        logger.error("\nCouldn't find any Scripture files after filtering.")
        return

    # --- Start: Added logic similar to find_by_iso.py for project logging ---
    # Ensure projects_dir is defined if iso_codes was empty but file_patterns was not
    if not original_cmd_line_iso_codes and file_patterns: # Check original ISOs
        projects_dir = SIL_NLP_ENV.pt_projects_dir
        
    if original_cmd_line_iso_codes or file_patterns: # Check original ISOs or if patterns were given
        existing_projects, missing_projects = split_files_by_projects(source_files, projects_dir) # Use filtered source_files

        if existing_projects:
            logger.info(f"\nThese {len(existing_projects)} files (after filtering) have a corresponding project folder:")
            for file_path, project_path_val in existing_projects.items(): # Renamed project to project_path_val
                logger.info(f"{get_stem_name(file_path)}, {project_path_val}")
            logger.info("")
        if missing_projects:
            logger.info(f"\nThese {len(missing_projects)} files (after filtering) don't have a corresponding project folder:")
            for file_path, _ in missing_projects.items():
                logger.info(f"{get_stem_name(file_path)}")
    
    logger.info(f"\nAll files considered for config (after filtering and project check): {len(source_files)}")
    for file_path in source_files:
        logger.info(f"    - {get_stem_name(file_path)}")
    # --- End: Added logic ---

    # --- Target File Filtering ---
    # Determine initial candidate target stems
    if args.target_files:
        candidate_target_stems = list(args.target_files) # Ensure it's a list
        source_of_target_candidates = "--target-files argument"
    else:
        # If --target-files is not given, derive from original_cmd_line_iso_codes if present
        if original_cmd_line_iso_codes:
            # Find all .txt files in scripture_dir that start with any of the original_cmd_line_iso_codes
            target_candidate_files_found = []
            for iso_code in original_cmd_line_iso_codes:
                target_candidate_files_found.extend(args.scripture_dir.glob(f"{iso_code}-*.txt"))
            # Remove duplicates that might arise if equivalent ISOs point to same files
            candidate_target_stems = sorted(list(set(get_stem_name(f) for f in target_candidate_files_found)))
            source_of_target_candidates = f"files matching command-line ISOs: {original_cmd_line_iso_codes}"
        elif file_patterns: # Fallback to file_patterns if no ISOs and no --target-files
            candidate_target_stems = list(file_patterns) # Ensure it's a list
            source_of_target_candidates = "file patterns from input"
        else:
            candidate_target_stems = []
            source_of_target_candidates = "no source (no --target-files, no command-line ISOs, no file patterns)"
    
    count_before_any_target_filtering = len(candidate_target_stems)

    # --- Target File Filtering Stages ---
    stems_after_iso_filter = candidate_target_stems
    if APPLY_TARGET_ISO_FILTER:
        if original_cmd_line_iso_codes: # Use the original command-line ISOs here
            iso_filtered_target_stems_intermediate = [
                stem for stem in stems_after_iso_filter # Start from current list
                if any(stem.startswith(f"{cmd_iso}-") for cmd_iso in original_cmd_line_iso_codes)
            ]
            if len(stems_after_iso_filter) != len(iso_filtered_target_stems_intermediate):
                logger.info(
                    f"\nFiltered target candidates from {len(stems_after_iso_filter)} (current) "
                    f"to {len(iso_filtered_target_stems_intermediate)} based on command-line ISOs: {original_cmd_line_iso_codes}"
                )
            stems_after_iso_filter = iso_filtered_target_stems_intermediate
        else:
            logger.info("\nSkipping target ISO filter as no command-line ISOs were provided.")
    else:
        logger.info("\nTarget ISO filtering is disabled by flag.")

    stems_after_date_filter = stems_after_iso_filter
    if APPLY_TARGET_DATE_FILTER:
        stems_after_date_filter_intermediate = filter_latest_dated_stems(stems_after_date_filter) # Start from current list
        if len(stems_after_date_filter) != len(stems_after_date_filter_intermediate):
            logger.info(
                f"Filtered target candidates from {len(stems_after_date_filter)} (current) to {len(stems_after_date_filter_intermediate)} "
                f"based on date patterns."
            )
        stems_after_date_filter = stems_after_date_filter_intermediate
    else:
        logger.info("\nTarget date filtering is disabled by flag.")

    # Exclusion filtering for targets was previously removed.
    # If you wanted to re-add it with a flag, it would go here, operating on stems_after_date_filter.
    filtered_target_files = stems_after_date_filter 

    # Create and output configuration
    config = create_alignment_config(source_files, filtered_target_files) # Use filtered source_files and filtered_target_files
    result = write_or_print_config(config, args.config_folder)
    
    if args.config_folder:
        logger.info(f"\nCreated alignment configuration in: {result}")
    else:
        logger.info("\nAlignment configuration:")
        logger.info(result)
    
    if count_before_any_target_filtering != len(filtered_target_files):
        logger.info(
            f"\nOverall, target candidates were filtered from {count_before_any_target_filtering} (from {source_of_target_candidates}) "
            f"to {len(filtered_target_files)} for the config.yml (due to command-line ISOs and/or date rules)."
        )
    elif not filtered_target_files and count_before_any_target_filtering > 0:
         logger.info(f"\nNo target files remaining for config.yml after all filtering stages (started with {count_before_any_target_filtering} from {source_of_target_candidates}).")

    logger.info(f"\nSource files written to config: {len(source_files)}")
    for file in source_files:
        logger.info(f"    - {get_stem_name(file)}")
    logger.info(f"\nTarget files written to config: {len(filtered_target_files)}")
    for file in filtered_target_files:
        logger.info(f"    - {file}")

if __name__ == "__main__":
    main()