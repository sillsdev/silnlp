import argparse
import logging
import re
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from silnlp.common.environment import SIL_NLP_ENV

LOGGER = logging.getLogger(__package__ + ".create_experiment_series")


def extract_isocode(source_string):
    """Extract the ISO code (first part before dash) from a source string."""
    if not isinstance(source_string, str):
        # print(f"Warning: extract_isocode received non-string input: {type(source_string)}, value: {source_string}")
        return None  # Return None for non-string inputs
    match = re.match(r"^([^-]+)", source_string)
    if match:
        return match.group(1)
    return source_string


def get_folder_name(sources, target):
    """Generate folder name from sources and target of first corpus pair."""
    # Extract isocodes from sources and target
    isocodes = [extract_isocode(src) for src in sources]
    target_code = extract_isocode(target)

    # Add target code to the list
    isocodes.append(target_code)

    # Remove duplicates while preserving order
    unique_codes = []
    for code in isocodes:
        if code not in unique_codes:
            unique_codes.append(code)

    # Join with underscores
    base_name = "_".join(unique_codes)

    # Check if we need a suffix higher than 1
    all_isocodes = [extract_isocode(src) for src in sources]
    all_isocodes.append(target_code)

    # Count occurrences of each ISO code
    code_counts = Counter(all_isocodes)

    # Use _1 as default, but increment if we have duplicated language codes
    has_duplicates = any(count > 1 for count in code_counts.values())

    if has_duplicates:
        return f"{base_name}_1"
    else:
        return f"{base_name}_1"


def find_config_file(path_arg):
    """Find the config.yml file based on the input path using SIL_NLP_ENV."""
    base_path = SIL_NLP_ENV.mt_experiments_dir

    path = Path(path_arg)
    if path.is_file() and path.name == "config.yml":
        LOGGER.info(f"Found config.yml with fully specified path: {path}")
        return path, f"Found config.yml with fully specified path: {path}"

    if path.is_dir() and (path / "config.yml").exists():
        LOGGER.info(f"Found config.yml in directory: {path}")
        return path / "config.yml", f"Found config.yml in directory: {path}"

    full_path = base_path / path_arg
    LOGGER.info(f"Looking in {full_path} for a config.yml file.")
    if full_path.is_dir() and (full_path / "config.yml").exists():
        LOGGER.info(f"Found config.yml in experiments dir: {full_path}")
        return full_path / "config.yml", f"Found config.yml in experiments dir: {full_path}"

    if (base_path / path_arg / "config.yml").exists():
        LOGGER.info(f"Found config.yml in experiments dir: {base_path / path_arg}")
        return base_path / path_arg / "config.yml", f"Found config.yml in experiments dir: {base_path / path_arg}"

    LOGGER.error(f"Could not find config.yml using path: {path_arg}")
    raise FileNotFoundError(f"Could not find config.yml using path: {path_arg}")


def get_relative_path(path, base="MT/experiments"):
    """Get path relative to a base directory."""
    path_str = str(path)
    base_index = path_str.find(base)
    if base_index != -1:
        return path_str[base_index + len(base) + 1 :]  # +1 for the slash
    return path_str


def get_input_dir_relative_path(input_dir, base="MT/experiments"):
    """Get the input directory path relative to the base directory."""
    if isinstance(input_dir, str):
        input_dir_str = input_dir
    else:
        input_dir_str = str(input_dir)

    # Clean path by standardizing separators
    input_dir_str = input_dir_str.replace("\\", "/")
    base = base.replace("\\", "/")

    # Check if the base path is in the input directory path
    base_index = input_dir_str.lower().find(base.lower())
    if base_index != -1:
        return input_dir_str[base_index + len(base) + 1 :]  # +1 for the slash

    # If not found, just return the last parts of the path
    parts = input_dir_str.split("/")
    if len(parts) >= 3:
        return "/".join(parts[-3:])  # Return last 3 parts like Country/Language/Folder
    return input_dir_str


def validate_language_codes(config):
    """Validate that all source and target language codes are defined in lang_codes section."""
    lang_codes = config["data"].get("lang_codes", {})
    missing_codes = set()

    for pair in config["data"]["corpus_pairs"]:
        # Check source language codes
        # Assuming pair['src'] is always a list of strings
        if isinstance(pair.get("src"), list):
            for src_item_str in pair["src"]:
                if not isinstance(src_item_str, str):
                    LOGGER.warning(
                        f"Expected string in source language list, but got {type(src_item_str)}: {src_item_str} in pair {pair}"
                    )
                    continue
                code = extract_isocode(src_item_str)
                if code and code not in lang_codes:  # Check if code is not None
                    missing_codes.add(code)
        elif pair.get("src") is not None:
            LOGGER.warning(
                f"Source languages 'src' in pair {pair} is not a list. Skipping source validation for this pair."
            )

        # Check target language code(s)
        trg_value = pair.get("trg")
        targets_to_check = []
        if isinstance(trg_value, str):
            targets_to_check.append(trg_value)
        elif isinstance(trg_value, list):
            for item in trg_value:
                if isinstance(item, str):
                    targets_to_check.append(item)
                else:
                    LOGGER.warning(
                        f"Non-string item '{item}' of type {type(item)} found in target list for pair {pair}. Skipping this item."
                    )
        elif trg_value is not None:
            LOGGER.warning(
                f"Target language 'trg' in pair {pair} is of unexpected type {type(trg_value)}. Value: {trg_value}. Skipping target validation for this pair."
            )

        for trg_lang_str in targets_to_check:
            trg_code = extract_isocode(trg_lang_str)
            if trg_code and trg_code not in lang_codes:  # Check if trg_code is not None
                missing_codes.add(trg_code)
    return list(missing_codes)


def calculate_folder_names(config, parent_dir):
    """Calculate all folder names that will be needed."""
    first_pair = config["data"]["corpus_pairs"][0]
    original_sources = first_pair["src"].copy()  # Keep original sources for iteration
    original_target_val = first_pair["trg"]

    folder_names = []

    # Determine the string to use for target in naming and ISO code extraction
    target_str_for_ops = None
    if isinstance(original_target_val, str):
        target_str_for_ops = original_target_val
    elif isinstance(original_target_val, list):
        if original_target_val:  # list is not empty
            if isinstance(original_target_val[0], str):
                target_str_for_ops = original_target_val[0]
            else:
                LOGGER.warning(f"First element of target list is not a string: {original_target_val[0]}")
            if len(original_target_val) > 1:
                LOGGER.warning(
                    f"Target is a list {original_target_val}, using first element '{target_str_for_ops}' for folder name logic."
                )
        else:  # list is empty
            LOGGER.warning(f"Target list is empty. Cannot determine target for folder name logic.")
    else:  # Not a string or list
        LOGGER.warning(
            f"Target is neither string nor list: {type(original_target_val)}. Cannot determine target for folder name logic."
        )

    current_sources_iter = original_sources.copy()
    # For each number of sources (removing one at a time)
    while len(current_sources_iter) > 1:  # Stop when we reach the last source
        current_sources_iter.pop()

        # Prepare list of ISO codes for base_name construction
        base_name_isos = [extract_isocode(src) for src in current_sources_iter if isinstance(src, str)]
        target_iso_for_name = extract_isocode(target_str_for_ops)
        if target_iso_for_name:
            base_name_isos.append(target_iso_for_name)

        # Get the base name without suffix
        unique_base_name_isos = []
        for code in base_name_isos:
            if code and code not in unique_base_name_isos:  # Ensure code is not None and unique
                unique_base_name_isos.append(code)
        base_name = "_".join(unique_base_name_isos)

        # Check if we need a suffix higher than 1 by analyzing the original full list
        # Get all unique isocodes in the original list
        all_isocodes_for_suffix_check = [extract_isocode(src) for src in current_sources_iter if isinstance(src, str)]
        if target_iso_for_name:  # Use the same target_iso extracted for base_name
            all_isocodes_for_suffix_check.append(target_iso_for_name)

        # Count occurrences of each isocodes in the original full config
        isocode_counts = Counter(code for code in all_isocodes_for_suffix_check if code is not None)

        # Determine suffix based on duplicate counts
        has_duplicates = any(count > 1 for count in isocode_counts.values())

        # Start with suffix 1
        suffix = 1

        # Try folders with incremented suffix if duplicates exist
        while True:
            folder_name = f"{base_name}_{suffix}"
            folder_path = parent_dir / folder_name

            # If no duplicates or folder doesn't exist, we can use this name
            if not has_duplicates or not folder_path.exists():
                break

            # If there are duplicates and folder exists, try next suffix
            suffix += 1

        folder_names.append((folder_name, parent_dir / folder_name, current_sources_iter.copy()))

    return folder_names


def generate_configs(config_file_path, overwrite=False):
    """Generate multiple configs by removing one source at a time from the first corpus pair."""
    # Read the original config file
    config_path = Path(config_file_path)
    input_dir = config_path.parent
    parent_dir = input_dir.parent  # Go up one level from the config file

    LOGGER.info(f"Reading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check for translate_config.yml
    translate_config_path = input_dir / "translate_config.yml"
    has_translate_config = translate_config_path.exists()

    if has_translate_config:
        LOGGER.info(f"Found translate_config.yml in {input_dir}")
    else:
        LOGGER.info(f"No translate_config.yml found in {input_dir}")

    # Validate language codes
    missing_codes = validate_language_codes(config)
    if missing_codes:
        LOGGER.warning(
            f"The following language codes are used but not defined in lang_codes section: {', '.join(missing_codes)}"
        )

    # Track created folders and their status for reporting
    folder_status = []
    created_folders = [get_relative_path(input_dir)]

    # We only modify the first corpus pair
    if not config["data"]["corpus_pairs"]:
        LOGGER.error("No corpus pairs found in the config file.")
        return created_folders, has_translate_config, folder_status

    # Calculate all folder names we'll need (reducing sources series)
    folder_configs = calculate_folder_names(config, parent_dir)

    LOGGER.info(f"Found {len(config['data']['corpus_pairs'][0]['src'])} sources in the first corpus pair")

    # Process each folder (reducing sources series)
    for folder_name, folder_path, sources in folder_configs:
        folder_existed = folder_path.exists()
        config_existed = (folder_path / "config.yml").exists()
        translate_config_existed = (folder_path / "translate_config.yml").exists()

        # If folder exists but we're not overwriting, skip file operations
        if folder_existed and not overwrite:
            folder_action = "Found existing folder (use --overwrite to update files)"
            config_action = "Left existing config.yml unchanged"
            translate_action = "Left existing translate_config.yml unchanged"
        else:
            # Create folder if it doesn't exist
            if not folder_existed:
                folder_path.mkdir(exist_ok=True)
                folder_action = "Created new folder"
            else:
                folder_action = "Found existing folder"

            # Create a new config with current sources
            new_config = deepcopy(config)
            new_config["data"]["corpus_pairs"][0]["src"] = sources

            # Write the new config
            with open(folder_path / "config.yml", "w") as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
            config_action = "Created new config.yml" if not config_existed else "Overwrote existing config.yml"

            # Copy translate_config.yml if it exists
            if has_translate_config:
                shutil.copy2(translate_config_path, folder_path / "translate_config.yml")
                translate_action = (
                    "Created new translate_config.yml"
                    if not translate_config_existed
                    else "Overwrote existing translate_config.yml"
                )
            else:
                translate_action = "No translate_config.yml to copy"

        # Add to created folders list and status tracking
        rel_path = get_relative_path(folder_path)
        created_folders.append(rel_path)
        folder_status.append(
            {
                "path": rel_path,
                "folder_action": folder_action,
                "config_action": config_action,
                "translate_action": translate_action,
                "sources_remaining": len(sources),
            }
        )

    # Now add single-source experiments for each source in the original list
    first_pair = config["data"]["corpus_pairs"][0]
    original_sources = first_pair["src"]
    target = first_pair["trg"]

    for single_src in original_sources:
        sources = [single_src]
        folder_name = get_folder_name(sources, target)
        folder_path = parent_dir / folder_name
        folder_existed = folder_path.exists()
        config_existed = (folder_path / "config.yml").exists()
        translate_config_existed = (folder_path / "translate_config.yml").exists()

        # If folder exists but we're not overwriting, skip file operations
        if folder_existed and not overwrite:
            folder_action = "Found existing folder (use --overwrite to update files)"
            config_action = "Left existing config.yml unchanged"
            translate_action = "Left existing translate_config.yml unchanged"
        else:
            # Create folder if it doesn't exist
            if not folder_existed:
                folder_path.mkdir(exist_ok=True)
                folder_action = "Created new folder"
            else:
                folder_action = "Found existing folder"

            # Create a new config with only the single source
            new_config = deepcopy(config)
            new_config["data"]["corpus_pairs"][0]["src"] = sources

            # Write the new config
            with open(folder_path / "config.yml", "w") as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
            config_action = "Created new config.yml" if not config_existed else "Overwrote existing config.yml"

            # Copy translate_config.yml if it exists
            if has_translate_config:
                shutil.copy2(translate_config_path, folder_path / "translate_config.yml")
                translate_action = (
                    "Created new translate_config.yml"
                    if not translate_config_existed
                    else "Overwrote existing translate_config.yml"
                )
            else:
                translate_action = "No translate_config.yml to copy"

        # Add to created folders list and status tracking
        rel_path = get_relative_path(folder_path)
        created_folders.append(rel_path)
        folder_status.append(
            {
                "path": rel_path,
                "folder_action": folder_action,
                "config_action": config_action,
                "translate_action": translate_action,
                "sources_remaining": 1,
            }
        )

    return created_folders, has_translate_config, folder_status


def update_notes_file(parent_dir, created_folders, has_translate_config):
    """Update the Notes.txt file with preprocessing and experiment commands."""
    # Update Notes.txt
    # Use SIL_NLP_ENV.mt_experiments_dir if parent_dir is not absolute
    if not parent_dir.is_absolute():
        parent_dir = SIL_NLP_ENV.mt_experiments_dir / parent_dir
    notes_path = parent_dir / "Notes.txt"
    notes_existed = notes_path.exists()

    # Generate preprocess commands
    preprocess_commands = []
    experiment_commands = []

    for folder in created_folders:
        # Convert path format to use backslashes for Windows compatibility
        folder_path = folder.replace("/", "\\")

        # Strip any "Drive:/MT/experiments/" prefix if present
        if "MT\\experiments\\" in folder_path:
            folder_path = folder_path.split("MT\\experiments\\")[1]

        # Add preprocess command
        preprocess_commands.append(f"poetry run python -m silnlp.nmt.preprocess --stats {folder_path}")

        # Add experiment command (with --translate only if translate_config.yml exists)
        translate_option = " --translate" if has_translate_config else ""
        experiment_commands.append(
            f"poetry run python -m silnlp.nmt.experiment --save-checkpoints --clearml-queue jobs_backlog{translate_option} {folder_path}"
        )

    # Write to Notes.txt
    with open(notes_path, "a") as notes_file:
        # Write preprocess commands
        notes_file.write("\n".join(preprocess_commands) + "\n")
        # Add a blank line between blocks
        notes_file.write("\n")
        # Write experiment commands
        notes_file.write("\n".join(experiment_commands) + "\n")

    action = "Updated existing Notes.txt" if notes_existed else "Created new Notes.txt"
    return action


def main():
    parser = argparse.ArgumentParser(description="Generate ML configuration files with reduced sources.")
    parser.add_argument("config_path", help="Path to a config.yml file or directory containing a config.yml file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing configuration files")

    args = parser.parse_args()

    try:
        config_file, location_msg = find_config_file(args.config_path)
        LOGGER.info(location_msg)

        # Get parent directory for Notes.txt
        parent_dir = config_file.parent.parent

        created_folders, has_translate_config, folder_status = generate_configs(config_file, args.overwrite)

        # Summary of actions
        LOGGER.info("\n=== Folder Generation Summary ===")
        for status in folder_status:
            LOGGER.info(f"\nFolder: {status['path']} ({status['sources_remaining']} sources)")
            LOGGER.info(f"  {status['folder_action']}")
            LOGGER.info(f"  {status['config_action']}")
            LOGGER.info(f"  {status['translate_action']}")

        # Only update Notes.txt if not in overwrite mode
        if not args.overwrite:
            notes_action = update_notes_file(parent_dir, created_folders, has_translate_config)
            LOGGER.info(f"\n{notes_action} with {len(created_folders)} commands for preprocessing and experiments")
        else:
            LOGGER.info("\nSkipped Notes.txt update (--overwrite mode)")

        LOGGER.info(
            f"\nProcessed {len(folder_status)} configurations from {len(created_folders[0].split('/')[0])} sources"
        )
        LOGGER.info("Configuration generation completed successfully.")

    except FileNotFoundError as e:
        LOGGER.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
