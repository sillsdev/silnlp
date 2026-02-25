import argparse
import csv
import logging
import re
from pathlib import Path

import yaml
import openpyxl

from silnlp.common.environment import SIL_NLP_ENV

from .script_utils import is_represented, predict_script_code
from .utils import two2three_iso

EXPERIMENTS_DIR = SIL_NLP_ENV.mt_experiments_dir
SCRIPTURE_DIR = SIL_NLP_ENV.mt_scripture_dir

LOGGER = logging.getLogger(__package__ + ".create_experiments")
SCRIPT_CACHE_FILE = "scripts.csv"
SAMPLE_SIZE = 3000  # bytes to read for script detection


def check_required_files(main_folder, workbook_file, template_config):
    """Check that required files exist"""
    print(f"Looking for the required files in the {main_folder} as follows:")
    print(f"Exists | Filename               | Purpose")
    print(f"{workbook_file.is_file()}   | {workbook_file.name:25}  | Define the experiments to be run.")
    print(f"{template_config.is_file()}   | {template_config.name:25} | Define the config.yml for the experiments.")

    if not workbook_file.is_file():
        LOGGER.error(f"\nExperiment workbook not found: {workbook_file}")
        return 1

    if not template_config.is_file():
        LOGGER.warning(
            f"\n{template_config}, not found, create {template_config} and try again."
        )
        return 1
    return 0


def read_experiments_xlsx(workbook_path):
    """Read the 'experiments' sheet from the workbook. Returns list of dicts."""
    wb = openpyxl.load_workbook(workbook_path, read_only=True)
    ws = wb["experiments"]
    rows_iter = ws.iter_rows(values_only=True)
    headers = [str(h).strip() for h in next(rows_iter)]
    rows = []
    for row in rows_iter:
        row_dict = {headers[i]: (str(v).strip() if v is not None else "") for i, v in enumerate(row)}
        rows.append(row_dict)
    wb.close()
    return rows


def get_scripts(workbook_path, rows, two2three_map):
    """Return dict of filename -> lang_code. Reads cached entries from the 'scripts'
    sheet in the workbook, predicts scripts for any new filenames, and updates
    the sheet if new entries were added."""
    cache = {}

    # Read existing cache from 'scripts' sheet if it exists
    wb = openpyxl.load_workbook(workbook_path)
    if "scripts" in wb.sheetnames:
        ws = wb["scripts"]
        rows_iter = ws.iter_rows(values_only=True)
        headers = [str(h).strip() for h in next(rows_iter)]
        for row in rows_iter:
            d = {headers[i]: (str(v).strip() if v is not None else "") for i, v in enumerate(row)}
            cache[d["filename"]] = {"iso": d["iso"], "lang_code": d["lang_code"]}

    # Collect all filenames needed
    needed = set()
    for row in rows:
        needed.update([row["Source 1"], row["Source 2"], row["Target"]])
    needed.discard("")

    # Add missing entries
    updated = False
    for filename in sorted(needed):
        if filename in cache:
            continue
        filepath = SCRIPTURE_DIR / f"{filename}.txt"
        if not filepath.is_file():
            LOGGER.warning(f"Cannot cache script for {filename}: {filepath} not found")
            continue
        try:
            text = filepath.read_text(encoding="utf-8-sig")[:SAMPLE_SIZE]
            script_code = predict_script_code(text)
            iso = extract_prefix(filename)
            three_letter = two2three_map.get(iso, iso)
            lang_code = f"{three_letter}_{script_code}"
            cache[filename] = {"iso": iso, "lang_code": lang_code}
            LOGGER.info(f"Cached script for {filename}: {lang_code}")
            updated = True
        except Exception as e:
            LOGGER.error(f"Error predicting script for {filename}: {e}")

    # Write back if updated
    if updated:
        if "scripts" in wb.sheetnames:
            del wb["scripts"]
        ws = wb.create_sheet("scripts")
        ws.append(["filename", "iso", "lang_code"])
        for fn in sorted(cache):
            ws.append([fn, cache[fn]["iso"], cache[fn]["lang_code"]])
        wb.save(workbook_path)
        LOGGER.info(f"Updated scripts sheet in {workbook_path} ({len(cache)} entries)")
    else:
        wb.close()

    return {fn: cache[fn]["lang_code"] for fn in cache}


def extract_prefix(project_name):
    """Extract the prefix (everything before the first dash) from a project name."""
    if not isinstance(project_name, str):
        return None
    match = re.match(r"^([^-]+)", project_name)
    if match:
        return match.group(1)
    return project_name


def resolve_lang_code(project_name, script_map):
    """Look up the lang_code for a project from the script cache."""
    lang_code = script_map.get(project_name)
    if not lang_code:
        raise RuntimeError(f"Could not find lang_code for {project_name} in scripts cache")
    return lang_code


def read_corpus_stats(stats_file):
    """Read script information from Align/corpus-stats.csv."""

    if not stats_file.is_file():
        LOGGER.error(f"File not found: {stats_file}")
        return {}

    script_mapping = {}
    try:
        content = stats_file.read_text(encoding="utf-8")
        lines = content.splitlines()

        if not lines:
            return {}

        # Try to detect if it's tab or comma from the first line
        delimiter = "\t" if "\t" in lines[0] else ","

        reader = csv.DictReader(lines, delimiter=delimiter)
        for row in reader:
            src_proj = row.get("src_project")
            trg_proj = row.get("trg_project")

            if src_proj:
                script_mapping[src_proj] = row.get("src_script")
            if trg_proj:
                script_mapping[trg_proj] = row.get("trg_script")

        LOGGER.info(f"Read {len(script_mapping)} script entries from {stats_file}")
    except Exception as e:
        LOGGER.error(f"Error reading {stats_file}: {e}")
        raise

    return script_mapping


def create_alignment_config(folder, rows):
    print(f"Creating Alignment config. Run the alignments before continuing.")
    all_src = set()
    all_trg = set()
    for row in rows:
        all_src.add(row["Source 1"])
        if row["Source 2"]:
            all_src.add(row["Source 2"])
        all_trg.add(row["Target"])

    config = {
        "data": {
            "aligner": "eflomal",
            "corpus_pairs": [
                {
                    "type": "train",
                    "src": sorted(list(all_src)),
                    "trg": sorted(list(all_trg)),
                    "mapping": "many_to_many",
                    "test_size": 0,
                    "val_size": 0,
                }
            ],
        }
    }

    align_dir = folder / "Align"
    align_dir.mkdir(exist_ok=True)
    with open(align_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    LOGGER.info(f"Created alignment config: {align_dir / 'config.yml'}")
    exit(0)


def check_scripture_files(rows):
    """Check that all Source 1, Source 2, and Target scripture files exist. Returns list of valid rows."""
    valid, missing_any = [], False
    for row in rows:
        src1, src2, trg = row["Source 1"], row["Source 2"], row["Target"]
        missing = [p for p in (src1, src2, trg) if not (SCRIPTURE_DIR / f"{p}.txt").is_file()]
        if missing:
            missing_any = True
            lang = row["Target_language"]
            LOGGER.warning(f"Skipping {lang}: missing scripture files: {', '.join(f'{m}.txt' for m in missing)}")
        else:
            valid.append(row)
    if not missing_any:
        LOGGER.info("All scripture files present.")
    return valid


def main():
    parser = argparse.ArgumentParser(description="Create NLLB experiment configurations with alignment and templates.")
    parser.add_argument("folder", help="Root experiment folder name (relative to mt_experiments_dir).")
    parser.add_argument(
        "--create-alignment-config", action="store_true", help="Create or update the Align/config.yml file."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing experiment configs.")
    parser.add_argument(
        "--template", default="config.yml", help="Path to a template YAML file. Defaults to 'config.yml' in the folder."
    )
    parser.add_argument(
        "--check-files", action="store_true", help="Check that scripture files exist for all experiments and exit."
    )

    args = parser.parse_args()
    main_folder = EXPERIMENTS_DIR / args.folder
    workbook_file = main_folder / "experiments.xlsx"
    two2three_file = main_folder / "two2three.csv"
    template_config = main_folder / args.template


    check_required_files(main_folder, workbook_file, template_config)
    rows = read_experiments_xlsx(workbook_file)

    if args.check_files:
        check_scripture_files(rows)
        return 0

    if args.create_alignment_config:
        create_alignment_config(main_folder, rows)

    # Main experiment generation
    valid_rows = check_scripture_files(rows)
    script_map = get_scripts(workbook_file, valid_rows, two2three_iso)

    if not script_map:
        LOGGER.error(f"\nCould not determine scripts for any projects.")
        return 1

    
    with open(template_config, "r", encoding="utf-8") as f:
        template_data = yaml.safe_load(f)

    for row in valid_rows:
        language = row["Target_language"]
        src1 = row["Source 1"]
        src2 = row["Source 2"]
        trg = row["Target"]
        corpus_books = row["corpus_books"]
        test_books = row["test_books"]

        experiments = [
            ("single", "one_to_one", [src1]),
            ("mixed", "mixed_src", [src1, src2]),
            ("many", "many_to_many", [src1, src2]),
        ]

        for suffix, mapping_type, src_list in experiments:
            if suffix != "single" and not src2:
                continue

            folder_name = f"{language}_{suffix}"
            folder_path = main_folder / folder_name
            folder_path.mkdir(exist_ok=True)

            config_file = folder_path / "config.yml"
            if config_file.is_file() and not args.overwrite:
                LOGGER.info(f"Skipping existing config: {config_file}")
                continue

            # lang_codes
            lang_codes = {}
            # We need to resolve all prefixes: src1, trg, and src2 (if it exists)
            projects_to_resolve = [src1, trg]
            if src2:
                projects_to_resolve.append(src2)

            for proj in projects_to_resolve:
                prefix = extract_prefix(proj)
                lang_codes[prefix] = resolve_lang_code(proj, script_map)
                
                if not lang_codes[prefix]:
                    raise RuntimeError(f"Could not find lang_code for {prefix} for {project_name}. Not present on scripts sheet in {workbook_file}.")

            # Special case: val,test pair uses only first source
            # The user example showed: src: tgl-TCB (not a list)

            config = {
                "data": {
                    "corpus_pairs": [
                        {
                            "type": "train",
                            "corpus_books": corpus_books,
                            "mapping": mapping_type,
                            "src": src_list,
                            "trg": trg,
                        },
                        {"type": "val,test", "corpus_books": test_books, "src": src1, "trg": trg},
                    ],
                    "lang_codes": lang_codes,
                }
            }

            # Merge with template
            # In the template, tokenizer and seed are top-level but should be under data
            for key, value in template_data.items():
                if key in ["tokenizer", "seed"]:
                    config["data"][key] = value
                else:
                    config[key] = value

            with open(config_file, "w", encoding="utf-8") as cf:
                yaml.dump(config, cf, default_flow_style=False, sort_keys=False)

            LOGGER.info(f"Created experiment config: {config_file}")

    return 0


if __name__ == "__main__":
    main()
