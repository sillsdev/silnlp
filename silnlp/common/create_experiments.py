import argparse
import csv
import logging
import re
from pathlib import Path

import yaml

from silnlp.common.environment import SIL_NLP_ENV

EXPERIMENTS_DIR = SIL_NLP_ENV.mt_experiments_dir
SCRIPTURE_DIR = SIL_NLP_ENV.mt_scripture_dir

LOGGER = logging.getLogger(__package__ + ".create_experiments")

def extract_prefix(project_name):
    """Extract the prefix (everything before the first dash) from a project name."""
    if not isinstance(project_name, str):
        return None
    match = re.match(r"^([^-]+)", project_name)
    if match:
        return match.group(1)
    return project_name

def read_two2three(folder):
    """Read two2three.csv mapping."""
    mapping_file = folder / "two2three.csv"
    if not mapping_file.is_file():
        # Try .tsv just in case
        mapping_file = folder / "two2three.tsv"
        if not mapping_file.is_file():
            LOGGER.warning(f"Neither two2three.csv nor two2three.tsv found in {folder}")
            return {}
    
    mapping = {}
    try:
        content = mapping_file.read_text(encoding="utf-8")
        lines = content.splitlines()
            
        # If it's .tsv use \t, else use ,
        delimiter = "\t" if mapping_file.suffix == ".tsv" else ","
        reader = csv.reader(lines, delimiter=delimiter)
        for row in reader:
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
        LOGGER.info(f"Read {len(mapping)} mappings from {mapping_file}")
    except Exception as e:
        LOGGER.error(f"Error reading {mapping_file}: {e}")
        raise
        
    return mapping

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

def resolve_lang_code(project_name, two2three_map, script_map):
    prefix = extract_prefix(project_name)
    if not prefix:
        raise RuntimeError(f"Could not extract prefix from {project_name}")
        
    three_letter = two2three_map.get(prefix, prefix)
    
    script = script_map.get(project_name)
    if not script:
        raise RuntimeError(f"Could not find script for {project_name} in corpus-stats.csv")
        
    return f"{three_letter}_{script}"


def create_alignment_config(folder, rows):
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
                    "val_size": 0
                }
            ]
        }
    }
    
    align_dir = folder / "Align"
    align_dir.mkdir(exist_ok=True)
    with open(align_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    LOGGER.info(f"Created alignment config: {align_dir / 'config.yml'}")


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
        else: valid.append(row)
    if not missing_any: LOGGER.info("All scripture files present.")
    return valid


def main():
    parser = argparse.ArgumentParser(description="Create NLLB experiment configurations with alignment and templates.")
    parser.add_argument("folder", help="Root experiment folder name (relative to mt_experiments_dir).")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("--create-alignment-config", action="store_true", help="Create or update the Align/config.yml file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing experiment configs.")
    parser.add_argument("--template", default="config.yml", help="Path to a template YAML file. Defaults to 'config.yml' in the folder.")
    parser.add_argument("--check-files", action="store_true", help="Check that scripture files exist for all experiments and exit.")

    args = parser.parse_args()
        
    main_folder = EXPERIMENTS_DIR / args.folder
    csv_file = main_folder / args.csv_file
    two2three_file = main_folder / "two2three.csv"
    align_dir = main_folder / "Align"
    corpus_stats = align_dir / "corpus-stats.csv"
    print(f"\nLooking in {main_folder} for files:\n{csv_file.is_file()}\t{csv_file.name}\n{two2three_file.is_file()}\t{two2three_file.name}")
    print(f"And in {align_dir} for:\n{corpus_stats.is_file()}\t{corpus_stats.name}\n")

    if not csv_file.is_file():
        LOGGER.error(f"\nExperiment defining CSV file not found: {csv_file}")
        return 1
    
    if not align_dir.is_dir():
        LOGGER.info(f"\nAlign dir {align_dir} doesn't exist, will create it.")
        align_dir.mkdir()
        args.create_alignment_config = True

    if args.create_alignment_config or not corpus_stats.is_file():
        LOGGER.info(f"\nWill create the alignment config: {corpus_stats}")
        args.create_alignment_config = True
    
    if not two2three_file.is_file():
        LOGGER.warning(f"\nReminder: Create {two2three_file} to show three letter equivalents of two letter iso codes and try again.")
        return 0

    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"This is the reader: {reader}")
        for row in reader:
            print(f"This is the row: {row}")
            rows.append(row)

    if args.check_files:
        check_scripture_files(rows)
        return 0

    if args.create_alignment_config:
        create_alignment_config(main_folder, rows)

    # Main experiment generation
    two2three_map = read_two2three(main_folder)
    script_map = read_corpus_stats(corpus_stats)
    
    if not script_map:
        LOGGER.error(f"\nProblem reading {corpus-stats}. Could not create the script_map.")
        return 1

    template_file = main_folder / args.template if args.template else main_folder / "experiment_template.yml"
    with open(template_file, "r", encoding="utf-8") as f:
        template_data = yaml.safe_load(f)

    valid_rows = check_scripture_files(rows)
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
                lang_codes[prefix] = resolve_lang_code(proj, two2three_map, script_map)
            
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
                            "trg": trg
                        },
                        {
                            "type": "val,test",
                            "corpus_books": test_books,
                            "src": src1,
                            "trg": trg
                        }
                    ],
                    "lang_codes": lang_codes
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
    logging.basicConfig(level=logging.INFO)
    main()
