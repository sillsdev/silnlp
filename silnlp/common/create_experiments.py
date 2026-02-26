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
SAMPLE_LINES = 100 
MODEL = "facebook/nllb-200"
RESULT_HEADERS = [
    "Target_language", "Mapping", "Source 1", "Source 2", "Target",
    "train_lines", "unique_train_lines", "test_lines",
    "src_mean_chars_per_token", "trg_mean_chars_per_token",
    "src_mean_tokens_per_verse", "trg_mean_tokens_per_verse",
    "Book", "BLEU", "chrF3++",
]

LOGGER = logging.getLogger(__package__ + ".create_experiments")


def read_experiments_xlsx(workbook_file):
    """Read the 'experiments' sheet from the workbook. Stop at the first empty row. Returns a list of dicts."""
    wb = openpyxl.load_workbook(workbook_file, read_only=True)
    ws = wb["experiments"]
    rows_iter = ws.iter_rows(values_only=True)
    headers = [str(header).strip() for header in next(rows_iter)]
    
    rows = []
    for row in rows_iter:
        row_dict = {headers[i]: (str(v).strip() if v is not None else "") for i, v in enumerate(row)}
        if ''.join(row_dict.values()).strip() == '':
            break
        rows.append(row_dict)
    wb.close()
    return rows


def read_scripture(file, max_lines=SAMPLE_LINES):
    "Read non-empty, non-range lines from a scripture file"
    lines_to_skip = set(['', '<range>', '...'])
    with open(file, 'r', encoding='utf-8') as file_in:
        lines = [line.strip() for line in file_in if line.strip() not in lines_to_skip]
        return ''.join(lines[:max_lines])


def update_sheet(wb, workbook_path, cache):
    """Write out the filename, filename_isocode, 3 letter isocode and script to the "scripts" sheet in the spreadsheet"""
    HEADERS = ["filename", "filename_iso", "language_iso", "script"]
    if "scripts" in wb.sheetnames:
        ws = wb["scripts"]
        # Build a map of filename -> row number for existing entries
        existing = {}
        for row_num, row in enumerate(ws.iter_rows(min_row=2, values_only=False), start=2):
            fn = str(row[0].value).strip() if row[0].value else ""
            if fn:
                existing[fn] = row_num

        for fn in sorted(cache):
            c = cache[fn]
            vals = [fn, c["filename_iso"], c["language_iso"], c["script"]]
            if fn in existing:
                for col, val in enumerate(vals, start=1):
                    ws.cell(row=existing[fn], column=col, value=val)
            else:
                ws.append(vals)
    else:
        ws = wb.create_sheet("scripts")
        ws.append(HEADERS)
        for fn in sorted(cache):
            c = cache[fn]
            ws.append([fn, c["filename_iso"], c["language_iso"], c["script"]])

    wb.save(workbook_path)
    LOGGER.info(f"Updated scripts sheet in {workbook_path} ({len(cache)} entries)")


def get_scripts(workbook_path, rows, two2three_iso):
    """Return dict of filename -> lang_code. Reads cached entries from the 'scripts'
    sheet in the workbook, predicts scripts for any new filenames, and updates
    the sheet if new entries were added."""
    cache = {}

    # Read existing cache from 'scripts' sheet if it exists
    wb = openpyxl.load_workbook(workbook_path)
    if "scripts" in wb.sheetnames:
        ws = wb["scripts"]
        rows_iter = ws.iter_rows(values_only=True)
        headers = [str(header).strip() for header in next(rows_iter)]
        for row in rows_iter:
            d = {headers[i]: (str(v).strip() if v is not None else "") for i, v in enumerate(row)}
            cache[d["filename"]] = {
                "filename_iso": d["filename_iso"],
                "language_iso": d["language_iso"],
                "script": d["script"],
            }

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
        file = SCRIPTURE_DIR / f"{filename}.txt"
        if not file.is_file():
            LOGGER.warning(f"Cannot cache script for {filename}: {file} not found")
            continue
        
        script_code = predict_script_code(read_scripture(file))
        if not is_represented(script_code=script_code, model=MODEL):
            if updated:
                update_sheet(wb, workbook_path, cache)
            LOGGER.error(f"Script {script_code} found for {file} is not known to the {MODEL} model.")

        filename_iso = extract_prefix(filename)
        language_iso = two2three_iso.get(filename_iso, filename_iso)
        if not (len(language_iso) == 3 and language_iso.isalpha()):
            LOGGER.warning(f"Skipping {filename}: language_iso '{language_iso}' is not a valid 3-letter code")
            continue

        cache[filename] = {
            "filename_iso": filename_iso,
            "language_iso": language_iso,
            "script": script_code,
        }
        LOGGER.info(f"Cached script for {filename}: {language_iso}_{script_code}")
        updated = True

    # Write back if updated
    if updated:
        update_sheet(wb, workbook_path, cache)
    else:
        wb.close()

    return {fn: f"{cache[fn]['language_iso']}_{cache[fn]['script']}" for fn in cache}


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
    valid, any_missing = [], []
    for row in rows:
        lang, src1, src2, trg = row["Target_language"], row["Source 1"], row["Source 2"], row["Target"]
        missing = [file for file in (src1, src2, trg) if not (SCRIPTURE_DIR / f"{file}.txt").is_file()]
        if missing:
            any_missing.append({lang: missing})
        else:
            valid.append(row)
    return valid, any_missing


def create_config(mapping_type, lang_codes, src_list, trg, corpus_books, test_books):
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
                {
                    "type": "test",
                    "corpus_books": test_books,
                    "src": src_list[0],
                    "trg": trg,
                },
            ],
            "lang_codes": lang_codes,
            "seed": 111,
            "tokenizer": {"update_src": True, "update_trg": True},
        },
       #"eval": {"early_stopping": None, "eval_steps": 1000, 'eval_strategy': 'no'},
       "model": "facebook/nllb-200-distilled-1.3B",
       #"train": {"max_steps": 7000, "save_steps": 5000, "save_strategy": "steps", "save_total_limit": 1},
    }
    return config


def count_lines_and_unique_lines(file):
    """Count the total and number of unique lines in a file."""
    if not file.is_file():
        LOGGER.warning(f"Couldn't find train.vref file {file}")
        return None, None
    
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    return len(lines), len(set(lines))


def get_tokenization_stats(stats_file):
    """Read Mean Tokens/Verse and Mean Characters/Token for Source and Target
    from tokenization_stats.csv. Returns dict with four keys, or empty dict if file missing."""

    with open(stats_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip category header row
        next(reader)  # skip column names row
        result = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            side = row[0].strip()
            if side == "Source":
                result["src_mean_tokens_per_verse"] = float(row[5])
                result["src_mean_chars_per_token"] = float(row[17])
            elif side == "Target":
                result["trg_mean_tokens_per_verse"] = float(row[5])
                result["trg_mean_chars_per_token"] = float(row[17])
    return result


def get_scores(scores_file):
    """Read scores file. Returns list of dicts with keys: Book, BLEU, chrF3++.
    One dict per row (per-book + ALL). Returns empty list if file missing."""
    results = []
    with open(scores_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        bleu_col = next((c for c in reader.fieldnames if c.lower() == "bleu"), None)
        chrf_col = next((c for c in reader.fieldnames if "chrf" in c.lower()), None)
        book_col = next((c for c in reader.fieldnames if c.lower() == "book"), None)
        for row in reader:
            results.append({
                "Book": row[book_col].strip() if book_col else "",
                "BLEU": float(row[bleu_col]) if bleu_col and row[bleu_col].strip() else None,
                "chrF3++": float(row[chrf_col]) if chrf_col and row[chrf_col].strip() else None,
            })
    return results


def collect_results(main_folder, valid_rows, workbook_file):
    """Collect results from all experiments and write to 'results' sheet."""
    all_results = []
    for row in valid_rows:
        language = row["Target_language"]
        src1, src2, trg = row["Source 1"], row["Source 2"], row["Target"]

        for suffix, mapping in [("many", "many_to_many"), ("mixed", "mixed_src")]:
            folder = main_folder / f"{language}_{suffix}"  
            if not folder.is_dir():
                LOGGER.warning(f"Folder not found: {folder}")
                continue

            train_vref_file, test_vref_file, stats_file, scores_file = (folder / file for file in ["train.vref.txt", "test.vref.txt", "tokenization_stats.csv", "scores-5000.csv", ])
            preprocess_files = [stats_file, train_vref_file, test_vref_file]
            missing_preprocess_files = [f for f in preprocess_files if not f.is_file()]
            if missing_preprocess_files:
                LOGGER.info(f"These preprocess files are missing, skipping this experiment. {missing_preprocess_files}")
                continue

            train_lines, unique_train_lines = count_lines_and_unique_lines(train_vref_file)
            test_lines, _= count_lines_and_unique_lines(test_vref_file)
            tok_stats = get_tokenization_stats(stats_file)
            
            if scores_file.is_file():
                scores = get_scores(scores_file)
            else:
                scores = [{"Book": "", "BLEU": None, "chrF3++": None}]
                LOGGER.warning(f"{scores_file} not found in {folder}")

            for s in scores:
                all_results.append([
                    language, mapping, src1, src2, trg,
                    train_lines, unique_train_lines, test_lines,
                    tok_stats.get("src_mean_chars_per_token"),
                    tok_stats.get("trg_mean_chars_per_token"),
                    tok_stats.get("src_mean_tokens_per_verse"),
                    tok_stats.get("trg_mean_tokens_per_verse"),
                    s["Book"], s["BLEU"], s["chrF3++"],
                ])
            print(f"Found these results for {language}_{mapping} experiment.\n{all_results[-1]}")
    # # Write to results sheet
    wb = openpyxl.load_workbook(workbook_file)
    if "results" in wb.sheetnames:
        del wb["results"]
    ws = wb.create_sheet("results")
    ws.append(RESULT_HEADERS)
    for r in all_results:
        ws.append(r)
    wb.save(workbook_file)
    LOGGER.info(f"Wrote {len(all_results)} result rows to {workbook_file}")


def main():
    parser = argparse.ArgumentParser(description="Create NLLB experiment configurations with alignment and templates.")
    parser.add_argument("folder", help="Root experiment folder name (relative to mt_experiments_dir).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing experiment configs or results.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="Create experiment configs.")
    group.add_argument("--collect-results", action="store_true", help="Collect the results of the experiments.")

    args = parser.parse_args()

    main_folder = EXPERIMENTS_DIR / args.folder
    workbook_file = main_folder / "experiments.xlsx"

    if not workbook_file.is_file():
        LOGGER.error(f"\nExperiment workbook not found: {workbook_file}. This spreadsheet is required to define the experiments to be run.")
        return 1

    rows = read_experiments_xlsx(workbook_file)
    LOGGER.info(f"Read {len(rows)} experiment definitions from the 'experiments' sheet in {workbook_file}")
    valid_rows, any_missing = check_scripture_files(rows)

    if valid_rows and not any_missing:
        LOGGER.info(f"All the scripture files required for the {len(valid_rows)} experiments exist.")
    else:
        LOGGER.warning(f"The following files are missing for these experiments:")
        for lang, missing in any_missing.items():
            print(f"{lang}   : {missing}")

        exit()


    
    script_map = get_scripts(workbook_file, valid_rows, two2three_iso)
    if not script_map:
        LOGGER.error(f"\nCould not determine scripts for any projects.")
        return 1

    if args.create:
        # Main experiment generation

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

                experiment_name = f"{language}_{suffix}"
                experiment_folder = main_folder / experiment_name
                experiment_folder.mkdir(exist_ok=True)

                config_file = experiment_folder / "config.yml"
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

                config = create_config(mapping_type, lang_codes, src_list, trg, corpus_books, test_books)

                with open(config_file, "w", encoding="utf-8") as cf:
                    yaml.dump(config, cf, default_flow_style=False, sort_keys=False)

                LOGGER.info(f"Created experiment config: {config_file}")

    if args.collect_results:
        collect_results(main_folder, valid_rows, workbook_file)

    return 0


if __name__ == "__main__":
    main()
