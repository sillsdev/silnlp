import argparse
import csv
import logging
import re
from pathlib import Path

import yaml

from silnlp.common.environment import SIL_NLP_ENV

LOGGER = logging.getLogger(__package__ + ".create_many2many_experiments")

LANG_CODES = {
    "blx": "blx_Latn",
    "ded": "ded_Latn",
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "gbi": "gbi_Latn",
    "gbj": "gbj_Orya",
    "id": "ind_Latn",
    "kxv": "kxv_Telu",
    "kzf": "kzf_Latn",
    "npi": "npi_Latn",
    "ory": "ory_Orya",
    "rjs": "rjs_Deva",
    "swh": "swh_Latn",
    "tel": "tel_Telu",
    "tgl": "tgl_Latn",
    "tl": "tgl_Latn",
    "wbi": "wbi_Latn",
}




def extract_isocode(source_string):
    """Extract the ISO code (first part before dash) from a source string."""
    if not isinstance(source_string, str):
        return None
    match = re.match(r"^([^-]+)", source_string)
    if match:
        return match.group(1)
    return source_string


def create_config(mapping_type, src_list, trg, corpus_books, test_books):
    config = {
        "data": {
            "corpus_pairs": [
                {
                    "corpus_books": corpus_books,
                    "mapping": mapping_type,
                    "src": src_list,
                    "trg": trg,
                    "test_books": test_books,
                    "type": "train,test",
                }
            ],
            "lang_codes": LANG_CODES,
            "seed": 111,
        },
        "model": "facebook/nllb-200-distilled-1.3B",
    }
    return config

def create_alignment_config(folder, rows):
    """Create the alignment config.yml file in a folder named 'Align'"""
    
    align_config = align_dir / "config.yml"

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
    align_config_existed = align_config.is_file()

    with open(align_config, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    if align_config_existed:
        print(f"Overwrote the alignment config: {align_config}")
    else:
        print(f"Created alignment config: {align_config}")


def create_config2(mapping_type, src_list, trg, corpus_books, test_books):
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
                    "type": "val,test",
                    "corpus_books": test_books,
                    "src": src_list[0],
                    "trg": trg,
                },
            ],
            "lang_codes": LANG_CODES,
            "seed": 111,
            "tokenizer": {"update_src": True, "update_trg": True},
        },
       "eval": {"early_stopping": None, "eval_steps": 1000, 'eval_strategy': 'no'},
       "model": "facebook/nllb-200-distilled-1.3B",
       "train": {"max_steps": 7000, "save_steps": 5000, "save_strategy": "steps", "save_total_limit": 1},
    }
    return config


def create_experiments(exp_dir, csv_path, overwrite):
    """Write all the experiment config.yml files"""

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            language = row["Target_language"]
            src1 = row["Source 1"]
            src2 = row["Source 2"]
            trg = row["Target"]
            corpus_books = row["corpus_books"]
            test_books = row["test_books"]

            if not src2:
                raise RuntimeError(f"Source 2 is missing for language {language}")

            experiments = [
                ("single", "one_to_one", [src1]),
                ("mixed", "mixed_src", [src1, src2]),
                ("many", "many_to_many", [src1, src2]),
            ]

            for suffix, mapping_type, src_list in experiments:
                folder_name = f"{language}_{suffix}"
                folder_path = exp_dir / folder_name
                folder_path.mkdir(exist_ok=True)

                config_file = folder_path / "config.yml"
                config_file_existed = config_file.is_file()

                if config_file_existed and not overwrite:
                    print(f"Overwrite not set, skipping existing experiment config: {config_file}")
                    continue

                config = create_config2(mapping_type, src_list, trg, corpus_books, test_books)

                with open(config_file, "w", encoding="utf-8") as cf:
                    yaml.dump(config, cf, default_flow_style=False, sort_keys=False)
                if config_file_existed:
                    print(f"Overwrote the experiemnt config file: {config_file}")
                else:
                    print(f"Created the experiemnt config file: {config_file}")
                
    return 0


def main():
    parser = argparse.ArgumentParser(description="Create many-to-many NLLB experiment configurations from a CSV file.")
    parser.add_argument("folder", help="Root experiment folder name (relative to mt_experiments_dir).")    
    parser.add_argument("csv_file", help="Path to the CSV file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing experiment configuration files.")
    parser.add_argument("--create_alignment_config", action="store_true", help="Create or update the Align/config.yml file.")
    parser.add_argument("--template", default="config.yml", help="Path to a template YAML file. Defaults to 'config.yml' in the folder.")

    args = parser.parse_args()
        
    exp_dir = SIL_NLP_ENV.mt_experiments_dir / args.folder
    csv_file = exp_dir / args.csv_file
    config_template_file = exp_dir / args.template
    two2three_file = exp_dir / "two2three.csv"
    align_dir = exp_dir / "Align"
    corpus_stats = align_dir / "corpus-stats.csv"

    LOGGER.info(f"\nLooking for corpus-stats.csv in {align_dir}")


    if not csv_file.is_file():
        LOGGER.error(f"\nExperiment defining CSV file not found: {csv_file}")
        return 1

    if not align_dir.is_dir():
        LOGGER.info(f"\nAlign dir {align_dir} doesn't exist, will create it and the alignment config.")
        align_dir.mkdir()
        args.create_alignment_config = True

    if not corpus_stats.is_file():
        LOGGER.info(f"\nCorpus Stats file: {corpus_stats} doesn't exist, will create the alignment config.")
        args.create_alignment_config = True
    
    if not two2three_file.is_file():
        LOGGER.warning(f"\nReminder: Create {two2three_file} to show three letter equivalents of two letter iso codes and try again.")
        return 0

    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.create_alignment_config:
        create_alignment_config(main_folder, rows)

    return create_experiments(exp_dir, csv_file, args.overwrite)

    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
