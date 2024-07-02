import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple, Union
import yaml

from .environment import SIL_NLP_ENV
from .find_by_iso import NLLB_LANG_CODES, load_language_data
NLLB_ISOS = NLLB_LANG_CODES.keys()

#print(len(NLLB_LANG_CODES), len(NLLB_ISOS))
#print(set(NLLB_ISOS) == set(NLLB_LANG_CODES.keys()))


IsoCode = str
IsoCodeSet = Set[IsoCode]

LANGUAGE_FAMILY_FILE = SIL_NLP_ENV.assets_dir / "languageFamilies.json"
LANGUAGE_DATA, COUNTRY_DATA, FAMILY_DATA = load_language_data(LANGUAGE_FAMILY_FILE)
THREE_TO_TWO_CSV = SIL_NLP_ENV.assets_dir / "three_to_two.csv"

def extract_iso_code(scripture: str) -> IsoCode:
    iso = scripture.split("-", maxsplit=1)[0]
    if len(iso) not in [2, 3]:
        raise ValueError(f"Couldn't guess the ISO code from the scripture filename: {scripture}")
    return iso


def choose_yes_no_cancel(prompt: str) -> bool:
    prompt += "\nChoose Y, N or Q to quit."
    choice: str = " "
    while choice not in ["n", "y", "c", "q"]:
        choice: str = input(prompt).strip()[0].lower()
    if choice == "y":
        return True
    elif choice == "n":
        return False
    elif choice in ["q","c"]:
        sys.exit(0)


def read_iso_codes(csv_file: Path) -> Dict[IsoCode, IsoCodeSet]:
    iso_dict = {}
    with open(csv_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            three_letter, two_letter = row["Three"], row["Two"]
            iso_dict.setdefault(three_letter, set()).add(two_letter)
            iso_dict.setdefault(two_letter, set()).add(three_letter)
    return iso_dict


ISO_EQUIVALENTS = read_iso_codes(THREE_TO_TWO_CSV)

def get_equivalent_isocodes(iso_codes: Union[IsoCode, List[IsoCode], IsoCodeSet]) -> IsoCodeSet:
    if isinstance(iso_codes, str):
        iso_codes = [iso_codes]
    result = set(iso_codes)
    for iso_code in iso_codes:
        result.update(ISO_EQUIVALENTS.get(iso_code, set()))
    return result


def find_related_languages(iso_codes: Union[IsoCode, List[IsoCode], IsoCodeSet], verbose=False) -> IsoCodeSet:
    iso_codes = get_equivalent_isocodes(iso_codes)
    related_codes = set()

    for iso_code in iso_codes:
        if iso_code in LANGUAGE_DATA:
            lang_info = LANGUAGE_DATA[iso_code]
            country_language_isos = COUNTRY_DATA.get(lang_info["Country"])
            family_language_isos = FAMILY_DATA.get(lang_info["Family"])
            
            if verbose:
                print(f"\nFound these {len(country_language_isos)} languages of {lang_info['Country']}")
                for country_language_iso in country_language_isos:
                    lang_info = LANGUAGE_DATA[country_language_iso]
                    print(f"{country_language_iso} : {lang_info['Name']} : language of {lang_info['Country']}.")
                    
                print(f"\nFound these {len(family_language_isos)} languages of family {lang_info['Family']}")
                for family_language_iso in family_language_isos:
                    lang_info = LANGUAGE_DATA[family_language_iso]
                    print(f"{family_language_iso} : {lang_info['Name']} : language of {lang_info['Family']}.")
                    
            related_codes.update(country_language_isos, [])
            related_codes.update(family_language_isos, [])

    return iso_codes.union(related_codes)


def find_scripture_file_info(scripture: str, scripture_dir: Path) -> Tuple[IsoCode, Path]:
    iso = extract_iso_code(scripture)

    scripture_filename = scripture if scripture.endswith(".txt") else f"{scripture}.txt"
    scripture_file = scripture_dir / scripture_filename

    if not scripture_file.is_file():
        raise FileNotFoundError(f"Couldn't find scripture file {scripture_file} in {scripture_dir}")

    return iso, scripture_file


def find_scriptures_by_iso(isos: IsoCodeSet, scripture_dir: Path) -> List[str]:
    return [
        filepath.stem
        for filepath in scripture_dir.iterdir()
        if filepath.suffix == ".txt" and filepath.stem.split("-")[0] in isos
    ]


def select_nllb_languages(isocodes: IsoCodeSet) -> List[IsoCode]:
    return sorted(isocodes.intersection(set(NLLB_ISOS)))


def confirm_write_config(config, config_file, message):
    if not config_file.is_file():
        with open(config_file, "w", encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"{message} {config_file} was created.")
    else:
        if choose_yes_no_cancel(f"Would you like to overwrite the current {message} at {config_file}"):
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"{message} {config_file} was overwritten.")
        else:
            print(f"{message} {config_file} was not overwritten.")


def read_corpus_stats(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader if float(row['filtered_count']) >= 5000 and float(row['filtered_align_score']) >= 0.2]


def trim_date(project_name):
    return re.sub(r'_\d{4}_\d{2}_\d{2}$', '', project_name)

def create_experiment_folder(experiment_series_dir, src_project, trg_project):
    src_no_date = trim_date(src_project)
    trg_no_date = trim_date(trg_project)
    experiment_dir = experiment_series_dir / f"{src_no_date}_{trg_no_date}_1"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def create_experiment_series_folder(experiment_name: str) -> Tuple[Path, Path]:
    experiment_series_dir = SIL_NLP_ENV.mt_experiments_dir / experiment_name
    experiment_series_dir.mkdir(parents=True, exist_ok=True)
    align_dir = experiment_series_dir / "align"
    align_dir.mkdir(parents=True, exist_ok=True)
    return experiment_series_dir, align_dir


def create_align_config(align_dir: Path, target_file: str, related_scriptures: List[str], source_files: List[str]):
    config = {
        "data": {
            "aligner": "fast_align",
            "corpus_pairs": [
                {
                    "type": "train",
                    "src": source_files + related_scriptures,
                    "trg": [target_file],
                    "mapping": "many_to_many",
                    "test_size": 0,
                    "val_size": 0,
                }
            ],
            "tokenize": False,
        }
    }

    align_config_file = align_dir / "config.yml"
    confirm_write_config(config, align_config_file, message="Alignment config file")
    return 


def run_alignments(experiment_series_dir: Path):
    align_dir = experiment_series_dir / "align"
    command = f"poetry run python -m silnlp.nmt.preprocess --stats {align_dir}"
    subprocess.run(command, shell=True, check=True)


def run_preprocess(experiment: str):
    command = f"poetry run python -m silnlp.nmt.experiment --preprocess {experiment}"
    subprocess.run(command, shell=True, check=True)


def run_training(experiment: str, translate):
    if translate:
        command = f"poetry run python -m silnlp.nmt.experiment --save-checkpoints --memory-growth --clearml-queue jobs_backlog --translate {experiment}"
    else:
        command = f"poetry run python -m silnlp.nmt.experiment --save-checkpoints --memory-growth --clearml-queue jobs_backlog {experiment}"
    
    subprocess.run(command, shell=True, check=True)

def filter_verse_counts(experiment_series_dir):
    command = f"poetry run python -m silnlp.common.filter_verse_counts {experiment_series_dir}"
    subprocess.run(command, shell=True, check=True)


def create_experiment_config(experiment_dir: Path, target: str, sources: List[str], iso_codes: Dict[str, str]):
    config = {
        "data": {
            "corpus_pairs": [
                {
                    "corpus_books": "NT",
                    "mapping": "mixed_src",
                    "src": sources,
                    "test_size": 250,
                    "trg": target,
                    "type": "train,test,val",
                    "val_size": 250,
                }
            ],
            "lang_codes": iso_codes,
            "seed": 111,
        },
        "eval": {
            "detokenize": False,
            "early_stopping": {"min_improvement": 0.1, "steps": 4},
            "per_device_eval_batch_size": 16,
        },
        "infer": {"infer_batch_size": 16, "num_beams": 2},
        "model": "facebook/nllb-200-1.3B",
        "params": {"label_smoothing_factor": 0.2, "learning_rate": 0.0002, "warmup_steps": 4000},
        "train": {"gradient_accumulation_steps": 4, "per_device_train_batch_size": 16},
    }

    config_file = experiment_dir / "config.yml"
    confirm_write_config(config, config_file, message="Experiment config file")
    return config_file


def create_translate_config(experiment_dir: Path, source_project: str, books: List[str], checkpoint: str = "best"):

    config = {"translate": [{"books": books, "src_project": source_project, "checkpoint": checkpoint}]}
    config_file = experiment_dir / "translate_config.yml"
    confirm_write_config(config, config_file, message="Translate config file")


def main():
    parser = argparse.ArgumentParser(description="Create experiment folders and run alignments for Scripture files.")
    parser.add_argument("target", type=str, help="Name of a target scripture file.")
    parser.add_argument("sources", type=str, nargs="*", help="Names of scripture files to use as sources.")
    parser.add_argument("--experiment", type=str, help="Name of experiment series folder to create.")
    parser.add_argument(
        "--source-project",
        type=Path,
        help="Path to the source project folder to translate. Books must also be specified.",
    )
    parser.add_argument(
        "--books",
        type=str,
        default=None,
        help="List of books to translate. Books must be specified if a source is given.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action='store_true',
        help="Print list of related languages.",
    )
    args = parser.parse_args()
    # Create the argument parser
    source_project = str(args.source_project)
    books = args.books

    scripture_dir = SIL_NLP_ENV.mt_scripture_dir
    target_iso, target_file = find_scripture_file_info(args.target, scripture_dir)

    target_isos = get_equivalent_isocodes(target_iso)
#    target_lang_name = None

    for iso in target_isos:
        if iso in LANGUAGE_DATA:
            lang_info = LANGUAGE_DATA[iso]
            target_lang_name = lang_info["Name"]
            print(f"Found {target_file} in language: {lang_info['Name']} spoken in {lang_info['Country']}.")
            print(f"{iso} is in language family: {lang_info['Family']}")

    related_isos = find_related_languages(target_isos, verbose=args.verbose)
    print(f"Found {len(related_isos)} languages spoken in the same country or in the same language family.")

    related_nllb_isos = select_nllb_languages(related_isos)
    print(f"Found {len(related_nllb_isos)} languages also known to NLLB.")
    for related_nllb_iso in related_nllb_isos:
        lang_info = LANGUAGE_DATA[related_nllb_iso]
        print(f"{related_nllb_iso}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}")

    related_scriptures = find_scriptures_by_iso(set(related_nllb_isos), scripture_dir)
    print(f"Found {len(related_scriptures)} in the {scripture_dir} folder.")
    for related_scripture in related_scriptures:
        print(f"{related_scripture}")

    source_files = [source.split(".")[0] for source in args.sources]  # Remove file extension if present

    # Create experiment folder and config file
    experiment_name = args.experiment or f"FT-{target_lang_name}"
    experiment_series_dir, align_dir = create_experiment_series_folder(experiment_name)
    align_config = create_align_config(align_dir, args.target.split(".")[0], related_scriptures, source_files)

    # Run alignments
    if choose_yes_no_cancel(
        f"The alignment config is {align_config}. You can edit that file now.\nWould you like to run the alignments now?"
    ):
        run_alignments(experiment_series_dir)
        filter_verse_counts(experiment_series_dir)

    # Load data from the alignment file: <experiment_series_dir>/align/corpus-stats.csv 
    # The columns have headings: src_project,trg_project,project_count,count,align_score,filtered_count,filtered_align_score
    # Filter the rows so only those where the filtered_count >= 5000, and the filtered_align_score >= 0.2
    # Each of these rows will become an experiment - ie a folder with a config.yml file and optionally a translate_config.yml file.
    
    # Create experiment_dir folder name for each filtered row:
        # Trim dates from the end of the src_project and trg_project that are of the form '_yyyy_mm_dd'
        # Use this template for the experiment_dir f"{src_project}_{trg_project}_1"
        # Create the folder experiment_series_dir/experiment_dir
        # Create the config.yml file with the complete list of lang_codes, and the full src_project and trg_project as src: and trg:
    
    corpus_stats_file = experiment_series_dir / "align" / "corpus-stats.csv"
    filtered_rows = read_corpus_stats(corpus_stats_file)
    
    for row in filtered_rows:
        src_project = row['src_project']
        trg_project = row['trg_project']
        
        experiment_dir = create_experiment_folder(experiment_series_dir, src_project, trg_project)
        
        # Extract language codes 
        src_lang_iso = extract_iso_code(src_project)
        trg_lang_iso = extract_iso_code(trg_project)    

        lang_codes = {src_lang_iso: NLLB_LANG_CODES[src_lang_iso] , trg_lang_iso: NLLB_LANG_CODES[trg_lang_iso]}
        
        experiment_config_file = create_experiment_config(experiment_dir, src_project, trg_project, lang_codes)
        
        # if a source_project and books are specified create translate_config.yml in the experiment folder
        if source_project and books:
            translate = True
            create_translate_config(experiment_dir, source_project, books, checkpoint="best")
            
        # Run preprocessing
        if choose_yes_no_cancel(
            f"The experiment config is {experiment_config_file}. You can edit that file now.\nWould you like to run preprocessing now?"
        ):
            run_preprocess(experiment_dir)

        # Run training
        if choose_yes_no_cancel(
            f"Would you like to train the model now?"
        ):
            run_training(experiment_dir, translate=translate)

if __name__ == "__main__":
    main()
