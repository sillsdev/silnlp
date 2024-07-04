import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import yaml

from .environment import SIL_NLP_ENV
from .iso_info import ALT_ISO
from .find_by_iso import NLLB_LANG_CODES, load_language_data

NLLB_ISOS = NLLB_LANG_CODES.keys()

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
    prompt = "\n" + prompt + "\nChoose Y, N, or Q to quit: "
    while True:
        choice = input(prompt).strip().lower()
        if choice in ["y", "n", "q"]:
            break
        print("Invalid choice, please choose Y, N, or Q.")
    
    if choice == "y":
        return True
    elif choice == "n":
        return False
    else:
        sys.exit(0)

def get_equivalent_isocodes(iso_codes: Union[IsoCode, List[IsoCode], IsoCodeSet]) -> IsoCodeSet:
    add_iso_codes = set()

    for iso_code in iso_codes:
        add_iso_codes.update(set(iso_code))
        add_iso = ALT_ISO.get_alternative(iso_code)
        if add_iso is not None:
            add_iso_codes.update(set(add_iso))
    return add_iso_codes


def get_tla_iso(iso_code) -> IsoCode:
    if len(iso_code) == 3:
        return iso_code
    if len(iso_code) == 2:
        tla_code = ALT_ISO.get_alternative(iso_code)
        print(f"Iso code is {iso_code} tla_code is {tla_code}")
        exit()
    
    if len(tla_code) == 3:
        return tla_code
    else:
        return None


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
        with open(config_file, "w", encoding="utf-8") as f:
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
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader if float(row["count"]) >= 5000 and float(row["align_score"]) >= 0.2]


def trim_date(project_name):
    return re.sub(r"_\d{4}_\d{2}_\d{2}$", "", project_name)


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


def create_align_config(
    align_dir: Path, target_file: str, related_scriptures: List[str], source_files: List[str]
) -> Path:
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
    return align_config_file


def run_alignments(experiment_series_dir: Path):
    align_dir = experiment_series_dir / "align"
    command = f"poetry run python -m silnlp.nmt.preprocess --stats {align_dir}"
    print(command)
    subprocess.run(command, shell=True, check=True)


def run_preprocess(experiment: str):
    command = f"poetry run python -m silnlp.nmt.experiment --preprocess {experiment}"
    print(command)
    subprocess.run(command, shell=True, check=True)


def run_training(experiment: str, translate):
    if translate:
        command = f"poetry run python -m silnlp.nmt.experiment --save-checkpoints --memory-growth --clearml-queue jobs_backlog --translate {experiment}"
    else:
        command = f"poetry run python -m silnlp.nmt.experiment --save-checkpoints --memory-growth --clearml-queue jobs_backlog {experiment}"
    print(command)
    subprocess.run(command, shell=True, check=True)


def filter_verse_counts(experiment_series_dir):
    command = f"poetry run python -m silnlp.common.filter_verse_counts {experiment_series_dir}"
    subprocess.run(command, shell=True, check=True)


def create_experiment_config(experiment_dir: Path, target: str, sources: List[str], lang_codes: Dict[str, str]):
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
            "lang_codes": lang_codes,
            "seed": 111,
        },
        "eval": {
            "detokenize": False,
            "early_stopping": {"min_improvement": 0.1, "steps": 4},
            "per_device_eval_batch_size": 16,
        },
        "infer": {"infer_batch_size": 16, "num_beams": 2},
        "model": "facebook/nllb-200-distilled-1.3B",
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
        "-v",
        "--verbose",
        action="store_true",
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
    if related_nllb_isos:
        print(f"Of these {len(related_nllb_isos)} languages are known to NLLB.")
        for related_nllb_iso in related_nllb_isos:
            lang_info = LANGUAGE_DATA[related_nllb_iso]
            print(f"{related_nllb_iso}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}")
    else:
        print(f"None of these {len(related_isos)} languages are known to NLLB.")

    if related_nllb_isos:
        related_scriptures = find_scriptures_by_iso(set(related_nllb_isos), scripture_dir)
        print(f"Found {len(related_scriptures)} in the {scripture_dir} folder.")
    else:
        related_scriptures = []

    if related_scriptures:
        print(f"\nFound {len(related_scriptures)} scripture files in the related languages in NLLB.")
        for related_scripture in related_scriptures:
            print(f"{related_scripture}")
    else:
        print(f"\nDidn't find any scripture files in related languages that are in NLLB.")

    source_files = [source.split(".")[0] for source in args.sources]  # Remove file extension if present
    source_isos = set(extract_iso_code(source) for source in args.sources)
    source_nllb_isos = select_nllb_languages(source_isos)

    if source_nllb_isos:
        print(f"Found {len(source_nllb_isos)} source languages also known to NLLB.")
        for source_nllb_iso in source_nllb_isos:
            lang_info = LANGUAGE_DATA[source_nllb_iso]
            print(f"{source_nllb_iso}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}")

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

    corpus_stats_file = experiment_series_dir / "align" / "corpus-stats.csv"
    filtered_rows = read_corpus_stats(corpus_stats_file)

    lang_codes = {}
    for row in filtered_rows:
        src_project = row["src_project"]
        trg_project = row["trg_project"]

        experiment_dir = create_experiment_folder(experiment_series_dir, trg_project=trg_project, src_project=src_project)

        # Extract language codes
        src_lang_iso = extract_iso_code(src_project)
        trg_lang_iso = extract_iso_code(trg_project)

        tla_src_lang_iso = get_tla_iso(src_lang_iso)
        tla_trg_lang_iso = get_tla_iso(trg_lang_iso)

        if tla_src_lang_iso in NLLB_LANG_CODES:
            lang_codes[src_lang_iso] = NLLB_LANG_CODES[tla_src_lang_iso]
        else:
            lang_codes[src_lang_iso] = f"{src_lang_iso}_Latn"

        if tla_trg_lang_iso in NLLB_LANG_CODES:
            lang_codes[trg_lang_iso] = NLLB_LANG_CODES[tla_trg_lang_iso]
        else:
            lang_codes[trg_lang_iso] = f"{trg_lang_iso}_Latn"
        
        experiment_config_file = create_experiment_config(experiment_dir,  target=trg_project, sources=[src_project], lang_codes=lang_codes)

        # if a source_project and books are specified create translate_config.yml in the experiment folder
        if source_project and books:
            translate = True
            create_translate_config(experiment_dir, source_project, books, checkpoint="best")
        else:
            translate = False

        two_folders = '/'.join(experiment_dir.parts[-2:])
        
        # Run preprocessing
        if choose_yes_no_cancel(
            f"The experiment config is {experiment_config_file}. You can edit that file now.\nWould you like to run preprocessing now?"
        ):
            run_preprocess(two_folders)

        # Run training
        if choose_yes_no_cancel(f"Would you like to train the model now?"):
            run_training(two_folders, translate=translate)


if __name__ == "__main__":
    main()
