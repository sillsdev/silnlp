import argparse
import logging
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .config import ALIGNERS, get_aligner, get_aligner_name, get_all_book_paths, load_config
from .utils import get_experiment_dirs, get_experiment_name

LOGGER = logging.getLogger(__package__ + ".align")


def align(aligner_ids: List[str], exp_dir: Path, book: Optional[str] = None) -> None:
    train_src_path = exp_dir / "src.txt"
    train_trg_path = exp_dir / "trg.txt"

    durations_path = exp_dir / "duration.csv"
    times: Dict[str, str] = {}
    if durations_path.is_file():
        with open(durations_path, "r", encoding="utf-8") as in_file:
            first_line = True
            for line in in_file:
                if first_line:
                    first_line = False
                    continue
                line = line.strip()
                aligner_name, delta_str = line.split(",")
                times[aligner_name] = delta_str

    for aligner_id in aligner_ids:
        aligner_id = aligner_id.strip().lower()
        aligner = get_aligner(aligner_id, exp_dir)
        if aligner.model_dir.is_dir():
            shutil.rmtree(aligner.model_dir)
        aligner_name = get_aligner_name(aligner_id)
        if book is None:
            LOGGER.info(f"Aligning using {aligner_name}")
        else:
            LOGGER.info(f"Aligning {book} using {aligner_name}")
        method_alignments_file_path = exp_dir / f"alignments.{aligner_id}.txt"

        start = time.perf_counter()
        aligner.train(train_src_path, train_trg_path)
        aligner.align(method_alignments_file_path)
        end = time.perf_counter()
        delta = timedelta(seconds=end - start)
        times[aligner_name] = str(delta)
        LOGGER.info(f"{aligner_name} duration: {delta}")

    with open(durations_path, "w", encoding="utf-8") as out_file:
        out_file.write("Model,Duration\n")
        for aligner_name, delta_str in times.items():
            out_file.write(f"{aligner_name},{delta_str}\n")


def extract_lexicons(aligner_ids: List[str], exp_dir: Path) -> None:
    for aligner_id in aligner_ids:
        aligner_id = aligner_id.strip().lower()
        aligner = get_aligner(aligner_id, exp_dir)
        aligner_name = get_aligner_name(aligner_id)
        print(f"--- {aligner_name} ---")
        aligner.extract_lexicon(exp_dir / f"lexicon.{aligner_id}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligns the parallel corpus for an experiment")
    parser.add_argument("experiments", type=str, help="Experiment pattern")
    parser.add_argument("--aligners", nargs="*", metavar="aligner", default=[], help="Aligners")
    parser.add_argument("--skip-align", default=False, action="store_true", help="Skips aligning corpora")
    parser.add_argument("--skip-extract-lexicon", default=False, action="store_true", help="Skips extracting lexicons")
    args = parser.parse_args()

    aligner_ids = list(ALIGNERS.keys() if len(args.aligners) == 0 else args.aligners)

    for exp_dir in get_experiment_dirs(args.experiments):
        exp_name = get_experiment_name(exp_dir)
        if not args.skip_align:
            LOGGER.info(f"Aligning {exp_name}")
            config = load_config(exp_dir)

            if config["by_book"]:
                for book, book_exp_dir in get_all_book_paths(exp_dir):
                    if book_exp_dir.is_dir():
                        align(aligner_ids, book_exp_dir, book)
            else:
                align(aligner_ids, exp_dir)

        if not args.skip_extract_lexicon:
            LOGGER.info(f"Extracting lexicons {exp_name}")
            extract_lexicons(aligner_ids, exp_dir)


if __name__ == "__main__":
    main()
