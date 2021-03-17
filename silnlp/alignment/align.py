import argparse
import os
import shutil
import time
from datetime import timedelta
from typing import Dict, List, Optional, cast

from .config import ALIGNERS, get_aligner, get_aligner_name, get_all_book_paths, load_config
from .utils import get_align_exp_dir


def align(aligner_ids: List[str], testament_dir: str, book: Optional[str] = None) -> None:
    train_src_path = os.path.join(testament_dir, "src.txt")
    train_trg_path = os.path.join(testament_dir, "trg.txt")

    durations_path = os.path.join(testament_dir, "duration.csv")
    times: Dict[str, str] = {}
    if os.path.isfile(durations_path):
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
        aligner = get_aligner(aligner_id, testament_dir)
        if os.path.isdir(aligner.model_dir):
            shutil.rmtree(aligner.model_dir)
        aligner_name = get_aligner_name(aligner_id)
        if book is None:
            print(f"--- {aligner_name} ---")
        else:
            print(f"--- {aligner_name} ({book}) ---")
        method_alignments_file_path = os.path.join(testament_dir, f"alignments.{aligner_id}.txt")

        start = time.perf_counter()
        aligner.train(train_src_path, train_trg_path)
        aligner.align(method_alignments_file_path)
        end = time.perf_counter()
        delta = timedelta(seconds=end - start)
        times[aligner_name] = str(delta)
        print(f"Duration: {delta}")

    with open(durations_path, "w", encoding="utf-8") as out_file:
        out_file.write("Model,Duration\n")
        for aligner_name, delta_str in times.items():
            out_file.write(f"{aligner_name},{delta_str}\n")


def extract_lexicons(aligner_ids: List[str], testament_dir: str) -> None:
    for aligner_id in aligner_ids:
        aligner_id = aligner_id.strip().lower()
        aligner = get_aligner(aligner_id, testament_dir)
        aligner_name = get_aligner_name(aligner_id)
        print(f"--- {aligner_name} ---")
        aligner.extract_lexicon(os.path.join(testament_dir, f"lexicon.{aligner_id}.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligns the parallel corpus for an experiment")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    parser.add_argument("--nt", default=False, action="store_true", help="Align NT")
    parser.add_argument("--ot", default=False, action="store_true", help="Align OT")
    parser.add_argument("--aligners", nargs="*", metavar="aligner", default=[], help="Aligners")
    parser.add_argument("--skip-align", default=False, action="store_true", help="Skips aligning corpora")
    parser.add_argument("--skip-extract-lexicon", default=False, action="store_true", help="Skips extracting lexicons")
    args = parser.parse_args()

    testaments: List[str] = []
    if args.nt:
        testaments.append("nt")
    if args.ot:
        testaments.append("ot")
    if len(testaments) == 0:
        testaments.extend(["nt", "ot"])

    aligner_ids = list(ALIGNERS.keys() if len(args.aligners) == 0 else args.aligners)

    for exp_name in cast(List[str], args.experiments):
        for testament in testaments:
            testament_dir = os.path.join(get_align_exp_dir(exp_name), testament)

            if not args.skip_align:
                print(f"=== Aligning ({exp_name.upper()} {testament.upper()}) ===")
                config = load_config(exp_name, testament)

                if config["by_book"]:
                    for book, book_root_dir in get_all_book_paths(testament_dir):
                        if os.path.isdir(book_root_dir):
                            align(aligner_ids, book_root_dir, book)
                else:
                    align(aligner_ids, testament_dir)

            if not args.skip_extract_lexicon:
                print(f"=== Extracting lexicons ({exp_name.upper()} {testament.upper()}) ===")
                extract_lexicons(aligner_ids, testament_dir)


if __name__ == "__main__":
    main()
