import argparse
import os
import shutil
import time
from datetime import timedelta
from typing import Dict, List, Optional

from nlp.alignment.config import ALIGNERS, get_aligner, get_all_book_paths, load_config
from nlp.common.utils import get_align_root_dir


def align(aligner_ids: List[str], root_dir: str, book: Optional[str] = None) -> None:
    train_src_path = os.path.join(root_dir, "src.txt")
    train_trg_path = os.path.join(root_dir, "trg.txt")

    durations_path = os.path.join(root_dir, "duration.csv")
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
        aligner = get_aligner(aligner_id, root_dir)
        if os.path.isdir(aligner.model_dir):
            shutil.rmtree(aligner.model_dir)
        if book is None:
            print(f"=== {aligner.name} ===")
        else:
            print(f"=== {aligner.name} ({book}) ===")
        method_out_file_path = os.path.join(root_dir, f"alignments.{aligner_id}.txt")

        start = time.perf_counter()
        aligner.align(train_src_path, train_trg_path, method_out_file_path)
        end = time.perf_counter()
        delta = timedelta(seconds=end - start)
        times[aligner.name] = str(delta)
        print(f"Duration: {delta}")

    with open(durations_path, "w", encoding="utf-8") as out_file:
        out_file.write("Model,Duration\n")
        for aligner_name, delta_str in times.items():
            out_file.write(f"{aligner_name},{delta_str}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligns the parallel corpus for an experiment")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    parser.add_argument("--aligners", nargs="*", metavar="aligner", default=[], help="Aligners")
    args = parser.parse_args()

    aligner_ids = list(ALIGNERS.keys() if len(args.aligners) == 0 else args.aligners)

    for exp_name in args.experiments:
        print(f"Aligning {exp_name}...")
        root_dir = get_align_root_dir(exp_name)
        config = load_config(exp_name)

        if config["by_book"]:
            for book, book_root_dir in get_all_book_paths(root_dir):
                if os.path.isdir(book_root_dir):
                    align(aligner_ids, book_root_dir, book)
        else:
            align(aligner_ids, root_dir)


if __name__ == "__main__":
    main()
