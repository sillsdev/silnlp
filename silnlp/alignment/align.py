import argparse
import os
import shutil
from typing import Iterable

from nlp.alignment.config import ALIGNERS, get_aligner
from nlp.common.utils import get_align_root_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligns the parallel corpus for an experiment")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--aligners", nargs="*", metavar="aligner", default=[], help="Aligners")
    args = parser.parse_args()

    root_dir = get_align_root_dir(args.experiment)

    aligner_ids: Iterable[str] = ALIGNERS.keys() if len(args.aligners) == 0 else args.aligners

    train_src_path = os.path.join(root_dir, "src.txt")
    train_trg_path = os.path.join(root_dir, "trg.txt")
    for aligner_id in aligner_ids:
        aligner_id = aligner_id.strip().lower()
        aligner = get_aligner(aligner_id, root_dir)
        if os.path.isdir(aligner.model_dir):
            shutil.rmtree(aligner.model_dir)
        print(f"=== {aligner.name} ===")
        method_out_file_path = os.path.join(root_dir, f"alignments.{aligner_id}.txt")
        aligner.align(train_src_path, train_trg_path, method_out_file_path)


if __name__ == "__main__":
    main()
