import argparse
import os
import shutil
import time
from datetime import timedelta
from typing import Dict, Iterable

from nlp.alignment.config import ALIGNERS, get_aligner
from nlp.common.utils import get_align_root_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligns the parallel corpus for an experiment")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    parser.add_argument("--aligners", nargs="*", metavar="aligner", default=[], help="Aligners")
    args = parser.parse_args()

    aligner_ids: Iterable[str] = ALIGNERS.keys() if len(args.aligners) == 0 else args.aligners

    for exp_name in args.experiments:
        print(f"Aligning {exp_name}...")
        root_dir = get_align_root_dir(exp_name)

        train_src_path = os.path.join(root_dir, "src.txt")
        train_trg_path = os.path.join(root_dir, "trg.txt")

        durations_path = os.path.join(root_dir, "durations.csv")
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
            print(f"=== {aligner.name} ===")
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


if __name__ == "__main__":
    main()
