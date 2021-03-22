import argparse
import os
import subprocess
from typing import List

from ..common.utils import get_git_revision_hash, get_mt_exp_dir, get_repo_dir
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an SMT model using the Machine library")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    for exp_name in args.experiments:
        print(f"=== Training ({exp_name}) ===")
        exp_dir = get_mt_exp_dir(exp_name)
        config = load_config(exp_name)

        src_file_path = exp_dir / "train.src.txt"
        trg_file_path = exp_dir / "train.trg.txt"
        engine_dir = exp_dir / f"engine{os.sep}"

        args_list: List[str] = [
            "dotnet",
            "machine",
            "train",
            "translation-model",
            str(engine_dir),
            str(src_file_path),
            str(trg_file_path),
            "-st",
            config["src_tokenizer"],
            "-tt",
            config["trg_tokenizer"],
            "-mt",
            config["model"],
            "-l",
        ]
        subprocess.run(args_list, cwd=get_repo_dir())


if __name__ == "__main__":
    main()
