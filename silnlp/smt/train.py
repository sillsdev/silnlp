import argparse
import os
import subprocess

from ..common.utils import get_git_revision_hash, get_mt_root_dir, get_repo_dir
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an SMT model using the Machine library")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    for exp_name in args.experiments:
        print(f"Training {exp_name}...")
        root_dir = get_mt_root_dir(exp_name)
        config = load_config(exp_name)

        src_file_path = os.path.join(root_dir, "train.src.txt")
        trg_file_path = os.path.join(root_dir, "train.trg.txt")
        engine_dir = os.path.join(root_dir, f"engine{os.sep}")

        subprocess.run(
            [
                "dotnet",
                "machine",
                "train",
                "translation-model",
                engine_dir,
                src_file_path,
                trg_file_path,
                "-st",
                "latin",
                "-tt",
                "latin",
                "-mt",
                config["model"],
            ],
            cwd=get_repo_dir(),
        )


if __name__ == "__main__":
    main()
