import argparse
import os
import subprocess

from nlp.common.utils import get_git_revision_hash, get_root_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains a SMT model using SIL.Machine.Translator")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)

    src_file_path = os.path.join(root_dir, "train.src.txt")
    trg_file_path = os.path.join(root_dir, "train.trg.txt")
    engine_dir = os.path.join(root_dir, f"engine{os.sep}")

    subprocess.run(
        [
            "dotnet",
            "translator",
            "train",
            engine_dir,
            "-s",
            src_file_path,
            "-t",
            trg_file_path,
            "-st",
            "latin",
            "-tt",
            "latin",
        ]
    )


if __name__ == "__main__":
    main()
