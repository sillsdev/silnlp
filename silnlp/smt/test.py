import argparse
import os
import subprocess

import sacrebleu

from nlp.common.corpus import load_corpus
from nlp.common.utils import get_git_revision_hash, get_root_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a SMT model using SIL.Machine.Translator")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)

    ref_file_path = os.path.join(root_dir, "test.trg.txt")
    predictions_file_path = os.path.join(root_dir, "test.trg-predictions.txt")

    if not os.path.isfile(predictions_file_path):
        src_file_path = os.path.join(root_dir, "test.src.txt")
        engine_dir = os.path.join(root_dir, f"engine{os.sep}")
        subprocess.run(
            [
                "dotnet",
                "translator",
                "translate",
                engine_dir,
                "-s",
                src_file_path,
                "-st",
                "latin",
                "-tt",
                "latin",
                "-o",
                predictions_file_path,
            ]
        )

    sys = load_corpus(predictions_file_path)
    ref = load_corpus(ref_file_path)

    for i in range(len(sys) - 1, 0, -1):
        if ref[i] == "" or sys[i] == "":
            del sys[i]
            del ref[i]

    bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
    print(bleu)


if __name__ == "__main__":
    main()
