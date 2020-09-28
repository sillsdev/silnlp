import argparse
import os
import subprocess

import sacrebleu

from nlp.common.corpus import load_corpus
from nlp.common.utils import get_git_revision_hash, get_mt_root_dir
from nlp.smt.config import load_config


def get_iso(lang: str) -> str:
    index = lang.find("-")
    return lang[:index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests an SMT model using the Machine library")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_mt_root_dir(exp_name)
    config = load_config(exp_name)
    src_iso = get_iso(config["src_lang"])
    trg_iso = get_iso(config["trg_lang"])

    ref_file_path = os.path.join(root_dir, "test.trg.txt")
    predictions_file_path = os.path.join(root_dir, "test.trg-predictions.txt")

    if not os.path.isfile(predictions_file_path):
        src_file_path = os.path.join(root_dir, "test.src.txt")
        engine_dir = os.path.join(root_dir, f"engine{os.sep}")
        subprocess.run(
            [
                "dotnet",
                "machine",
                "translate",
                engine_dir,
                src_file_path,
                predictions_file_path,
                "-st",
                "latin",
                "-tt",
                "latin",
            ]
        )

    sys = load_corpus(predictions_file_path)
    ref = load_corpus(ref_file_path)

    for i in range(len(sys) - 1, 0, -1):
        if ref[i] == "" or sys[i] == "":
            del sys[i]
            del ref[i]

    bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
    with open(os.path.join(root_dir, "bleu.csv"), "w", encoding="utf-8") as bleu_file:
        bleu_file.write("src_iso,trg_iso,BLEU,1-gram,2-gram,3-gram,4-gram,BP,hyp_len,ref_len,sent_len\n")
        bleu_file.write(
            f"{src_iso},{trg_iso},{bleu.score:.2f},{bleu.precisions[0]:.2f},{bleu.precisions[1]:.2f},"
            f"{bleu.precisions[2]:.2f},{bleu.precisions[3]:.2f},{bleu.bp:.3f},{bleu.sys_len:d},{bleu.ref_len:d},"
            f"{len(sys):d}\n"
        )
    print(
        f"{src_iso} -> {trg_iso}: {bleu.score:.2f} {bleu.precisions[0]:.2f}/{bleu.precisions[1]:.2f}"
        f"/{bleu.precisions[2]:.2f}/{bleu.precisions[3]:.2f}/{bleu.bp:.3f}"
    )


if __name__ == "__main__":
    main()
