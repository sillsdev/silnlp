import argparse
import os
import subprocess
from typing import List

import sacrebleu

from ..common.corpus import load_corpus
from ..common.metrics import compute_meteor_score, compute_ter_score, compute_wer_score
from ..common.utils import get_git_revision_hash, get_mt_root_dir, get_repo_dir
from .config import load_config

SUPPORTED_SCORERS = {"bleu", "chrf3", "meteor", "wer", "ter"}


def get_iso(lang: str) -> str:
    index = lang.find("-")
    return lang[:index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests an SMT model using the Machine library")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        choices=SUPPORTED_SCORERS,
        help=f"List of scorers - {SUPPORTED_SCORERS}",
    )
    parser.add_argument("--force-infer", default=False, action="store_true", help="Force inferencing")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    scorers: List[str] = []
    if args.scorers is None:
        scorers.append("bleu")
    else:
        scorers = list(set(map(lambda s: s.lower(), args.scorers)))
    scorers.sort()

    exp_name = args.experiment
    root_dir = get_mt_root_dir(exp_name)
    config = load_config(exp_name)
    src_iso = get_iso(config["src_lang"])
    trg_iso = get_iso(config["trg_lang"])

    ref_file_path = os.path.join(root_dir, "test.trg.txt")
    predictions_file_path = os.path.join(root_dir, "test.trg-predictions.txt")

    if args.force_infer or not os.path.isfile(predictions_file_path):
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
                "-t",
                config["trg_tokenizer"],
                "-rt",
                config["trg_tokenizer"],
                "-mt",
                config["model"],
                "-l",
            ],
            cwd=get_repo_dir(),
        )

    sys = list(load_corpus(predictions_file_path))
    ref = list(load_corpus(ref_file_path))

    for i in range(len(sys) - 1, 0, -1):
        if ref[i] == "" or sys[i] == "":
            del sys[i]
            del ref[i]

    sent_len = len(sys)
    print("Test results")
    with open(os.path.join(root_dir, "scores.csv"), "w", encoding="utf-8") as scores_file:
        scores_file.write("src_iso,trg_iso,sent_len,scorer,score\n")
        for scorer in scorers:
            if scorer == "bleu":
                bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
                scorer_name = "BLEU"
                score_str = f"{bleu.score:.2f}/{bleu.precisions[0]:.2f}/{bleu.precisions[1]:.2f}"
                score_str += f"/{bleu.precisions[2]:.2f}/{bleu.precisions[3]:.2f}/{bleu.bp:.3f}/{bleu.sys_len:d}"
                score_str += f"/{bleu.ref_len:d}"
            elif scorer == "chrf3":
                chrf3 = sacrebleu.corpus_chrf(sys, [ref], order=6, beta=3, remove_whitespace=True)
                chrf3_score: float = chrf3.score * 100
                scorer_name = "chrF3"
                score_str = f"{chrf3_score:.2f}"
            elif scorer == "meteor":
                meteor_score = compute_meteor_score(trg_iso, sys, [ref])
                if meteor_score is None:
                    continue
                scorer_name = "METEOR"
                score_str = f"{meteor_score:.2f}"
            elif scorer == "wer":
                wer_score = compute_wer_score(sys, ref)
                if wer_score == 0:
                    continue
                scorer_name = "WER"
                score_str = f"{wer_score:.2f}"
            elif scorer == "ter":
                ter_score = compute_ter_score(sys, [ref])
                if ter_score == 0:
                    continue
                scorer_name = "TER"
                score_str = f"{ter_score:.2f}"
            else:
                continue

            score_line = f"{src_iso},{trg_iso},{sent_len:d},{scorer_name},{score_str}"
            scores_file.write(f"{score_line}\n")
            print(score_line)


if __name__ == "__main__":
    main()
