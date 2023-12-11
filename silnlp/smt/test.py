import argparse
from contextlib import ExitStack
from typing import List

import sacrebleu
from machine.corpora import TextFileTextCorpus, TextRow
from machine.translation import MAX_SEGMENT_LENGTH
from machine.translation.thot import ThotSmtModel
from tqdm import tqdm

from ..common.corpus import load_corpus
from ..common.environment import SIL_NLP_ENV
from ..common.metrics import compute_meteor_score, compute_wer_score
from ..common.utils import get_git_revision_hash, get_mt_exp_dir
from .config import create_word_detokenizer, create_word_tokenizer, get_thot_word_alignment_type, load_config

SUPPORTED_SCORERS = {"bleu", "spbleu", "chrf3", "meteor", "wer", "ter"}


def get_iso(lang: str) -> str:
    index = lang.find("-")
    return lang[:index]


def is_valid(row: TextRow) -> bool:
    return not row.is_empty and len(row.segment) <= MAX_SEGMENT_LENGTH


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

    get_git_revision_hash()

    scorers: List[str] = []
    if args.scorers is None:
        scorers.append("bleu")
    else:
        scorers = list(set(map(lambda s: s.lower(), args.scorers)))
    scorers.sort()

    exp_name = args.experiment
    exp_dir = get_mt_exp_dir(exp_name)
    SIL_NLP_ENV.copy_experiment_from_bucket(exp_name)
    config = load_config(exp_name)
    src_iso = get_iso(config["src_lang"])
    trg_iso = get_iso(config["trg_lang"])

    ref_file_path = exp_dir / "test.trg.txt"
    predictions_file_path = exp_dir / "test.trg-predictions.txt"

    if args.force_infer or not predictions_file_path.is_file():
        src_file_path = exp_dir / "test.src.txt"
        engine_config_file_path = exp_dir / "engine" / "smt.cfg"

        src_corpus = (
            TextFileTextCorpus(src_file_path)
            .tokenize(create_word_tokenizer(config["src_tokenizer"]))
            .unescape_spaces()
            .lowercase()
        )

        detokenizer = create_word_detokenizer(config["trg_tokenizer"])

        with ExitStack() as stack:
            model = stack.enter_context(
                ThotSmtModel(get_thot_word_alignment_type(config["model"]), engine_config_file_path)
            )
            out_file = stack.enter_context(predictions_file_path.open("w", encoding="utf-8", newline="\n"))
            count = src_corpus.count()
            rows = stack.enter_context(src_corpus.get_rows())

            for row in tqdm(rows, total=count, bar_format="{l_bar}{bar:40}{r_bar}"):
                if is_valid(row):
                    result = model.translate(row.segment)
                    translation = detokenizer.detokenize(result.target_tokens)
                    out_file.write(translation + "\n")
                else:
                    out_file.write("\n")

    sys = list(load_corpus(predictions_file_path))
    ref = list(load_corpus(ref_file_path))

    for i in range(len(sys) - 1, 0, -1):
        if ref[i] == "" or sys[i] == "":
            del sys[i]
            del ref[i]

    sent_len = len(sys)
    print("Test results")
    with (exp_dir / "scores.csv").open("w", encoding="utf-8") as scores_file:
        scores_file.write("src_iso,trg_iso,sent_len,scorer,score\n")
        for scorer in scorers:
            if scorer == "bleu":
                bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
                scorer_name = "BLEU"
                score_str = f"{bleu.score:.2f}/{bleu.precisions[0]:.2f}/{bleu.precisions[1]:.2f}"
                score_str += f"/{bleu.precisions[2]:.2f}/{bleu.precisions[3]:.2f}/{bleu.bp:.3f}/{bleu.sys_len:d}"
                score_str += f"/{bleu.ref_len:d}"
            elif scorer == "chrf3":
                chrf3 = sacrebleu.corpus_chrf(sys, [ref], char_order=6, beta=3, remove_whitespace=True)
                chrf3_score: float = chrf3.score
                scorer_name = "chrF3"
                score_str = f"{chrf3_score:.2f}"
            elif scorer == "meteor":
                meteor_score = compute_meteor_score(trg_iso, sys, [ref])
                if meteor_score is None:
                    continue
                scorer_name = "METEOR"
                score_str = f"{meteor_score:.2f}"
            elif scorer == "wer":
                wer_score = compute_wer_score(sys, [ref])
                if wer_score == 0:
                    continue
                scorer_name = "WER"
                score_str = f"{wer_score/100:.2f}"
            elif scorer == "ter":
                ter_score = sacrebleu.corpus_ter(sys, [ref])
                if ter_score.score >= 0:
                    scorer_name = "TER"
                    score_str = f"{ter_score.score:.2f}"
            elif scorer == "spbleu":
                spbleu_score = sacrebleu.corpus_bleu(
                    sys,
                    [ref],
                    lowercase=True,
                    tokenize="flores200",
                )
                scorer_name = "spBLEU"
                score_str = f"{spbleu_score.score:.2f}"
            else:
                continue

            score_line = f"{src_iso},{trg_iso},{sent_len:d},{scorer_name},{score_str}"
            scores_file.write(f"{score_line}\n")
            print(score_line)


if __name__ == "__main__":
    main()
