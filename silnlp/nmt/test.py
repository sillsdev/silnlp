import argparse
import os
from typing import List

import sacrebleu
import sentencepiece as sp

from nlp.nmt.config import create_runner, get_root_dir, load_config


def detokenize(model_file: str, input_file: str, output_file: str) -> List[str]:
    spp = sp.SentencePieceProcessor()
    spp.load(model_file)

    sentences: List[str] = list()
    with open(input_file, "r", encoding="utf-8") as in_file:
        with open(output_file, "w", encoding="utf-8") as out_file:
            for line in in_file:
                line = line.strip()
                sentence = spp.decode_pieces(line.split(" "))
                sentences.append(sentence)
                out_file.write(sentence + "\n")

    return sentences


def load_reference(input_file: str) -> List[str]:
    sentences: List[str] = list()
    with open(input_file, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            sentences.append(line)

    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a NMT model using OpenNMT-tf")
    parser.add_argument("task", help="Task name")
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Enable mixed precision")
    args = parser.parse_args()

    task_name = args.task
    root_dir = get_root_dir(task_name)
    config = load_config(task_name)
    runner = create_runner(config, mixed_precision=args.mixed_precision)

    print("Generating predictions...")
    features_file = os.path.join(root_dir, "test.src.txt")
    predictions_file = os.path.join(root_dir, "test-predictions.trg.txt")
    runner.infer(features_file, predictions_file=predictions_file)

    print("Detokenizing predictions...")
    model_file = os.path.join(root_dir, "trg-sp.model")
    predictions_detok_file = os.path.join(root_dir, "test-predictions.trg.detok.txt")
    sys = detokenize(model_file, predictions_file, predictions_detok_file)

    print("Calculating BLEU...")
    ref_file = os.path.join(root_dir, "test.trg.detok.txt")
    ref = load_reference(ref_file)
    bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
    print("BLEU:", "{0:.1f}".format(bleu.score))


if __name__ == "__main__":
    main()
