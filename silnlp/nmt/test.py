import argparse
import logging
import os
from typing import IO, Dict, List, Tuple

logging.basicConfig()

import sacrebleu
import sentencepiece as sp
import tensorflow as tf

from nlp.nmt.config import create_runner, get_root_dir, load_config


class TestResults:
    def __init__(self, src_iso: str, trg_iso: str, bleu: sacrebleu.BLEU, sent_len: int) -> None:
        self.src_iso = src_iso
        self.trg_iso = trg_iso
        self.bleu = bleu
        self.sent_len = sent_len

    def write(self, file: IO) -> None:
        file.write(
            f"{self.src_iso},{self.trg_iso},{self.bleu.score:.2f},{self.bleu.bp:.3f},{self.bleu.sys_len:d},"
            f"{self.bleu.ref_len:d},{self.sent_len:d}\n"
        )


def load_test_data(
    model_file: str,
    src_file_path: str,
    pred_file_path: str,
    ref_file_path: str,
    output_file_path: str,
    default_trg_iso: str,
) -> Dict[str, Tuple[List[str], List[str]]]:
    spp = sp.SentencePieceProcessor()
    spp.load(model_file)

    dataset: Dict[str, Tuple[List[str], List[str]]] = dict()
    with open(src_file_path, "r", encoding="utf-8") as src_file, open(
        pred_file_path, "r", encoding="utf-8"
    ) as pred_file, open(ref_file_path, "r", encoding="utf-8") as ref_file, open(
        output_file_path, "w", encoding="utf-8"
    ) as out_file:
        for src_line, pred_line, ref_line in zip(src_file, pred_file, ref_file):
            src_line = src_line.strip()
            pred_line = pred_line.strip()
            ref_line = ref_line.strip()
            detok_pred_line = spp.decode_pieces(pred_line.split(" "))
            iso = default_trg_iso
            if src_line.startswith("<2"):
                index = src_line.index(">")
                val = src_line[2:index]
                if val != "qaa":
                    iso = val
            if iso not in dataset:
                dataset[iso] = (list(), list())
            dataset[iso][0].append(detok_pred_line)
            dataset[iso][1].append(ref_line)
            out_file.write(detok_pred_line + "\n")

    return dataset


def load_reference(input_file: str) -> List[str]:
    sentences: List[str] = list()
    with open(input_file, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            sentences.append(line)

    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a NMT model using OpenNMT-tf")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use")
    args = parser.parse_args()

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)
    config = load_config(exp_name)
    data_config: dict = config.get("data", {})
    src_langs = data_config.get("src_langs", [])
    trg_langs = data_config.get("trg_langs", [])
    runner = create_runner(config, memory_growth=args.memory_growth)

    use_saved_model = False
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.lower()
        if checkpoint == "best":
            use_saved_model = True
        else:
            checkpoint_path = os.path.join(config["model_dir"], f"ckpt-{args.checkpoint}")

    features_paths: List[str] = []
    predictions_paths: List[str] = []
    ref_paths: List[str] = []
    predictions_detok_paths: List[str] = []
    for src_iso in src_langs:
        prefix = "test" if len(src_langs) == 1 else f"test.{src_iso}"
        features_paths.append(os.path.join(root_dir, f"{prefix}.src.txt"))
        predictions_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.txt"))
        ref_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok.txt"))
        predictions_detok_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.detok.txt"))

    print("Inferencing...")
    if use_saved_model:
        runner.saved_model_infer_multiple(features_paths, predictions_paths)
    else:
        runner.infer_multiple(features_paths, predictions_paths, checkpoint_path=checkpoint_path)

    print("Scoring...")
    scores: List[TestResults] = list()
    overall_sys: List[str] = list()
    overall_ref: List[str] = list()
    model_file_path = os.path.join(root_dir, "sp.model" if data_config.get("share_vocab", True) else "trg-sp.model")
    for src_iso, features_path, predictions_path, ref_path, predictions_detok_path in zip(
        src_langs, features_paths, predictions_paths, ref_paths, predictions_detok_paths
    ):
        dataset = load_test_data(
            model_file_path, features_path, predictions_path, ref_path, predictions_detok_path, trg_langs[0]
        )

        for trg_iso, data in dataset.items():
            sys = data[0]
            ref = data[1]
            overall_sys.extend(sys)
            overall_ref.extend(ref)
            bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
            scores.append(TestResults(src_iso, trg_iso, bleu, len(sys)))

    if len(src_langs) > 1 or len(trg_langs) > 1:
        bleu = sacrebleu.corpus_bleu(overall_sys, [overall_ref], lowercase=True)
        scores.append(TestResults("ALL", "ALL", bleu, len(overall_sys)))

    print("Test results")
    with open(os.path.join(root_dir, "bleu.csv"), "w", encoding="utf-8") as bleu_file:
        bleu_file.write("src_iso,trg_iso,BLEU,BP,hyp_len,ref_len,sent_len\n")
        for results in scores:
            results.write(bleu_file)
            print(f"{results.src_iso} -> {results.trg_iso}:", results.bleu)


if __name__ == "__main__":
    main()
