import argparse
import os
from typing import Dict, List, Tuple, IO

import sacrebleu
import sentencepiece as sp

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
                iso = src_line[2:index]
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
    parser.add_argument("task", help="Task name")
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Enable mixed precision")
    args = parser.parse_args()

    task_name = args.task
    root_dir = get_root_dir(task_name)
    config = load_config(task_name)
    data_config: dict = config.get("data", {})
    src_langs = data_config.get("src_langs", [])
    trg_langs = data_config.get("trg_langs", [])
    runner = create_runner(config, mixed_precision=args.mixed_precision)

    scores: List[TestResults] = list()
    overall_sys: List[str] = list()
    overall_ref: List[str] = list()
    model_file_path = os.path.join(root_dir, "sp.model" if data_config.get("share_vocab", True) else "trg-sp.model")
    for src_iso in src_langs:
        print(f'Testing source "{src_iso}" data set...')
        prefix = "test" if len(src_langs) == 1 else f"test.{src_iso}"
        features_file_path = os.path.join(root_dir, f"{prefix}.src.txt")
        predictions_file_path = os.path.join(root_dir, f"{prefix}.trg-predictions.txt")
        runner.infer(features_file_path, predictions_file=predictions_file_path)

        ref_file_path = os.path.join(root_dir, f"{prefix}.trg.detok.txt")
        predictions_detok_file_path = os.path.join(root_dir, f"{prefix}.trg-predictions.detok.txt")
        dataset = load_test_data(
            model_file_path,
            features_file_path,
            predictions_file_path,
            ref_file_path,
            predictions_detok_file_path,
            trg_langs[0],
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
