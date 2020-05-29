import argparse
import logging
import os
from glob import glob
from typing import IO, Dict, List, Tuple

logging.basicConfig()

import sacrebleu
import sentencepiece as sp
import tensorflow as tf

from nlp.nmt.config import create_runner, get_root_dir, load_config, parse_langs


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


def get_ref_index(ref_file_path: str) -> int:
    root = os.path.splitext(os.path.basename(ref_file_path))[0]
    ref_index_str = os.path.splitext(root)[1].lstrip(".")
    return int(ref_index_str) if ref_index_str.isnumeric() else 0


def load_test_data(
    model_file: str,
    src_file_path: str,
    pred_file_path: str,
    ref_files_path: str,
    output_file_path: str,
    default_trg_iso: str,
    num_refs: int,
) -> Dict[str, Tuple[List[str], List[List[str]]]]:
    spp = sp.SentencePieceProcessor()
    spp.load(model_file)

    dataset: Dict[str, Tuple[List[str], List[str]]] = dict()
    with open(src_file_path, "r", encoding="utf-8") as src_file, open(
        pred_file_path, "r", encoding="utf-8"
    ) as pred_file, open(output_file_path, "w", encoding="utf-8") as out_file:
        ref_file_paths = sorted(glob(ref_files_path), key=get_ref_index)
        ref_files: List[IO] = []
        try:
            for ref_file_path in ref_file_paths:
                ref_files.append(open(ref_file_path, "r", encoding="utf-8"))
                if num_refs > 0 and len(ref_files) == num_refs:
                    break
            for lines in zip(src_file, pred_file, *ref_files):
                src_line = lines[0].strip()
                pred_line = lines[1].strip()
                detok_pred_line = spp.decode_pieces(pred_line.split(" "))
                iso = default_trg_iso
                if src_line.startswith("<2"):
                    index = src_line.index(">")
                    val = src_line[2:index]
                    if val != "qaa":
                        iso = val
                if iso not in dataset:
                    dataset[iso] = ([], [None] * len(ref_files))
                dataset[iso][0].append(detok_pred_line)
                for ref_index in range(len(ref_files)):
                    ref_line = lines[ref_index + 2].strip()
                    if dataset[iso][1][ref_index] is None:
                        dataset[iso][1][ref_index] = []
                    dataset[iso][1][ref_index].append(ref_line)
                out_file.write(detok_pred_line + "\n")
        finally:
            for ref_file in ref_files:
                ref_file.close()

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a NMT model using OpenNMT-tf")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use")
    parser.add_argument("--best", default=False, action="store_true", help="Test best evaluated model")
    parser.add_argument("--num-refs", type=int, default=1, help="Number of references to test")
    args = parser.parse_args()

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)
    config = load_config(exp_name)
    data_config: dict = config.get("data", {})
    src_langs, _ = parse_langs(data_config.get("src_langs", []))
    trg_langs, _ = parse_langs(data_config.get("trg_langs", []))
    runner = create_runner(config, memory_growth=args.memory_growth)

    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(config["model_dir"], f"ckpt-{args.checkpoint}")

    features_paths: List[str] = []
    predictions_paths: List[str] = []
    refs_paths: List[str] = []
    predictions_detok_paths: List[str] = []
    for src_iso in src_langs:
        prefix = "test" if len(src_langs) == 1 else f"test.{src_iso}"
        features_paths.append(os.path.join(root_dir, f"{prefix}.src.txt"))
        predictions_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.txt"))
        refs_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok*.txt"))
        predictions_detok_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.detok.txt"))

    print("Inferencing...")
    step: int
    if args.best:
        step = runner.saved_model_infer_multiple(features_paths, predictions_paths)
    else:
        step = runner.infer_multiple(features_paths, predictions_paths, checkpoint_path=checkpoint_path)

    print("Scoring...")
    default_trg_iso = next(iter(trg_langs))
    scores: List[TestResults] = []
    overall_sys: List[str] = []
    overall_refs: List[List[str]] = []
    model_file_path = os.path.join(root_dir, "sp.model" if data_config.get("share_vocab", True) else "trg-sp.model")
    for src_iso, features_path, predictions_path, refs_path, predictions_detok_path in zip(
        src_langs, features_paths, predictions_paths, refs_paths, predictions_detok_paths
    ):
        dataset = load_test_data(
            model_file_path,
            features_path,
            predictions_path,
            refs_path,
            predictions_detok_path,
            default_trg_iso,
            args.num_refs,
        )

        for trg_iso, data in dataset.items():
            sys, refs = data
            overall_sys.extend(sys)
            for i, ref in enumerate(refs):
                if i == len(overall_refs):
                    overall_refs.append([])
                overall_refs[i].extend(ref)
            bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
            scores.append(TestResults(src_iso, trg_iso, bleu, len(sys)))

        os.replace(predictions_path, f"{predictions_path}.{step}")
        os.replace(predictions_detok_path, f"{predictions_detok_path}.{step}")

    if len(src_langs) > 1 or len(trg_langs) > 1:
        bleu = sacrebleu.corpus_bleu(overall_sys, overall_refs, lowercase=True)
        scores.append(TestResults("ALL", "ALL", bleu, len(overall_sys)))

    print(f"Test results ({len(overall_refs)} reference(s))")
    bleu_file_root = f"bleu-{step}"
    if len(overall_refs) > 1:
        bleu_file_root += f"-{len(overall_refs)}"
    with open(os.path.join(root_dir, f"{bleu_file_root}.csv"), "w", encoding="utf-8") as bleu_file:
        bleu_file.write("src_iso,trg_iso,BLEU,BP,hyp_len,ref_len,sent_len\n")
        for results in scores:
            results.write(bleu_file)
            print(f"{results.src_iso} -> {results.trg_iso}:", results.bleu)


if __name__ == "__main__":
    main()
