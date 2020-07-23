import argparse
import logging
import os
import random
from glob import glob
from typing import IO, Dict, List, Set, Tuple
from itertools import combinations

logging.basicConfig()

import sacrebleu

from nlp.common.utils import get_git_revision_hash
from nlp.nmt.config import get_root_dir, load_config, parse_langs
from nlp.nmt.utils import decode_sp


class TestResults:
    def __init__(self, src_iso: str, trg_iso: str, bleu: sacrebleu.BLEU, sent_len: int, projects: List[str]) -> None:
        self.src_iso = src_iso
        self.trg_iso = trg_iso
        self.bleu = bleu
        self.sent_len = sent_len
        self.num_refs = len(projects)
        self.refs = "_".join(sorted(projects))

    def write(self, file: IO) -> None:
        file.write(
            f"{self.src_iso},{self.trg_iso},{self.num_refs},{self.refs},{self.bleu.score:.2f},"
            f"{self.bleu.precisions[0]:.2f},{self.bleu.precisions[1]:.2f},{self.bleu.precisions[2]:.2f},"
            f"{self.bleu.precisions[3]:.2f},{self.bleu.bp:.3f},{self.bleu.sys_len:d},"
            f"{self.bleu.ref_len:d},{self.sent_len:d}\n"
        )


def parse_ref_file_path(ref_file_path: str) -> Tuple[str, str]:
    parts = os.path.basename(ref_file_path).split(".")
    return parts[2], parts[3]


def is_ref_project(ref_projects: List[str], ref_file_path: str) -> bool:
    _, trg_project = parse_ref_file_path(ref_file_path)
    return trg_project in ref_projects


def is_train_project(train_projects: Dict[str, Set[str]], ref_file_path: str) -> bool:
    trg_iso, trg_project = parse_ref_file_path(ref_file_path)
    projects = train_projects.get(trg_iso)
    return projects is None or trg_project in projects


def load_test_data(
    src_file_path: str,
    pred_file_path: str,
    ref_files_path: str,
    output_file_path: str,
    default_trg_iso: str,
    ref_projects: List[str],
    train_projects: Dict[str, Set[str]],
) -> Dict[str, Tuple[List[str], List[List[str]]]]:
    dataset: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    with open(src_file_path, "r", encoding="utf-8") as src_file, open(
        pred_file_path, "r", encoding="utf-8"
    ) as pred_file, open(output_file_path, "w", encoding="utf-8") as out_file:
        ref_file_paths = glob(ref_files_path)
        select_rand_ref_line = False
        if len(ref_file_paths) > 1:
            filtered: List[str] = list(filter(lambda p: is_ref_project(ref_projects, p), ref_file_paths))
            if len(filtered) == 0:
                # no refs specified, so randomly select verses from all available train refs to build one ref
                select_rand_ref_line = True
                ref_file_paths = list(filter(lambda p: is_train_project(train_projects, p), ref_file_paths))
            else:
                # use specified refs only
                ref_file_paths = filtered
        ref_files: List[IO] = []
        try:
            for ref_file_path in ref_file_paths:
                ref_files.append(open(ref_file_path, "r", encoding="utf-8"))
            for lines in zip(src_file, pred_file, *ref_files):
                src_line = lines[0].strip()
                pred_line = lines[1].strip()
                detok_pred_line = decode_sp(pred_line)
                iso = default_trg_iso
                if src_line.startswith("<2"):
                    index = src_line.index(">")
                    val = src_line[2:index]
                    if val != "qaa":
                        iso = val
                if iso not in dataset:
                    dataset[iso] = ([], [])
                sys, refs = dataset[iso]
                sys.append(detok_pred_line)
                if select_rand_ref_line:
                    ref_index = random.randint(0, len(ref_files) - 1)
                    ref_line = lines[ref_index + 2].strip()
                    if len(refs) == 0:
                        refs.append([])
                    refs[0].append(ref_line)
                else:
                    for ref_index in range(len(ref_files)):
                        ref_line = lines[ref_index + 2].strip()
                        if len(refs) == ref_index:
                            refs.append([])
                        refs[ref_index].append(ref_line)
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
    parser.add_argument("--ref-projects", nargs="*", metavar="project", default=[], help="Reference projects")
    parser.add_argument("--num-refs", type=int, default=0, help="Max. number of reference projects")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)
    config = load_config(exp_name)
    data_config: dict = config["data"]
    src_langs, _, _ = parse_langs(data_config["src_langs"])
    trg_langs, trg_train_projects, _ = parse_langs(data_config["trg_langs"])

    random.seed(data_config["seed"])

    ref_project_lists: List[List[str]] = []
    if args.num_refs == 0:
        ref_project_lists = [args.ref_projects]
    else:
        if args.num_refs > len(args.ref_projects):
            args.num_refs = len(args.ref_projects)
        for i in range(0, args.num_refs):
            comb = combinations(args.ref_projects, i + 1)
            for c in comb:
                ref_project_lists.append(list(c))

    scores: List[TestResults] = []
    for ref_projects in ref_project_lists:
        features_paths: List[str] = []
        predictions_paths: List[str] = []
        refs_paths: List[str] = []
        predictions_detok_paths: List[str] = []
        for src_iso in sorted(src_langs):
            prefix = "test" if len(src_langs) == 1 else f"test.{src_iso}"
            src_features_path = os.path.join(root_dir, f"{prefix}.src.txt")
            if os.path.isfile(src_features_path):
                # all target data is stored in a single file
                features_paths.append(src_features_path)
                predictions_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.txt"))
                refs_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok*.txt"))
                predictions_detok_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.detok.txt"))
            else:
                # target data is split into separate files
                for trg_iso in sorted(trg_langs):
                    prefix = f"test.{src_iso}.{trg_iso}"
                    features_paths.append(os.path.join(root_dir, f"{prefix}.src.txt"))
                    predictions_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.txt"))
                    refs_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok*.txt"))
                    predictions_detok_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.detok.txt"))

            #            print("Inferencing...")
            step: int
            step = 0
        #        if args.best:
        #            best_model_path, _ = get_best_model_dir(config)
        #            if os.path.isfile(os.path.join(best_model_path, "ckpt.index")):
        #                step = runner.infer_multiple(
        #                    features_paths, predictions_paths, checkpoint_path=os.path.join(best_model_path, "ckpt")
        #                )
        #            else:
        #                step = runner.saved_model_infer_multiple(features_paths, predictions_paths)
        #        else:
        #            step = runner.infer_multiple(features_paths, predictions_paths, checkpoint_path=checkpoint_path)

        print(f"Scoring... {ref_projects}")
        #        ref_projects: Set[str] = set(args.ref_projects)
        default_src_iso = next(iter(src_langs))
        default_trg_iso = next(iter(trg_langs))
        overall_sys: List[str] = []
        overall_refs: List[List[str]] = []
        for features_path, predictions_path, refs_path, predictions_detok_path in zip(
            features_paths, predictions_paths, refs_paths, predictions_detok_paths
        ):
            features_filename = os.path.basename(features_path)
            src_iso = default_src_iso
            if features_filename != "test.src.txt":
                src_iso = features_filename.split(".")[1]
            dataset = load_test_data(
                features_path,
                predictions_path,
                refs_path,
                predictions_detok_path,
                default_trg_iso,
                ref_projects,
                trg_train_projects,
            )

            for trg_iso, (sys, refs) in dataset.items():
                start_index = len(overall_sys)
                overall_sys.extend(sys)
                for i, ref in enumerate(refs):
                    if i == len(overall_refs):
                        overall_refs.append([""] * start_index)
                    overall_refs[i].extend(ref)
                # ensure that all refs are the same length as the sys
                for overall_ref in filter(lambda r: len(r) < len(overall_sys), overall_refs):
                    overall_ref.extend([""] * (len(overall_sys) - len(overall_ref)))
                bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
                scores.append(TestResults(src_iso, trg_iso, bleu, len(sys), ref_projects))

        #            os.replace(predictions_path, f"{predictions_path}.{step}")
        #            os.replace(predictions_detok_path, f"{predictions_detok_path}.{step}")

        if len(src_langs) > 1 or len(trg_langs) > 1:
            bleu = sacrebleu.corpus_bleu(overall_sys, overall_refs, lowercase=True)
            scores.append(TestResults("ALL", "ALL", bleu, len(overall_sys), overall_refs))

    print(f"Test results ({len(args.ref_projects)} reference(s), {args.num_refs} combinations)")
    bleu_file_root = f"bleu-{step}"
    if len(args.ref_projects) > 0:
        ref_projects_suffix = "_".join(sorted(args.ref_projects))
        bleu_file_root += f"-{ref_projects_suffix}"
    with open(os.path.join(root_dir, f"{bleu_file_root}.csv"), "w", encoding="utf-8") as bleu_file:
        bleu_file.write(
            "src_iso,trg_iso,num_refs,references,BLEU,1-gram,2-gram,3-gram,4-gram,BP,hyp_len,ref_len,sent_len\n"
        )
        for results in scores:
            results.write(bleu_file)
            print(f"{results.src_iso} -> {results.trg_iso}:", results.bleu)


if __name__ == "__main__":
    main()
