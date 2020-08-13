import argparse
import logging
import os
import random
from glob import glob
from typing import IO, Dict, List, Set, Tuple

logging.basicConfig()

import sacrebleu
import yaml

from nlp.common.utils import get_git_revision_hash
from nlp.nmt.config import create_runner, get_root_dir, load_config, parse_langs
from nlp.nmt.utils import decode_sp, get_best_model_dir


class PairScore:
    def __init__(self, src_iso: str, trg_iso: str, bleu: sacrebleu.BLEU, sent_len: int, projects: Set[str]) -> None:
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


def parse_ref_file_path(ref_file_path: str, default_trg_iso: str) -> Tuple[str, str]:
    parts = os.path.basename(ref_file_path).split(".")
    if len(parts) == 5:
        return default_trg_iso, parts[3]
    return parts[2], parts[5]


def is_ref_project(ref_projects: Set[str], ref_file_path: str) -> bool:
    _, trg_project = parse_ref_file_path(ref_file_path, "qaa")
    return trg_project in ref_projects


def is_train_project(train_projects: Dict[str, Set[str]], ref_file_path: str, default_trg_iso: str) -> bool:
    trg_iso, trg_project = parse_ref_file_path(ref_file_path, default_trg_iso)
    projects = train_projects[trg_iso]
    return trg_project in projects


def load_test_data(
    src_file_path: str,
    pred_file_path: str,
    ref_files_path: str,
    output_file_path: str,
    default_trg_iso: str,
    ref_projects: Set[str],
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
                ref_file_paths = list(
                    filter(lambda p: is_train_project(train_projects, p, default_trg_iso), ref_file_paths)
                )
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


def test_checkpoint(
    root_dir: str,
    config: dict,
    src_langs: Set[str],
    trg_langs: Set[str],
    trg_train_projects: Dict[str, Set[str]],
    force_infer: bool,
    memory_growth: bool,
    ref_projects: Set[str],
    checkpoint_path: str,
    step: int,
) -> List[PairScore]:
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
            predictions_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.txt.{step}"))
            refs_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok*.txt"))
            predictions_detok_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.detok.txt.{step}"))
        else:
            # target data is split into separate files
            for trg_iso in sorted(trg_langs):
                prefix = f"test.{src_iso}.{trg_iso}"
                features_paths.append(os.path.join(root_dir, f"{prefix}.src.txt"))
                predictions_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.txt.{step}"))
                refs_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok*.txt"))
                predictions_detok_paths.append(os.path.join(root_dir, f"{prefix}.trg-predictions.detok.txt.{step}"))

    if force_infer or any(not os.path.isfile(f) for f in predictions_detok_paths):
        runner = create_runner(config, memory_growth=memory_growth)
        print(f"Inferencing checkpoint {step}...")
        if os.path.basename(checkpoint_path) == "saved_model.pb":
            runner.saved_model_infer_multiple(features_paths, predictions_paths)
        else:
            runner.infer_multiple(features_paths, predictions_paths, checkpoint_path=checkpoint_path)

    data_config: dict = config["data"]
    print(f"Scoring checkpoint {step}...")
    default_src_iso = next(iter(src_langs))
    default_trg_iso = next(iter(trg_langs))
    scores: List[PairScore] = []
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
            bleu = sacrebleu.corpus_bleu(
                sys, refs, lowercase=True, tokenize=data_config.get("sacrebleu_tokenize", "13a")
            )
            scores.append(PairScore(src_iso, trg_iso, bleu, len(sys), ref_projects))

    if len(src_langs) > 1 or len(trg_langs) > 1:
        bleu = sacrebleu.corpus_bleu(overall_sys, overall_refs, lowercase=True)
        scores.append(PairScore("ALL", "ALL", bleu, len(overall_sys), ref_projects))

    bleu_file_root = f"bleu-{step}"
    if len(ref_projects) > 0:
        ref_projects_suffix = "_".join(sorted(ref_projects))
        bleu_file_root += f"-{ref_projects_suffix}"
    with open(os.path.join(root_dir, f"{bleu_file_root}.csv"), "w", encoding="utf-8") as bleu_file:
        bleu_file.write(
            "src_iso,trg_iso,num_refs,references,BLEU,1-gram,2-gram,3-gram,4-gram,BP,hyp_len,ref_len,sent_len\n"
        )
        for results in scores:
            results.write(bleu_file)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a NMT model using OpenNMT-tf")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use")
    parser.add_argument("--last", default=False, action="store_true", help="Test last checkpoint")
    parser.add_argument("--best", default=False, action="store_true", help="Test best evaluated checkpoint")
    parser.add_argument("--ref-projects", nargs="*", metavar="project", default=[], help="Reference projects")
    parser.add_argument("--force-infer", default=False, action="store_true", help="Force inferencing")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)
    config = load_config(exp_name)
    model_dir: str = config["model_dir"]
    data_config: dict = config["data"]
    src_langs, _, _ = parse_langs(data_config["src_langs"])
    trg_langs, trg_train_projects, _ = parse_langs(data_config["trg_langs"])
    ref_projects: Set[str] = set(args.ref_projects)

    random.seed(data_config["seed"])

    best_model_path, best_step = get_best_model_dir(model_dir)
    results: Dict[int, List[PairScore]] = {}
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(model_dir, f"ckpt-{args.checkpoint}")
        step = int(args.checkpoint)
        results[step] = test_checkpoint(
            root_dir,
            config,
            src_langs,
            trg_langs,
            trg_train_projects,
            args.force_infer,
            args.memory_growth,
            ref_projects,
            checkpoint_path,
            step,
        )

    if args.best:
        step = best_step
        checkpoint_path = os.path.join(best_model_path, "ckpt")
        if not os.path.isfile(checkpoint_path + ".index"):
            checkpoint_path = os.path.join(model_dir, "saved_model.pb")

        if step not in results:
            results[step] = test_checkpoint(
                root_dir,
                config,
                src_langs,
                trg_langs,
                trg_train_projects,
                args.force_infer,
                args.memory_growth,
                ref_projects,
                checkpoint_path,
                step,
            )

    if args.last or (not args.best and args.checkpoint is None):
        with open(os.path.join(model_dir, "checkpoint"), "r", encoding="utf-8") as file:
            checkpoint_config = yaml.safe_load(file)
            checkpoint_prefix: str = checkpoint_config["model_checkpoint_path"]
            parts = os.path.basename(checkpoint_prefix).split("-")
            checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
            step = int(parts[-1])

        if step not in results:
            results[step] = test_checkpoint(
                root_dir,
                config,
                src_langs,
                trg_langs,
                trg_train_projects,
                args.force_infer,
                args.memory_growth,
                ref_projects,
                checkpoint_path,
                step,
            )

    for step in sorted(results.keys()):
        num_refs = results[step][0].num_refs
        if num_refs == 0:
            num_refs = 1
        checkpoint_str = "best " if step == best_step else ""
        checkpoint_str += f"checkpoint {step}"
        print(f"Test results for {checkpoint_str} ({num_refs} reference(s))")
        for score in results[step]:
            print(
                f"{score.src_iso} -> {score.trg_iso}: {score.bleu.score:.2f} {score.bleu.precisions[0]:.2f}"
                f"/{score.bleu.precisions[1]:.2f}/{score.bleu.precisions[2]:.2f}/{score.bleu.precisions[3]:.2f}"
                f"/{score.bleu.bp:.3f}"
            )


if __name__ == "__main__":
    main()
