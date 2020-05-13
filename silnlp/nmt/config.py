import argparse
import logging
import os
from typing import Dict, Iterable, List, Set, Tuple

logging.basicConfig()

import opennmt
import tensorflow as tf
import yaml

from nlp.common.environment import paratextPreprocessedDir
from nlp.nmt.noise import WordDropout
from nlp.nmt.runner import RunnerEx

_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL: Dict[int, int] = {
    logging.CRITICAL: 3,
    logging.ERROR: 2,
    logging.WARNING: 1,
    logging.INFO: 0,
    logging.DEBUG: 0,
    logging.NOTSET: 0,
}


def set_log_level(log_level: int) -> None:
    tf.get_logger().setLevel(log_level)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL[log_level])


def get_root_dir(exp_name: str) -> str:
    return os.path.join(paratextPreprocessedDir, "tests", exp_name)


def load_config(exp_name: str) -> dict:
    root_dir = get_root_dir(exp_name)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {
        "model_dir": os.path.join(root_dir, "run"),
        "data": {
            "train_features_file": os.path.join(root_dir, "train.src.txt"),
            "train_labels_file": os.path.join(root_dir, "train.trg.txt"),
            "eval_features_file": os.path.join(root_dir, "val.src.txt"),
            "eval_labels_file": os.path.join(root_dir, "val.trg.txt"),
        },
        "train": {
            "average_last_checkpoints": 0,
            "maximum_features_length": 150,
            "maximum_labels_length": 150,
            "keep_checkpoint_max": 3,
        },
        "eval": {
            "external_evaluators": "bleu",
            "steps": 1000,
            "early_stopping": {"metric": "bleu", "min_improvement": 0.2, "steps": 4},
            "export_on_best": "bleu",
            "max_exports_to_keep": 1,
        },
        "params": {"length_penalty": 0.2},
    }

    config = opennmt.load_config([config_path], config)
    data_config: dict = config["data"]
    if data_config.get("share_vocab", True):
        data_config["source_vocabulary"] = os.path.join(root_dir, "onmt.vocab")
        data_config["target_vocabulary"] = os.path.join(root_dir, "onmt.vocab")
    else:
        data_config["source_vocabulary"] = os.path.join(root_dir, "src-onmt.vocab")
        data_config["target_vocabulary"] = os.path.join(root_dir, "trg-onmt.vocab")
    return config


def create_runner(
    config: dict, mixed_precision: bool = False, log_level: int = logging.INFO, memory_growth: bool = False
) -> RunnerEx:
    set_log_level(log_level)

    if memory_growth:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, enable=True)

    data_config: dict = config.get("data", {})
    train_config: dict = config.get("train", {})

    model = opennmt.models.TransformerBase()

    add_noise: bool = train_config.get("add_noise", False)
    if add_noise:
        single_target = len(data_config.get("trg_langs", [])) == 1
        source_noiser = opennmt.data.WordNoiser(subword_token="▁", is_spacer=True)
        source_noiser.add(WordDropout(0.1, skip_first_word=not single_target))
        model.features_inputter.set_noise(source_noiser, probability=1.0)

        target_noiser = opennmt.data.WordNoiser(subword_token="▁", is_spacer=True)
        target_noiser.add(WordDropout(0.1))
        target_noiser.add(opennmt.data.WordPermutation(3))
        model.labels_inputter.set_noise(target_noiser, probability=1.0)

    return RunnerEx(model, config, auto_config=True, mixed_precision=mixed_precision)


def parse_langs(langs: Iterable[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    isos: Set[str] = set()
    lang_projects: Dict[str, Set[str]] = {}
    for lang in langs:
        parts = lang.split("-")
        isos.add(parts[0])
        if len(parts) == 2:
            lang_projects[parts[0]] = set(parts[1].split(","))
    return isos, lang_projects


def main() -> None:
    parser = argparse.ArgumentParser(description="Creates a NMT experiment config file")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--src-langs", nargs="*", metavar="lang", default=[], help="Source language")
    parser.add_argument("--trg-langs", nargs="*", metavar="lang", default=[], help="Target language")
    parser.add_argument("--vocab-size", type=int, help="Shared vocabulary size")
    parser.add_argument("--src-vocab-size", type=int, help="Source vocabulary size")
    parser.add_argument("--trg-vocab-size", type=int, help="Target vocabulary size")
    parser.add_argument("--parent", type=str, help="Parent experiment name")
    parser.add_argument("--mirror", default=False, action="store_true", help="Mirror train and validation data sets")
    parser.add_argument("--force", default=False, action="store_true", help="Overwrite existing config file")
    args = parser.parse_args()

    root_dir = get_root_dir(args.experiment)
    config_path = os.path.join(root_dir, "config.yml")
    if os.path.isfile(config_path) and not args.force:
        print('The experiment config file already exists. Use "--force" if you want to overwrite the existing config.')
        return

    os.makedirs(root_dir, exist_ok=True)

    config: dict = {"data": {"src_langs": args.src_langs, "trg_langs": args.trg_langs}}
    data_config: dict = config["data"]
    if args.parent is not None:
        data_config["parent"] = args.parent
        parent_config = load_config(args.parent)
        parent_data_config: dict = parent_config["data"]
        if "share_vocab" in parent_data_config:
            data_config["share_vocab"] = parent_data_config["share_vocab"]
        if "vocab_size" in parent_data_config:
            data_config["vocab_size"] = parent_data_config["vocab_size"]
        if "src_vocab_size" in parent_data_config:
            data_config["src_vocab_size"] = parent_data_config["src_vocab_size"]
        if "trg_vocab_size" in parent_data_config:
            data_config["trg_vocab_size"] = parent_data_config["trg_vocab_size"]
    if args.vocab_size is not None:
        data_config["vocab_size"] = args.vocab_size
    elif args.src_vocab_size is not None or args.trg_vocab_size is not None:
        data_config["share_vocab"] = False
        if args.src_vocab_size is not None:
            data_config["src_vocab_size"] = args.src_vocab_size
        elif "vocab_size" in data_config:
            data_config["src_vocab_size"] = data_config["vocab_size"]
            del data_config["vocab_size"]
        if args.trg_vocab_size is not None:
            data_config["trg_vocab_size"] = args.trg_vocab_size
        elif "vocab_size" in data_config:
            data_config["trg_vocab_size"] = data_config["vocab_size"]
            del data_config["vocab_size"]
    if args.mirror:
        data_config["mirror"] = True
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file)
    print("Config file created")


if __name__ == "__main__":
    main()
