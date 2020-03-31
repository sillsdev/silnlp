import argparse
import logging
import os
from typing import Dict, List

import opennmt
import tensorflow as tf
import yaml

from nlp.common.environment import paratextPreprocessedDir
from nlp.nmt.noise import WordDropout

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


def get_root_dir(task_name: str) -> str:
    return os.path.join(paratextPreprocessedDir, "tests", task_name)


def load_config(task_name: str) -> dict:
    root_dir = get_root_dir(task_name)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {
        "model_dir": os.path.join(root_dir, "run"),
        "data": {
            "train_features_file": os.path.join(root_dir, "train.src.txt"),
            "train_labels_file": os.path.join(root_dir, "train.trg.txt"),
            "eval_features_file": os.path.join(root_dir, "val.src.txt"),
            "eval_labels_file": os.path.join(root_dir, "val.trg.txt"),
        },
        "train": {"average_last_checkpoints": 0},
        "eval": {
            "external_evaluators": "bleu",
            "steps": 1000,
            "early_stopping": {"metric": "bleu", "min_improvement": 0.2, "steps": 4},
        },
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


def create_runner(config: dict, mixed_precision: bool = False, log_level: int = logging.INFO) -> opennmt.Runner:
    set_log_level(log_level)

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

    return opennmt.Runner(model, config, auto_config=True, mixed_precision=mixed_precision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Creates a NMT task config file")
    parser.add_argument("task", help="Task name")
    parser.add_argument("--src-langs", nargs="*", metavar="lang", default=[], help="Source language")
    parser.add_argument("--trg-langs", nargs="*", metavar="lang", default=[], help="Target language")
    parser.add_argument("--vocab-size", type=int, help="Shared vocabulary size")
    parser.add_argument("--src-vocab-size", type=int, help="Source vocabulary size")
    parser.add_argument("--trg-vocab-size", type=int, help="Target vocabulary size")
    args = parser.parse_args()

    root_dir = get_root_dir(args.task)
    os.makedirs(root_dir, exist_ok=True)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {"data": {"src_langs": args.src_langs, "trg_langs": args.trg_langs}}
    data_config: dict = config["data"]
    if args.vocab_size is not None:
        data_config["vocab_size"] = args.vocab_size
    elif args.src_vocab_size is not None and args.trg_vocab_size is not None:
        data_config["share_vocab"] = False
        data_config["src_vocab_size"] = args.src_vocab_size
        data_config["trg_vocab_size"] = args.trg_vocab_size
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file)
    print("Created config file")


if __name__ == "__main__":
    main()
