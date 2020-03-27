import argparse
import os
import logging

import opennmt
import tensorflow as tf

from nlp.common.environment import paratextPreprocessedDir
from nlp.nmt.noise import WordDropout


_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL = {
    logging.CRITICAL: 3,
    logging.ERROR: 2,
    logging.WARNING: 1,
    logging.INFO: 0,
    logging.DEBUG: 0,
    logging.NOTSET: 0,
}


def _set_log_level(log_level):
    tf.get_logger().setLevel(log_level)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL[log_level])


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains a NMT model using OpenNMT-tf")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--add-noise", action="store_true", help="Add artificial noise")
    parser.add_argument("--single-target", action="store_true", help="The model has a single target language")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logs verbosity.",
    )
    args = parser.parse_args()

    _set_log_level(getattr(logging, args.log_level))

    name = args.task
    root_dir = os.path.join(paratextPreprocessedDir, "tests", name)

    config = {
        "model_dir": os.path.join(root_dir, "run"),
        "data": {
            "train_features_file": os.path.join(root_dir, "train.src.txt"),
            "train_labels_file": os.path.join(root_dir, "train.trg.txt"),
            "eval_features_file": os.path.join(root_dir, "val.src.txt"),
            "eval_labels_file": os.path.join(root_dir, "val.trg.txt"),
            "source_vocabulary": os.path.join(root_dir, "onmt.vocab"),
            "target_vocabulary": os.path.join(root_dir, "onmt.vocab"),
        },
        "train": {"average_last_checkpoints": 0},
        "eval": {
            "external_evaluators": "bleu",
            "steps": 1000,
            "early_stopping": {"metric": "bleu", "min_improvement": 0.2, "steps": 4},
        },
    }

    model = opennmt.models.TransformerBase()

    add_noise = args.add_noise
    single_target = args.single_target
    if add_noise:
        source_noiser = opennmt.data.WordNoiser(subword_token="▁", is_spacer=True)
        source_noiser.add(WordDropout(0.1, skip_first_word=not single_target))
        model.features_inputter.set_noise(source_noiser, probability=1.0)

        target_noiser = opennmt.data.WordNoiser(subword_token="▁", is_spacer=True)
        target_noiser.add(WordDropout(0.1))
        target_noiser.add(opennmt.data.WordPermutation(3))
        model.labels_inputter.set_noise(target_noiser, probability=1.0)

    runner = opennmt.Runner(model, config, auto_config=True, mixed_precision=True)
    runner.train(num_devices=1, with_eval=True)


if __name__ == "__main__":
    main()
