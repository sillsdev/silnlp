import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..common.environment import SIL_NLP_ENV
from ..common.tf_utils import enable_memory_growth
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import Config, get_mt_exp_dir
from .test import test


@dataclass
class SILExperiment:
    name: str
    make_stats: bool = False
    mixed_precision: bool = False
    num_devices: int = 1
    clearml_queue: Optional[str] = None

    def __post_init__(self):
        self.clearml = SILClearML(self.name, self.clearml_queue)
        self.name: str = self.clearml.name
        self.config: Config = self.clearml.config
        self.rev_hash = get_git_revision_hash()
        self.config.set_seed()

    def run(self):
        self.preprocess()
        self.train()
        self.test()

    def preprocess(self):
        SIL_NLP_ENV.copy_experiment_from_bucket(self.name, extensions=(".yml"))
        # Do some basic checks before starting the experiment
        exp_dir = Path(get_mt_exp_dir(self.name))
        if not exp_dir.exists():
            raise RuntimeError(f"ERROR: Experiment folder {exp_dir} does not exist.")
        config_file = Path(exp_dir, "config.yml")
        if not config_file.exists():
            raise RuntimeError(f"ERROR: Config file does not exist in experiment folder {exp_dir}.")
        self.config.preprocess(self.make_stats)
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def train(self):
        os.system("nvidia-smi")
        SIL_NLP_ENV.copy_experiment_from_bucket(
            self.name, extensions=(".txt", ".vocab", ".model", ".yml", ".csv"), copy_run=True
        )

        model = self.config.create_model(self.mixed_precision, self.num_devices)
        model.save_effective_config(self.config.exp_dir / f"effective-config-{self.rev_hash}.yml")

        print(f"=== Training ({self.name}) ===")
        model.train()
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)
        print("Training completed")

    def test(self):
        SIL_NLP_ENV.copy_experiment_from_bucket(
            self.name, extensions=(".txt", ".vocab", ".model", ".yml", ".csv", ".json")
        )
        test(
            experiment=self.name,
            last=True,
            best=True,
            scorers={"bleu", "sentencebleu", "chrf3", "wer", "ter"},
        )
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment - preprocesses, train and test")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Output corpus statistics")
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Enable mixed precision")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to train on")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run it locally and register it with ClearML.",
    )
    args = parser.parse_args()

    if args.memory_growth:
        enable_memory_growth()

    exp = SILExperiment(
        name=args.experiment,
        make_stats=args.stats,
        mixed_precision=args.mixed_precision,
        num_devices=args.num_devices,
        clearml_queue=args.clearml_queue,
    )
    exp.run()


if __name__ == "__main__":
    main()
