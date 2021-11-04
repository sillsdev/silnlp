import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

logging.basicConfig()

from ..common.environment import SIL_NLP_ENV
from ..common.utils import get_git_revision_hash
from .config import Config, create_runner, load_config
from .test import test


@dataclass
class SILExperiment:
    name: str
    make_stats: bool = False
    mixed_precision: bool = False
    memory_growth: bool = False
    num_devices: int = 1

    def __post_init__(self):
        self.config: Config = load_config(self.name)
        self.rev_hash = get_git_revision_hash()

    def run(self):
        self.preprocess()
        self.train()
        self.test()

    def preprocess(self):
        self.config.set_seed()
        self.config.preprocess(self.make_stats)
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def train(self):
        os.system("nvidia-smi")
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

        runner = create_runner(self.config, mixed_precision=self.mixed_precision, memory_growth=self.memory_growth)
        runner.save_effective_config(str(self.config.exp_dir / f"effective-config-{self.rev_hash}.yml"), training=True)

        checkpoint_path: Optional[str] = None
        if not (self.config.exp_dir / "run").is_dir() and self.config.has_parent:
            checkpoint_path = str(self.config.exp_dir / "parent")

        print(f"=== Training ({self.name}) ===")
        runner.train(num_devices=self.num_devices, with_eval=True, checkpoint_path=checkpoint_path)
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)
        print("Training completed")

    def test(self):
        test(
            experiment=self.name,
            last=True,
            avg=True,
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
    args = parser.parse_args()

    exp = SILExperiment(
        name=args.experiment,
        make_stats=args.stats,
        mixed_precision=args.mixed_precision,
        memory_growth=args.memory_growth,
        num_devices=args.num_devices,
    )
    exp.run()


if __name__ == "__main__":
    main()
