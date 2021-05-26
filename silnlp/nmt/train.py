import argparse
import logging
import os
from typing import Optional


import tensorflow as tf

from ..common.utils import get_git_revision_hash
from .config import create_runner, load_config

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Enable mixed precision")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to train on")
    parser.add_argument(
        "--eager-execution",
        default=False,
        action="store_true",
        help="Enable TensorFlow eager execution.",
    )
    args = parser.parse_args()

    rev_hash = get_git_revision_hash()

    if args.eager_execution:
        tf.config.run_functions_eagerly(True)

    for exp_name in args.experiments:
        config = load_config(exp_name)
        config.set_seed()
        runner = create_runner(config, mixed_precision=args.mixed_precision, memory_growth=args.memory_growth)
        runner.save_effective_config(str(config.exp_dir / f"effective-config-{rev_hash}.yml"), training=True)

        checkpoint_path: Optional[str] = None
        if not (config.exp_dir / "run").is_dir() and config.has_parent:
            checkpoint_path = str(config.exp_dir / "parent")

        print(f"=== Training ({exp_name}) ===")
        runner.train(num_devices=args.num_devices, with_eval=True, checkpoint_path=checkpoint_path)
        print("Training completed")


if __name__ == "__main__":
    main()
