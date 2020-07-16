import argparse
import logging
import os
from typing import Optional

logging.basicConfig()

from nlp.nmt.config import create_runner, get_root_dir, load_config
from nlp.nmt.utils import get_git_revision_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains a NMT model using OpenNMT-tf")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Enable mixed precision")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to train on")
    args = parser.parse_args()

    rev_hash = get_git_revision_hash()
    print("Git commit:", rev_hash)

    for exp_name in args.experiments:
        config = load_config(exp_name)
        root_dir = get_root_dir(exp_name)
        runner = create_runner(config, mixed_precision=args.mixed_precision, memory_growth=args.memory_growth)
        runner.save_effective_config(os.path.join(root_dir, f"effective-config-{rev_hash}.yml"), training=True)

        checkpoint_path: Optional[str] = None
        data_config: dict = config["data"]
        if not os.path.isdir(os.path.join(root_dir, "run")) and "parent" in data_config:
            checkpoint_path = os.path.join(root_dir, "parent")

        print(f"Training {exp_name}...")
        runner.train(num_devices=args.num_devices, with_eval=True, checkpoint_path=checkpoint_path)
        print("Training completed")


if __name__ == "__main__":
    main()
