import argparse
import logging
import os
from typing import Optional

logging.basicConfig()

import tensorflow as tf

from nlp.nmt.config import create_runner, get_root_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains a NMT model using OpenNMT-tf")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Enable mixed precision")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    args = parser.parse_args()

    exp_name = args.experiment
    config = load_config(exp_name)
    root_dir = get_root_dir(exp_name)
    runner = create_runner(config, mixed_precision=args.mixed_precision, memory_growth=args.memory_growth)

    checkpoint_path: Optional[str] = None
    data_config: dict = config.get("data", {})
    if "parent" in data_config:
        checkpoint_path = os.path.join(root_dir, "parent")

    print("Training...")
    runner.train(num_devices=1, with_eval=True, checkpoint_path=checkpoint_path)

    print("Training completed")


if __name__ == "__main__":
    main()
