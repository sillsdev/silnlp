import argparse
import logging
import os

logging.basicConfig()

from nlp.nmt.config import create_runner, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Average checkpoints of an OpenNMT-tf model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--max-count", type=int, default=3, help="The maximum number of checkpoints to average")
    args = parser.parse_args()

    exp_name = args.experiment
    config = load_config(exp_name)
    runner = create_runner(config)
    output = os.path.join(config["model_dir"], "avg")
    runner.average_checkpoints(output, args.max_count)


if __name__ == "__main__":
    main()