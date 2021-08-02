import argparse
import logging

LOGGER = logging.getLogger(__package__ + ".preprocess")

from .config import get_git_revision_hash, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesses the parallel corpus for an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Output corpus statistics")
    args = parser.parse_args()

    get_git_revision_hash()

    exp_name = args.experiment
    config = load_config(exp_name)

    config.set_seed()
    config.preprocess(args.stats)


if __name__ == "__main__":
    main()
