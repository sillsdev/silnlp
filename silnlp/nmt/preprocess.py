import argparse
import logging

from ..common.utils import get_git_revision_hash
from .config_utils import load_config

LOGGER = logging.getLogger((__package__ or "") + ".preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesses the parallel corpus for an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Compute tokenization statistics")
    parser.add_argument(
        "--force-align", default=False, action="store_true", help="Force recalculation of all alignment scores"
    )
    args = parser.parse_args()

    get_git_revision_hash()

    exp_name = args.experiment
    config = load_config(exp_name)

    config.set_seed()
    config.preprocess(args.stats, args.force_align)


if __name__ == "__main__":
    main()
