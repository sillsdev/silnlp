import argparse
import logging

from ..common.utils import get_git_revision_hash
from .clearml_connection import TAGS_LIST, SILClearML
from .config_utils import load_config

LOGGER = logging.getLogger(__package__ + ".train")

# As of TF 2.7, deterministic mode is slower, so we will disable it for now.
# os.environ["TF_DETERMINISTIC_OPS"] = "True"
# os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "True"


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    parser.add_argument("--disable-mixed-precision", default=False, action="store_true", help="Disable mixed precision")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to train on")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    parser.add_argument(
        "--clearml-tag",
        metavar="tag",
        choices=TAGS_LIST,
        default=None,
        type=str,
        help=f"Tag to add to the ClearML Task - {TAGS_LIST}",
    )
    args = parser.parse_args()
    experiments = args.experiments

    if args.clearml_queue is not None:
        experiments_str = ", ".join(experiments)
        clearml = SILClearML(experiments_str, args.clearml_queue, tag=args.clearml_tag, skip_config=True)

    rev_hash = get_git_revision_hash()

    for exp_name in experiments:
        config = load_config(exp_name)
        config.set_seed()
        model = config.create_model(not args.disable_mixed_precision, args.num_devices)
        model.save_effective_config(config.exp_dir / f"effective-config-{rev_hash}.yml")

        LOGGER.info(f"Training {exp_name}")
        try:
            model.train()
        except RuntimeError as e:
            LOGGER.warning(str(e))
        LOGGER.info(f"Finished training {exp_name}")


if __name__ == "__main__":
    main()
