import argparse
import logging

from ..common.environment import SIL_NLP_ENV
from ..common.tf_utils import enable_eager_execution, enable_memory_growth
from ..common.utils import get_git_revision_hash
from .config_utils import load_config

LOGGER = logging.getLogger(__package__ + ".train")

# As of TF 2.7, deterministic mode is slower, so we will disable it for now.
# os.environ["TF_DETERMINISTIC_OPS"] = "True"
# os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "True"


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an NMT model")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    parser.add_argument("--disable-mixed-precision", default=False, action="store_true", help="Disable mixed precision")
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
        enable_eager_execution()

    if args.memory_growth:
        enable_memory_growth()

    for exp_name in args.experiments:
        SIL_NLP_ENV.copy_experiment_from_bucket(exp_name)
        config = load_config(exp_name)
        config.set_seed()
        model = config.create_model(not args.disable_mixed_precision, args.num_devices)
        model.save_effective_config(config.exp_dir / f"effective-config-{rev_hash}.yml")

        LOGGER.info(f"Training {exp_name}")
        try:
            model.train()
            SIL_NLP_ENV.copy_experiment_to_bucket(exp_name)
        except RuntimeError as e:
            LOGGER.warning(str(e))
        LOGGER.info(f"Finished training {exp_name}")


if __name__ == "__main__":
    main()
