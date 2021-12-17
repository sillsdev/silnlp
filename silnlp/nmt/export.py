import argparse

from opennmt.utils.exporters import make_exporter

from .config import create_runner, get_checkpoint_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Exports an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--checkpoint", type=str, default="last", help="Checkpoint to use (last, best, avg, or checkpoint #)"
    )
    parser.add_argument("--format", type=str, default="saved_model", help="The output directory")
    args = parser.parse_args()

    exp_name = args.experiment
    config = load_config(exp_name)
    runner = create_runner(config)
    checkpoint_path, step = get_checkpoint_path(config.model_dir, args.checkpoint)
    output = config.exp_dir / "export" / str(step)

    runner.export(str(output), checkpoint_path=str(checkpoint_path), exporter=make_exporter(args.format))


if __name__ == "__main__":
    main()
