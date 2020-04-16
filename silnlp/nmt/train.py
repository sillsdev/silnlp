import argparse

from nlp.nmt.config import create_runner, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains a NMT model using OpenNMT-tf")
    parser.add_argument("task", help="Task name")
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Enable mixed precision")
    args = parser.parse_args()

    task_name = args.task
    config = load_config(task_name)
    runner = create_runner(config, mixed_precision=args.mixed_precision)

    print("Training...")
    runner.train(num_devices=1, with_eval=True)

    print("Training completed")


if __name__ == "__main__":
    main()
