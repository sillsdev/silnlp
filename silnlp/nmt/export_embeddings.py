import argparse
import logging
import os

logging.basicConfig()

from nlp.nmt.config import create_runner, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Exports embeddings from an OpenNMT-tf model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--side", type=str, default="target", choices=["source", "target"], help="Word embedding side")
    parser.add_argument("--output", type=str, required=True, help="Output word2vec file")
    args = parser.parse_args()

    exp_name = args.experiment
    config = load_config(exp_name)
    runner = create_runner(config)
    runner.export_embeddings(args.side, args.output)


if __name__ == "__main__":
    main()
