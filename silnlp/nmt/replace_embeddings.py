import argparse
import logging

logging.basicConfig()

from .config import create_runner, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace embeddings in an OpenNMT-tf model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--side", type=str, default="target", choices=["source", "target"], help="Word embedding side")
    parser.add_argument("--embedding", type=str, required=True, help="Input word2vec file")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for new model")
    args = parser.parse_args()

    exp_name = args.experiment
    config = load_config(exp_name)
    runner = create_runner(config)
    runner.replace_embeddings(args.side, args.embedding, args.output, args.vocab)


if __name__ == "__main__":
    main()
