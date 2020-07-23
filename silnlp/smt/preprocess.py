import argparse
import os

from typing import Tuple

from nlp.common.corpus import get_corpus_path, get_parallel_corpus, split_parallel_corpus, write_corpus
from nlp.common.utils import get_git_revision_hash, get_root_dir
from nlp.smt.config import load_config


def parse_lang(lang: str) -> Tuple[str, str]:
    index = lang.find("-")
    return lang[:index], lang[index + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesses text corpora into train and test datasets for SMT training"
    )
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)
    config = load_config(exp_name)

    src_iso, src_project = parse_lang(config["src_lang"])
    trg_iso, trg_project = parse_lang(config["trg_lang"])

    src_file_path = get_corpus_path(src_iso, src_project)
    trg_file_path = get_corpus_path(trg_iso, trg_project)

    train = get_parallel_corpus(src_file_path, trg_file_path)

    test_size: int = config["test_size"]
    train, test = split_parallel_corpus(train, test_size)

    write_corpus(os.path.join(root_dir, "train.src.txt"), train["source"])
    write_corpus(os.path.join(root_dir, "train.trg.txt"), train["target"])

    write_corpus(os.path.join(root_dir, "test.src.txt"), test["source"])
    write_corpus(os.path.join(root_dir, "test.trg.txt"), test["target"])


if __name__ == "__main__":
    main()
