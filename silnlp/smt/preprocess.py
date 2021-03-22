import argparse
from statistics import mean
from typing import Tuple

from ..alignment.utils import add_alignment_scores
from ..common.canon import get_books
from ..common.corpus import (
    exclude_books,
    get_scripture_path,
    get_scripture_parallel_corpus,
    include_books,
    split_parallel_corpus,
    write_corpus,
)
from ..common.environment import MT_SCRIPTURE_DIR
from ..common.utils import get_git_revision_hash, get_mt_exp_dir, set_seed
from .config import load_config


def parse_lang(lang: str) -> Tuple[str, str]:
    index = lang.find("-")
    return lang[:index], lang[index + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesses text corpora into train and test datasets for SMT training"
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Output corpus statistics")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name: str = args.experiment
    exp_dir = get_mt_exp_dir(exp_name)
    config = load_config(exp_name)

    set_seed(config["seed"])

    src_iso, src_project = parse_lang(config["src_lang"])
    trg_iso, trg_project = parse_lang(config["trg_lang"])

    vref_file_path = MT_SCRIPTURE_DIR / "vref.txt"
    src_file_path = get_scripture_path(src_iso, src_project)
    trg_file_path = get_scripture_path(trg_iso, trg_project)

    corpus_books = get_books(config.get("corpus_books", []))
    test_books = get_books(config.get("test_books", []))

    corpus = get_scripture_parallel_corpus(vref_file_path, src_file_path, trg_file_path)
    if len(corpus_books) > 0:
        train = include_books(corpus, corpus_books)
        if len(corpus_books.intersection(test_books)) > 0:
            train = exclude_books(train, test_books)
    elif len(test_books) > 0:
        train = exclude_books(corpus, test_books)
    else:
        train = corpus

    if args.stats:
        corpus_len = len(train)
        add_alignment_scores(train)
        alignment_score = mean(train["score"])
        print(f"{src_project} -> {trg_project} stats")
        print(f"- count: {corpus_len}")
        print(f"- alignment: {alignment_score:.4f}")

    test_size: int = config["test_size"]
    if len(test_books) > 0:
        test = include_books(corpus, test_books)
        if test_size > 0:
            _, test = split_parallel_corpus(test, test_size)
    else:
        train, test = split_parallel_corpus(train, test_size)

    write_corpus(exp_dir / "train.src.txt", train["source"])
    write_corpus(exp_dir / "train.trg.txt", train["target"])

    write_corpus(exp_dir / "test.vref.txt", map(lambda vr: str(vr), test["vref"]))
    write_corpus(exp_dir / "test.src.txt", test["source"])
    write_corpus(exp_dir / "test.trg.txt", test["target"])


if __name__ == "__main__":
    main()
