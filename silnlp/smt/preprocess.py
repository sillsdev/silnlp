import argparse
from statistics import mean
from typing import Dict, Optional, Set, Tuple

from machine.scripture import VerseRef, get_books

from ..alignment.utils import add_alignment_scores
from ..common.corpus import (
    exclude_books,
    get_scripture_parallel_corpus,
    get_scripture_path,
    include_books,
    load_corpus,
    split_parallel_corpus,
    write_corpus,
)
from ..common.environment import SIL_NLP_ENV
from ..common.utils import get_git_revision_hash, get_mt_exp_dir, set_seed
from .config import load_config


def parse_lang(lang: str) -> Tuple[str, str]:
    index = lang.find("-")
    return lang[:index], lang[index + 1 :]


def get_test_indices(config: dict) -> Optional[Set[int]]:
    exp_name = config.get("use_test_set_from")
    if exp_name is None:
        return None

    exp_dir = get_mt_exp_dir(exp_name)
    vref_path = exp_dir / "test.vref.txt"
    if not vref_path.is_file():
        return None

    vrefs: Dict[str, int] = {}
    for i, vref_str in enumerate(load_corpus(SIL_NLP_ENV.assets_dir / "vref.txt")):
        vrefs[vref_str] = i

    test_indices: Set[int] = set()
    for vref_str in load_corpus(vref_path):
        vref = VerseRef.from_string(vref_str)
        if vref.has_multiple:
            vref.simplify()
        test_indices.add(vrefs[str(vref)])
    return test_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesses text corpora into train and test datasets for SMT training"
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Output corpus statistics")
    args = parser.parse_args()

    get_git_revision_hash()

    exp_name: str = args.experiment
    exp_dir = get_mt_exp_dir(exp_name)
    config = load_config(exp_name)

    set_seed(config["seed"])

    src_iso, src_project = parse_lang(config["src_lang"])
    trg_iso, trg_project = parse_lang(config["trg_lang"])

    src_file_path = get_scripture_path(src_iso, src_project)
    trg_file_path = get_scripture_path(trg_iso, trg_project)

    corpus_books = get_books(config.get("corpus_books", []))
    test_books = get_books(config.get("test_books", []))

    corpus = get_scripture_parallel_corpus(src_file_path, trg_file_path)
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

    test_indices = get_test_indices(config)
    test_size: int = config["test_size"]
    if len(test_books) > 0:
        test = include_books(corpus, test_books)
        if test_size > 0:
            _, test = split_parallel_corpus(test, test_size, test_indices)
    else:
        train, test = split_parallel_corpus(train, test_size, test_indices)

    write_corpus(exp_dir / "train.src.txt", train["source"])
    write_corpus(exp_dir / "train.trg.txt", train["target"])

    write_corpus(exp_dir / "test.vref.txt", (str(vr) for vr in test["vref"]))
    write_corpus(exp_dir / "test.src.txt", test["source"])
    write_corpus(exp_dir / "test.trg.txt", test["target"])


if __name__ == "__main__":
    main()
