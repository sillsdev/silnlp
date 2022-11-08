import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, List, Optional, TextIO, Tuple

import pandas as pd
from machine.corpora import ParallelTextCorpus, ParallelTextRow, TextFileTextCorpus
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef
from machine.tokenization import LatinWordTokenizer, WhitespaceTokenizer
from nltk.translate import Alignment

from ..common.corpus import get_mt_corpus_path, get_scripture_parallel_corpus
from ..common.environment import SIL_NLP_ENV
from ..common.stemmer import Stemmer
from ..common.utils import set_seed
from .config import get_all_book_paths, get_stemmer, load_config
from .lexicon import Lexicon
from .utils import get_experiment_dirs, get_experiment_name

LOGGER = logging.getLogger(__package__ + ".preprocess")


def get_vref(verse: dict) -> VerseRef:
    id = str(verse["manuscript"]["words"][0]["id"])
    return VerseRef.from_bbbcccvvv(int(id[:-4]), ORIGINAL_VERSIFICATION)


def get_segment(segInfo: dict, use_lemma: bool = False) -> List[str]:
    words: Iterable[str] = (w["lemma" if use_lemma else "text"] for w in segInfo["words"])
    return [w.replace(" ", "~") for w in words]


def get_alignment(verse: dict, primary_links_only: bool = False) -> Alignment:
    links: List[List[List[int]]] = verse["links"]
    pairs: List[Tuple[int, int]] = []
    for link in links:
        src_indices = link[0]
        trg_indices = link[1]
        if primary_links_only:
            if len(src_indices) > 0 and len(trg_indices) > 0:
                pairs.append((src_indices[0], trg_indices[0]))
        else:
            for src_index in src_indices:
                for trg_index in trg_indices:
                    pairs.append((src_index, trg_index))
    return Alignment(pairs)


def write_datasets(
    exp_dir: Path,
    src_stemmer: Stemmer,
    trg_stemmer: Stemmer,
    rows: Iterable[ParallelTextRow],
    is_scripture: bool,
    has_gold_alignments: bool,
) -> None:
    with ExitStack() as stack:
        train_refs_file: Optional[TextIO] = None
        if is_scripture:
            train_refs_path = exp_dir / "refs.txt"
            train_refs_file = stack.enter_context(train_refs_path.open("w", encoding="utf-8", newline="\n"))
        train_src_path = exp_dir / "src.txt"
        train_src_file = stack.enter_context(train_src_path.open("w", encoding="utf-8", newline="\n"))
        train_trg_path = exp_dir / "trg.txt"
        train_trg_file = stack.enter_context(train_trg_path.open("w", encoding="utf-8", newline="\n"))
        gold_alignments_file: Optional[TextIO] = None
        if has_gold_alignments:
            gold_alignments_path = exp_dir / "alignments.gold.txt"
            gold_alignments_file = stack.enter_context(gold_alignments_path.open("w", encoding="utf-8", newline="\n"))

        for row in rows:
            if train_refs_file is not None:
                train_refs_file.write(str(row.ref) + "\n")
            train_src_file.write(" ".join(src_stemmer.stem(row.source_segment)) + "\n")
            train_trg_file.write(" ".join(trg_stemmer.stem(row.target_segment)) + "\n")
            if gold_alignments_file is not None:
                gold_alignments_file.write(
                    " ".join(f"{wp.source_index}-{wp.target_index}" for wp in row.aligned_word_pairs) + "\n"
                )


def is_in_book(row: ParallelTextRow, book: str) -> bool:
    ref: VerseRef = row.ref
    return ref.book == book


def add_alignment(lexicon: Lexicon, source: List[str], target: List[str], alignment: Alignment) -> None:
    for src_index, trg_index in alignment:
        src_word = source[src_index]
        trg_word = target[trg_index]
        lexicon.increment(src_word, trg_word)


def load_gold_alignment(corpus_name: str, use_src_lemma: bool) -> Tuple[ParallelTextCorpus, Lexicon]:
    corpus_path = SIL_NLP_ENV.align_gold_dir / (corpus_name + ".alignment.json")
    verses: List[dict]
    with corpus_path.open("r", encoding="utf-8") as f:
        verses = json.load(f)

    vrefs: List[VerseRef] = []
    src_verses: List[str] = []
    trg_verses: List[str] = []
    alignments: List[str] = []
    lexicon = Lexicon()
    for verse in verses:
        vref = get_vref(verse)
        source = get_segment(verse["manuscript"], use_lemma=use_src_lemma)
        target = get_segment(verse["translation"])
        alignment = get_alignment(verse)
        vrefs.append(vref)
        src_verses.append(" ".join(source))
        trg_verses.append(" ".join(target))
        alignments.append(str(alignment))
        add_alignment(lexicon, source, target, get_alignment(verse, primary_links_only=True))
    lexicon.normalize()
    df = pd.DataFrame({"ref": vrefs, "source": src_verses, "target": trg_verses, "alignment": alignments})
    return (
        ParallelTextCorpus.from_pandas(df).tokenize(WhitespaceTokenizer()),
        lexicon,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesses Clear gold standard alignments")
    parser.add_argument("experiments", type=str, help="Experiment pattern")
    args = parser.parse_args()

    for exp_dir in get_experiment_dirs(args.experiments):
        exp_name = get_experiment_name(exp_dir)
        print(f"=== Preprocessing ({exp_name}) ===")
        config = load_config(exp_dir)

        set_seed(config["seed"])

        corpus_name: Optional[str] = config.get("corpus", config.get("gold"))
        lexicon: Optional[Lexicon]
        has_gold_alignments = False
        if corpus_name is not None:
            corpus, lexicon = load_gold_alignment(corpus_name, config["use_src_lemma"])
            is_scripture = True
            has_gold_alignments = True
        else:
            src_file_path = get_mt_corpus_path(config["src"])
            trg_file_path = get_mt_corpus_path(config["trg"])
            if (
                src_file_path.parent == SIL_NLP_ENV.mt_scripture_dir
                or trg_file_path.parent == SIL_NLP_ENV.mt_scripture_dir
            ):
                corpus = ParallelTextCorpus.from_pandas(
                    get_scripture_parallel_corpus(src_file_path, trg_file_path), ref_column="vref"
                )
                is_scripture = True
            else:
                src_corpus = TextFileTextCorpus(src_file_path)
                trg_corpus = TextFileTextCorpus(trg_file_path)
                corpus = src_corpus.align_rows(trg_corpus)
                is_scripture = False
            corpus = corpus.tokenize(LatinWordTokenizer())
            lexicon = None

        src_casing: str = config["src_casing"]
        src_casing = src_casing.lower()
        trg_casing: str = config["trg_casing"]
        trg_casing = trg_casing.lower()

        if src_casing == "lower" and trg_casing == "lower":
            corpus = corpus.lowercase()
        elif src_casing == "lower":
            corpus = corpus.lowercase_source()
        elif trg_casing == "lower":
            corpus = corpus.lowercase_target()

        src_normalize: bool = config["src_normalize"]
        trg_normalize: bool = config["trg_normalize"]

        if src_normalize and trg_normalize:
            corpus = corpus.nfc_normalize()
        elif src_normalize:
            corpus = corpus.nfc_normalize_source()
        elif trg_normalize:
            corpus = corpus.nfc_normalize_target()

        src_stemmer = get_stemmer(config["src_stemmer"])
        src_stemmer.train(row.source_segment for row in corpus)

        trg_stemmer = get_stemmer(config["trg_stemmer"])
        trg_stemmer.train(row.target_segment for row in corpus)

        if is_scripture and config["by_book"]:
            for book, book_exp_dir in get_all_book_paths(exp_dir):
                with corpus.get_rows() as rows:
                    book_corpus = [row for row in rows if is_in_book(row, book)]
                if len(book_corpus) > 0:
                    book_exp_dir.mkdir(exist_ok=True)
                    write_datasets(
                        book_exp_dir,
                        src_stemmer,
                        trg_stemmer,
                        book_corpus,
                        is_scripture=True,
                        has_gold_alignments=has_gold_alignments,
                    )
        else:
            with corpus.get_rows() as rows:
                write_datasets(exp_dir, src_stemmer, trg_stemmer, rows, is_scripture, has_gold_alignments)
            if lexicon is not None:
                lexicon.write(exp_dir / "lexicon.gold.txt")


if __name__ == "__main__":
    main()
