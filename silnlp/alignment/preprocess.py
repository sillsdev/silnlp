import argparse
import json
import logging
import os
import unicodedata
from typing import Iterable, List, Tuple

logging.basicConfig(level=logging.INFO)

from nltk.translate import Alignment

from ..common.corpus import write_corpus
from ..common.environment import ALIGN_GOLD_STANDARDS_DIR
from ..common.stemmer import Stemmer
from ..common.utils import get_align_root_dir, set_seed
from ..common.verse_ref import VerseRef
from .config import get_all_book_paths, get_stemmer, load_config
from .lexicon import Lexicon


class ParallelSegment:
    def __init__(self, ref: str, source: List[str], target: List[str], alignment: Alignment) -> None:
        self.ref = ref
        self.source = source
        self.target = target
        self.alignment = alignment


def get_ref(verse: dict) -> str:
    id = str(verse["manuscript"]["words"][0]["id"])
    return id[:-4]


def get_segment(segInfo: dict, use_lemma: bool = False) -> List[str]:
    words: Iterable[str] = (w["lemma" if use_lemma else "text"] for w in segInfo["words"])
    return [unicodedata.normalize("NFC", w.lower().replace(" ", "~")) for w in words]


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


def write_datasets(root_dir: str, src_stemmer: Stemmer, trg_stemmer: Stemmer, corpus: List[ParallelSegment]) -> None:
    train_refs_path = os.path.join(root_dir, "refs.txt")
    write_corpus(train_refs_path, map(lambda s: s.ref, corpus))

    train_src_path = os.path.join(root_dir, "src.txt")
    write_corpus(train_src_path, map(lambda s: " ".join(s.source), corpus))

    train_trg_path = os.path.join(root_dir, "trg.txt")
    write_corpus(train_trg_path, map(lambda s: " ".join(src_stemmer.stem(s.target)), corpus))
    write_corpus(train_trg_path, map(lambda s: " ".join(trg_stemmer.stem(s.target)), corpus))

    test_alignments_path = os.path.join(root_dir, "alignments.gold.txt")
    write_corpus(test_alignments_path, map(lambda s: str(s.alignment), corpus))


def is_in_book(segment: ParallelSegment, book: str) -> bool:
    ref = VerseRef.from_bbbcccvvv(int(segment.ref))
    return ref.book == book


def add_alignment(lexicon: Lexicon, source: List[str], target: List[str], alignment: Alignment) -> None:
    for src_index, trg_index in alignment:
        src_word = source[src_index]
        trg_word = target[trg_index]
        lexicon.increment(src_word, trg_word)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesses Clear gold standard alignments")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    root_dir = get_align_root_dir(args.experiment)
    config = load_config(args.experiment)

    set_seed(config["seed"])

    corpus_name: str = config["corpus"]

    corpus_path = os.path.join(ALIGN_GOLD_STANDARDS_DIR, corpus_name + ".alignment.json")
    verses: List[dict]
    with open(corpus_path, "r", encoding="utf-8") as f:
        verses = json.load(f)

    use_src_lemma: bool = config["use_src_lemma"]
    corpus: List[ParallelSegment] = []
    lexicon = Lexicon()
    for verse in verses:
        ref_str = get_ref(verse)
        source = get_segment(verse["manuscript"], use_lemma=use_src_lemma)
        target = get_segment(verse["translation"])
        alignment = get_alignment(verse)
        corpus.append(ParallelSegment(ref_str, source, target, alignment))
        add_alignment(lexicon, source, target, get_alignment(verse, primary_links_only=True))
    lexicon.normalize()

    src_stemmer = get_stemmer(config["src_stemmer"])
    src_stemmer.train(map(lambda s: s.source, corpus))

    trg_stemmer = get_stemmer(config["trg_stemmer"])
    trg_stemmer.train(map(lambda s: s.target, corpus))

    if config["by_book"]:
        for book, book_root_dir in get_all_book_paths(root_dir):
            book_corpus = list(filter(lambda s: is_in_book(s, book), corpus))
            if len(book_corpus) > 0:
                os.makedirs(book_root_dir, exist_ok=True)
                write_datasets(book_root_dir, src_stemmer, trg_stemmer, book_corpus)
    else:
        write_datasets(root_dir, src_stemmer, trg_stemmer, corpus)
        lexicon.write(os.path.join(root_dir, "lexicon.gold.txt"))


if __name__ == "__main__":
    main()
