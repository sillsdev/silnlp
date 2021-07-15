import argparse
import json
import logging
import unicodedata
from pathlib import Path
from typing import Iterable, List, Tuple

logging.basicConfig(level=logging.INFO)

from nltk.translate import Alignment

from ..common.corpus import write_corpus
from ..common.environment import SNE
from ..common.stemmer import Stemmer
from ..common.utils import set_seed
from ..common.verse_ref import VerseRef
from .config import get_all_book_paths, get_stemmer, load_config
from .lexicon import Lexicon
from .utils import get_experiment_dirs, get_experiment_name


class ParallelSegment:
    def __init__(self, ref: str, source: List[str], target: List[str], alignment: Alignment) -> None:
        self.ref = ref
        self.source = source
        self.target = target
        self.alignment = alignment


def get_ref(verse: dict) -> str:
    id = str(verse["manuscript"]["words"][0]["id"])
    return id[:-4]


def get_segment(segInfo: dict, casing: str, normalize: bool, use_lemma: bool = False) -> List[str]:
    words: Iterable[str] = (w["lemma" if use_lemma else "text"] for w in segInfo["words"])
    return [transform_token(w, casing, normalize) for w in words]


def transform_token(token: str, casing: str, normalize: bool) -> str:
    token = token.replace(" ", "~")
    casing = casing.lower()
    if casing == "lower":
        token = token.lower()
    if normalize:
        token = unicodedata.normalize("NFC", token)
    return token


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


def write_datasets(exp_dir: Path, src_stemmer: Stemmer, trg_stemmer: Stemmer, corpus: List[ParallelSegment]) -> None:
    train_refs_path = exp_dir / "refs.txt"
    write_corpus(train_refs_path, map(lambda s: s.ref, corpus))

    train_src_path = exp_dir / "src.txt"
    write_corpus(train_src_path, map(lambda s: " ".join(s.source), corpus))

    train_trg_path = exp_dir / "trg.txt"
    write_corpus(train_trg_path, map(lambda s: " ".join(src_stemmer.stem(s.target)), corpus))
    write_corpus(train_trg_path, map(lambda s: " ".join(trg_stemmer.stem(s.target)), corpus))

    test_alignments_path = exp_dir / "alignments.gold.txt"
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
    parser.add_argument("experiments", type=str, help="Experiment pattern")
    args = parser.parse_args()

    for exp_dir in get_experiment_dirs(args.experiments):
        exp_name = get_experiment_name(exp_dir)
        print(f"=== Preprocessing ({exp_name}) ===")
        config = load_config(exp_dir)

        set_seed(config["seed"])

        corpus_name: str = config["corpus"]

        corpus_path = SNE._ALIGN_GOLD_DIR / (corpus_name + ".alignment.json")
        verses: List[dict]
        with open(corpus_path, "r", encoding="utf-8") as f:
            verses = json.load(f)

        use_src_lemma: bool = config["use_src_lemma"]
        src_casing: str = config["src_casing"]
        src_normalize: bool = config["src_normalize"]
        trg_casing: str = config["trg_casing"]
        trg_normalize: bool = config["trg_normalize"]
        corpus: List[ParallelSegment] = []
        lexicon = Lexicon()
        for verse in verses:
            ref_str = get_ref(verse)
            source = get_segment(verse["manuscript"], src_casing, src_normalize, use_lemma=use_src_lemma)
            target = get_segment(verse["translation"], trg_casing, trg_normalize)
            alignment = get_alignment(verse)
            corpus.append(ParallelSegment(ref_str, source, target, alignment))
            add_alignment(lexicon, source, target, get_alignment(verse, primary_links_only=True))
        lexicon.normalize()

        src_stemmer = get_stemmer(config["src_stemmer"])
        src_stemmer.train(map(lambda s: s.source, corpus))

        trg_stemmer = get_stemmer(config["trg_stemmer"])
        trg_stemmer.train(map(lambda s: s.target, corpus))

        if config["by_book"]:
            for book, book_exp_dir in get_all_book_paths(exp_dir):
                book_corpus = list(filter(lambda s: is_in_book(s, book), corpus))
                if len(book_corpus) > 0:
                    book_exp_dir.mkdir(exist_ok=True)
                    write_datasets(book_exp_dir, src_stemmer, trg_stemmer, book_corpus)
        else:
            write_datasets(exp_dir, src_stemmer, trg_stemmer, corpus)
            lexicon.write(exp_dir / "lexicon.gold.txt")


if __name__ == "__main__":
    main()
