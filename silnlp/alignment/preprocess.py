import argparse
import json
import os
from typing import List, Tuple

from nltk.translate import Alignment

from nlp.alignment.config import load_config
from nlp.common.corpus import write_corpus
from nlp.common.environment import align_gold_standards_dir
from nlp.common.utils import get_align_root_dir


class ParallelSegment:
    def __init__(self, ref: str, source: str, target: str, alignment: Alignment) -> None:
        self.ref = ref
        self.source = source
        self.target = target
        self.alignment = alignment


def get_ref(verse: dict) -> str:
    id = str(verse["manuscript"]["words"][0]["id"])
    return id[:-4]


def get_segment(segInfo: dict) -> str:
    words: List[dict] = segInfo["words"]
    return " ".join(map(lambda w: w["text"], words)).lower()


def get_alignment(verse: dict) -> Alignment:
    links: List[List[List[int]]] = verse["links"]
    pairs: List[Tuple[int, int]] = []
    for link in links:
        src_indices = link[0]
        trg_indices = link[1]
        for src_index in src_indices:
            for trg_index in trg_indices:
                pairs.append((src_index, trg_index))
    return Alignment(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesses GBI gold standard alignments")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    root_dir = get_align_root_dir(args.experiment)
    config = load_config(args.experiment)
    corpus_name: str = config["corpus"]

    corpus_path = os.path.join(align_gold_standards_dir, corpus_name + ".alignment.json")
    verses: List[dict]
    with open(corpus_path, "r", encoding="utf-8") as f:
        verses = json.load(f)

    corpus: List[ParallelSegment] = []
    for verse in verses:
        ref = get_ref(verse)
        source = get_segment(verse["manuscript"])
        target = get_segment(verse["translation"])
        alignment = get_alignment(verse)
        corpus.append(ParallelSegment(ref, source, target, alignment))

    train_refs_path = os.path.join(root_dir, "refs.txt")
    write_corpus(train_refs_path, map(lambda s: s.ref, corpus))

    train_src_path = os.path.join(root_dir, "src.txt")
    write_corpus(train_src_path, map(lambda s: s.source, corpus))

    train_trg_path = os.path.join(root_dir, "trg.txt")
    write_corpus(train_trg_path, map(lambda s: s.target, corpus))

    test_alignments_path = os.path.join(root_dir, "alignments.gold.txt")
    write_corpus(test_alignments_path, map(lambda s: str(s.alignment), corpus))


if __name__ == "__main__":
    main()
