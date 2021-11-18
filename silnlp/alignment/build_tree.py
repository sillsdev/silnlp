import argparse
import csv
import logging
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from machine.corpora import ESCAPE_SPACES, LOWERCASE, NFC_NORMALIZE, pipeline
from machine.tokenization import LatinWordTokenizer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ..common.corpus import get_scripture_parallel_corpus
from .config import get_aligner
from .utils import compute_alignment_score

LOGGER = logging.getLogger(__package__ + ".build_tree")


def compute_similarity_score(corpus: pd.DataFrame, aligner_id: str) -> float:
    with TemporaryDirectory() as td:
        temp_dir = Path(td)
        src_tok_output_path = temp_dir / "tokenize-src-output.txt"
        trg_tok_output_path = temp_dir / "tokenize-trg-output.txt"

        tokenize_verses(corpus["source"], src_tok_output_path)
        tokenize_verses(corpus["target"], trg_tok_output_path)

        aligner = get_aligner(aligner_id, temp_dir)

        sym_align_path = temp_dir / "sym-align.txt"
        aligner.train(src_tok_output_path, trg_tok_output_path)
        aligner.align(sym_align_path)

        direct_lexicon = aligner.get_direct_lexicon(include_special_tokens=True)
        inverse_lexicon = aligner.get_inverse_lexicon(include_special_tokens=True)

        scores: List[float] = []
        with src_tok_output_path.open("r", encoding="utf-8") as src_tok_output_file, trg_tok_output_path.open(
            "r", encoding="utf-8"
        ) as trg_tok_output_file, sym_align_path.open("r", encoding="utf-8") as sym_align_file:
            for src_sentence, trg_sentence, alignment in zip(src_tok_output_file, trg_tok_output_file, sym_align_file):
                src_sentence = src_sentence.strip()
                trg_sentence = trg_sentence.strip()
                alignment = alignment.strip()
                if len(alignment) > 0:
                    scores.append(
                        compute_alignment_score(direct_lexicon, inverse_lexicon, src_sentence, trg_sentence, alignment)
                    )
        return mean(scores)


def tokenize_verses(verses: Iterable[str], output_path: Path) -> None:
    tokenizer = LatinWordTokenizer()
    processor = pipeline(ESCAPE_SPACES, NFC_NORMALIZE, LOWERCASE)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_stream:
        for verse in verses:
            tokens = processor.process(tokenizer.tokenize(verse))
            output_stream.write(" ".join(tokens) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Builds a phylogenetic tree of languages from Bible translations")
    parser.add_argument("--corpus", type=str, required=True, help="The corpus folder.")
    parser.add_argument("--metadata", type=str, required=True, help="The metadata file.")
    parser.add_argument("--scores", type=str, required=True, help="The output scores file.")
    parser.add_argument("--image", type=str, help="The output image file.")
    parser.add_argument("--country", type=str, help="The country to include.")
    parser.add_argument("--family", type=str, help="The language family to include.")
    parser.add_argument("--aligner", type=str, default="fast_align", help="The aligner.")
    parser.add_argument("--recompute", default=False, action="store_true", help="Recompute scores")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)

    metadata_path = Path(args.metadata)
    country: Optional[str] = args.country
    family: Optional[str] = args.family
    if country is None and family is None:
        raise RuntimeError("Either --country or --family must be specified.")

    title = ""
    if family is not None:
        title = family
    if country is not None:
        if len(title) > 0:
            title += "-"
        title += country

    recompute: bool = args.recompute
    country_lower = None if country is None else country.lower()
    family_lower = None if family is None else family.lower()
    projects: Set[str] = set()
    with metadata_path.open("r", encoding="utf-8-sig") as metadata_file:
        reader = csv.reader(metadata_file)
        for r in reader:
            resource, _, _, trans_family, trans_country = r
            if country_lower is not None and trans_country.lower() != country_lower:
                continue
            if family_lower is not None and trans_family.lower() != family_lower:
                continue
            projects.add(Path(resource).stem)

    scores_path = Path(args.scores)
    trans_scores: Dict[FrozenSet[str], float] = {}
    if scores_path.is_file():
        with scores_path.open("r", encoding="utf-8-sig") as in_file:
            for row in in_file:
                row = row.strip()
                project1, project2, score_str = row.split(",", maxsplit=3)
                if project1 in projects and project2 in projects:
                    trans_scores[frozenset([project1, project2])] = float(score_str)

    LOGGER.info(f"Found {len(projects)} translations from {title}")

    project_list = list(projects)
    isos: Set[str] = set()
    pair_count = 0
    for i in range(len(project_list)):
        for j in range(i + 1, len(project_list)):
            project1 = project_list[i]
            project2 = project_list[j]
            iso1 = project1.split("-")[0]
            iso2 = project2.split("-")[0]
            if iso1 != iso2:
                isos.add(iso1)
                isos.add(iso2)
                if recompute or frozenset([project1, project2]) not in trans_scores:
                    pair_count += 1

    if pair_count > 0:
        LOGGER.info(f"Computing similarity scores for {pair_count} translation pairs")
    iso_list = list(isos)
    iso_indices = {iso: i for i, iso in enumerate(iso_list)}
    iso_scores: List[List[float]] = [[0.0] * len(iso_list) for _ in iso_list]
    pair_num = 1
    with scores_path.open("a", encoding="utf-8", newline="\n") as out_file:
        for i in range(len(project_list)):
            for j in range(i + 1, len(project_list)):
                project1 = project_list[i]
                project2 = project_list[j]
                iso1 = project1.split("-")[0]
                iso2 = project2.split("-")[0]
                if iso1 != iso2:
                    score = trans_scores.get(frozenset([project1, project2]))
                    if recompute or score is None:
                        LOGGER.info(f"Processing {project1} <-> {project2} ({pair_num}/{pair_count})")
                        corpus = get_scripture_parallel_corpus(
                            corpus_path / (project1 + ".txt"), corpus_path / (project2 + ".txt")
                        )
                        score = compute_similarity_score(corpus, args.aligner)
                        trans_scores[frozenset([project1, project2])] = score
                        out_file.write(f"{project1},{project2},{score}\n")
                        out_file.flush()
                        pair_num += 1
                    index1 = iso_indices[iso1]
                    index2 = iso_indices[iso2]
                    if score > iso_scores[index1][index2]:
                        iso_scores[index1][index2] = score
                        iso_scores[index2][index1] = score

    LOGGER.info("Building tree")
    sim_matrix = np.array(iso_scores)
    sim_matrix = sim_matrix / min(1.0, float(np.max(sim_matrix)) + 0.01)

    dist_matrix = 1.0 - squareform(sim_matrix)
    linkage_matrix = linkage(dist_matrix, method="ward", optimal_ordering=True)
    plt.figure(title, figsize=(10, 8))
    plt.title(title)
    dendrogram(linkage_matrix, labels=iso_list, leaf_rotation=90, leaf_font_size=8)

    image: Optional[str] = args.image
    if image is None:
        plt.show()
    else:
        plt.savefig(image)


if __name__ == "__main__":
    main()
