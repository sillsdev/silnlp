import argparse
import csv
import logging
from math import sqrt
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from machine.corpora import escape_spaces, lowercase, nfc_normalize
from machine.tokenization import LatinWordTokenizer
from matplotlib.widgets import Slider
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sortedcontainers import SortedSet

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
        return 0 if len(scores) == 0 else mean(scores)


def tokenize_verses(verses: Iterable[str], output_path: Path) -> None:
    tokenizer = LatinWordTokenizer()
    with output_path.open("w", encoding="utf-8", newline="\n") as output_stream:
        for verse in verses:
            tokens = lowercase(nfc_normalize(escape_spaces(tokenizer.tokenize(verse))))
            output_stream.write(" ".join(tokens) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize similarity of languages/projects")
    parser.add_argument("--corpus", type=str, required=True, help="The corpus folder")
    parser.add_argument("--metadata", type=str, required=True, help="The metadata file")
    parser.add_argument("--scores", type=str, required=True, help="The similarity scores file")
    parser.add_argument("--image", type=str, help="The image file")
    parser.add_argument("--country", type=str, help="The country to include")
    parser.add_argument("--family", type=str, help="The language family to include")
    parser.add_argument("--aligner", type=str, default="fast_align", help="The alignment model")
    parser.add_argument("--recompute", default=False, action="store_true", help="Recompute similarity scores")
    parser.add_argument("--graph-type", type=str, default="tree", choices=["tree", "network"], help="Type of graph")
    parser.add_argument(
        "--data-type", type=str, default="language", choices=["language", "project"], help="Type of data"
    )
    parser.add_argument("--threshold", type=float, default=1.0, help="Similarity threshold")
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
    projects = SortedSet()
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

    LOGGER.info(f"Found {len(projects)} projects from {title}")

    project_list = list(projects)
    isos = SortedSet()
    remaining_pair_count = 0
    total_pair_count = 0
    for i in range(len(project_list)):
        for j in range(i + 1, len(project_list)):
            project1 = project_list[i]
            project2 = project_list[j]
            item1 = project1.split("-")[0]
            item2 = project2.split("-")[0]
            if item1 != item2:
                isos.add(item1)
                isos.add(item2)
                if recompute or frozenset([project1, project2]) not in trans_scores:
                    remaining_pair_count += 1
                total_pair_count += 1

    if remaining_pair_count > 0:
        LOGGER.info(f"Computing similarity scores for {remaining_pair_count} project pairs")
    pair_num = total_pair_count - remaining_pair_count + 1
    pair_scores: Dict[FrozenSet[str], List[float]] = {}
    data_type: str = args.data_type
    with scores_path.open("a", encoding="utf-8", newline="\n") as out_file:
        for i in range(len(project_list)):
            for j in range(i + 1, len(project_list)):
                project1 = project_list[i]
                project2 = project_list[j]
                item1 = project1.split("-")[0]
                item2 = project2.split("-")[0]
                if item1 != item2:
                    score = trans_scores.get(frozenset([project1, project2]))
                    if recompute or score is None:
                        LOGGER.info(f"Processing {project1} <-> {project2} ({pair_num}/{total_pair_count})")
                        corpus = get_scripture_parallel_corpus(
                            corpus_path / (project1 + ".txt"), corpus_path / (project2 + ".txt")
                        )
                        score = compute_similarity_score(corpus, args.aligner)
                        trans_scores[frozenset([project1, project2])] = score
                        out_file.write(f"{project1},{project2},{score}\n")
                        out_file.flush()
                        pair_num += 1
                    if data_type == "project":
                        pair = frozenset([project1, project2])
                    else:
                        iso1 = project1.split("-")[0]
                        iso2 = project2.split("-")[0]
                        pair = frozenset([iso1, iso2])
                    scores = pair_scores.get(pair)
                    if scores is None:
                        scores = []
                        pair_scores[pair] = scores
                    scores.append(score)

    items = project_list if data_type == "project" else list(isos)
    item_indices = {item: i for i, item in enumerate(items)}
    scores_matrix: List[List[float]] = [[0.0] * len(items) for _ in items]
    for pair, scores in pair_scores.items():
        item1, item2 = pair
        index1 = item_indices[item1]
        index2 = item_indices[item2]
        score = median(scores)
        scores_matrix[index1][index2] = score
        scores_matrix[index2][index1] = score

    image: Optional[str] = args.image

    LOGGER.info("Building graph")
    sim_matrix = np.array(scores_matrix)
    sim_matrix = sim_matrix / min(1.0, float(np.max(sim_matrix)) + 0.01)

    graph_type = args.graph_type
    if graph_type == "network":
        sim_matrix = 3000**sim_matrix
        sim_matrix[sim_matrix == 1] = 0

        graph = nx.to_networkx_graph(sim_matrix)

        figure, axes = plt.subplots(1, 2 if image is None else 1, num=f"{title} Network", figsize=(12, 8))
        ax_main: plt.Axes
        ax_slider: Optional[plt.Axes] = None
        if isinstance(axes, plt.Axes):
            ax_main = axes
            ax_main.set_position([0, 0, 1, 1])
        else:
            ax_main, ax_slider = axes
            ax_main.set_position([0.04, 0.08, 0.92, 0.88])
            ax_slider.set_position([0.2, 0.03, 0.65, 0.03])

        def draw_graph(sim: float) -> None:
            ax_main.clear()
            pos = nx.spring_layout(graph, seed=111, k=1.5 / sqrt(len(items)))
            nx.draw_networkx(
                graph,
                ax=ax_main,
                pos=pos,
                labels={i: item for i, item in enumerate(items)},
                font_size=8,
                verticalalignment="bottom",
                node_size=10,
                edge_color=[
                    "lightgray" if data["weight"] >= (3000**sim) else (1, 1, 1, 0)
                    for _, _, data in graph.edges(data=True)
                ],
            )
            figure.canvas.draw_idle()

        threshold: float = args.threshold
        draw_graph(threshold)
        if ax_slider is not None:
            slider = Slider(ax_slider, "Similarity", 0.0, 1.0, threshold, valstep=0.01)
            slider.on_changed(lambda x: draw_graph(x))
    else:
        sim_matrix = sim_matrix / min(1.0, float(np.max(sim_matrix)) + 0.01)

        dist_matrix = 1.0 - squareform(sim_matrix)
        linkage_matrix = linkage(dist_matrix, method="ward", optimal_ordering=True)
        figure, ax = plt.subplots(num=f"{title} Network", figsize=(12, 8))
        ax.set_position([0.06, 0.125, 0.9, 0.8])
        dendrogram(linkage_matrix, labels=items, leaf_rotation=90, leaf_font_size=6 if len(items) >= 40 else 8, ax=ax)

    if image is None:
        plt.show()
    else:
        plt.savefig(image)


if __name__ == "__main__":
    main()
