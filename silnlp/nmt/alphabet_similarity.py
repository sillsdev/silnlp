import argparse
from pathlib import Path
from typing import List, Set

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..common.environment import SNE
from .config import load_config


def get_corpus_path(project: str) -> Path:
    return SNE._MT_SCRIPTURE_DIR / f"{project}.txt"


def computeSimilarity(projects: List[str]) -> None:
    charSets = list()
    projects.sort()
    resourceDf = pd.DataFrame(columns=["CharCount", "Mean"])
    resourceDf.reindex(projects)

    for project in projects:
        with open(get_corpus_path(project), "r", encoding="utf-8") as f:
            text = f.read()
        thisCharSet = list()
        for i in range(len(text)):
            if text[i] not in thisCharSet:
                thisCharSet.append(text[i])
        thisCharSet.sort()
        print(f"Resource: {project}\t#chars: {len(thisCharSet)}")
        resourceDf.at[project, "CharCount"] = len(thisCharSet)
        charSets.append({"resource": project, "chars": thisCharSet})

    # Create a matrix for storing the similarity metrics
    similarityDf = pd.DataFrame(columns=[projects])
    similarityDf.reindex(projects)

    totalSimilarity = 0.0
    for i in range(len(projects) - 1):
        # Get the charset for resource 1 and make a set from it
        project1 = projects[i]
        charSet1 = charSets[i]
        l1 = charSet1["chars"]
        l1set = set(l1)

        for j in range(i + 1, len(projects)):
            # Get the charset for resource 2 and make a set from it.
            project2 = projects[j]
            charSet2 = charSets[j]
            l2 = charSet2["chars"]
            l2set = set(l2)

            # Calculate the differences between sets 1 and 2
            diff1v2 = l1set.difference(l2set)
            diff2v1 = l2set.difference(l1set)

            # Calculate the similarity score
            similarity = (1 - (len(diff1v2) / len(l1))) * (1 - (len(diff2v1) / len(l2))) * 100
            totalSimilarity += similarity
            #            print(f"Similarity ({file_name1:12} vs {file_name2:12}):\t{similarity:5.1f}")

            # Update similarity score in the dataframe
            similarityDf.at[project1, project2] = similarity
            similarityDf.at[project2, project1] = similarity

    totalSimilarity = totalSimilarity / (len(projects) * (len(projects) - 1) / 2)
    print(f"Overall similarity: {totalSimilarity:5.1f}")

    # Summarize, sort, and display the mean similarity value for each language
    resourceDf["Mean"] = similarityDf.mean(axis=1)
    print(resourceDf.sort_values("Mean"))

    # Show the per-language pair similarity scores as a heatmap
    similarityDf.fillna(0, inplace=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(similarityDf, square=True, ax=ax)
    plt.show()


def get_iso(file_path: Path) -> str:
    file_name = file_path.name
    parts = file_name.split("-")
    return parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculates alphabet similarity between text corpora in a multilingual data set"
    )
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    exp_name = args.experiment
    config = load_config(exp_name)

    projects: Set[str] = set()
    for pair in config.corpus_pairs:
        if pair.is_scripture:
            for src_file in pair.src_files:
                projects.add(f"{src_file.iso}-{src_file.project}")
            for trg_file in pair.trg_files:
                projects.add(f"{trg_file.iso}-{trg_file.project}")

    computeSimilarity(list(projects))


if __name__ == "__main__":
    main()
