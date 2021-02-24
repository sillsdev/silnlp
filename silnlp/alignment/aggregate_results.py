import os
from typing import Dict, List, Set

import pandas as pd

from ..common.canon import ALL_BOOK_IDS, book_id_to_number
from ..common.environment import ALIGN_EXPERIMENTS_DIR
from .config import get_all_book_paths

ALIGNERS = ["PT", "Clear-2", "FastAlign", "HMM", "IBM-1", "IBM-2", "IBM-4"]
METRICS = ["AER", "F-Score", "Precision", "Recall"]
TESTAMENTS = ["nt", "ot"]


def aggregate_testament_results(translations: List[str]) -> None:
    for testament in TESTAMENTS:
        data: Dict[str, pd.DataFrame] = {}
        available_books: Set[str] = set()
        available_aligners: Set[str] = set()
        for translation in translations:
            scores_path = os.path.join(ALIGN_EXPERIMENTS_DIR, f"{translation}.{testament}", "scores.csv")
            if os.path.isfile(scores_path):
                df = pd.read_csv(scores_path, index_col=[0, 1])
                data[translation] = df
                available_books.update(df.index.get_level_values("Book"))
                available_aligners.update(df.index.get_level_values("Model"))
        available_books.remove("ALL")

        for metric in METRICS:
            output_path = os.path.join(ALIGN_EXPERIMENTS_DIR, f"{testament}.all.{metric}.csv")
            with open(output_path, "w") as output_file:
                output_file.write("Model," + ",".join(filter(lambda t: t in data, translations)) + "\n")
                for aligner in ALIGNERS:
                    output_file.write(aligner)
                    for translation in translations:
                        df = data.get(translation)
                        if df is None:
                            continue
                        output_file.write(",")
                        if ("ALL", aligner) in df.index:
                            output_file.write(str(df.at[("ALL", aligner), metric]))
                    output_file.write("\n")

            for aligner in available_aligners:
                output_path = os.path.join(ALIGN_EXPERIMENTS_DIR, f"{testament}.all.{aligner}.{metric}.csv")
                with open(output_path, "w") as output_file:
                    output_file.write("Book," + ",".join(filter(lambda t: t in data, translations)) + "\n")
                    for book_id in sorted(available_books, key=lambda b: book_id_to_number(b)):
                        output_file.write(book_id)
                        for translation in translations:
                            df = data.get(translation)
                            if df is None:
                                continue
                            output_file.write(",")
                            if (book_id, aligner) in df.index:
                                output_file.write(str(df.at[(book_id, aligner), metric]))
                        output_file.write("\n")


def aggregate_book_results(translations: List[str]) -> None:
    for testament in TESTAMENTS:
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        available_translations: Set[str] = set()
        available_aligners: Set[str] = set()
        for translation in translations:
            root_dir = os.path.join(ALIGN_EXPERIMENTS_DIR, f"{translation}.{testament}-by-book")
            if not os.path.isdir(root_dir):
                continue
            available_translations.add(translation)
            for book_id, book_dir in get_all_book_paths(root_dir):
                scores_path = os.path.join(book_dir, "scores.csv")
                if os.path.isfile(scores_path):
                    df = pd.read_csv(scores_path, index_col=[0, 1])
                    book_dict = data.get(book_id, {})
                    book_dict[translation] = df
                    data[book_id] = book_dict
                    available_aligners.update(df.index.get_level_values("Model"))

        for metric in METRICS:
            for aligner in available_aligners:
                output_path = os.path.join(ALIGN_EXPERIMENTS_DIR, f"{testament}.single.{aligner}.{metric}.csv")
                with open(output_path, "w") as output_file:
                    output_file.write(
                        "Book," + ",".join(filter(lambda t: t in available_translations, translations)) + "\n"
                    )
                    for book_id in ALL_BOOK_IDS:
                        book_dict = data.get(book_id, {})
                        if len(book_dict) == 0:
                            continue
                        output_file.write(book_id)
                        for translation in translations:
                            if translation not in available_translations:
                                continue
                            df = book_dict.get(translation)
                            output_file.write(",")
                            if df is not None and ("ALL", aligner) in df.index:
                                output_file.write(str(df.at[("ALL", aligner), metric]))
                        output_file.write("\n")


def main() -> None:
    translations = [
        "mcl",
        "ccb",
        "cuvmp",
        "rcuv",
        "esv",
        "kjv",
        "niv11",
        "niv84",
        "nrsv",
        "rsv",
        "hovr",
        "shk",
        "khov",
        "nrt",
    ]

    aggregate_testament_results(translations)
    aggregate_book_results(translations)


if __name__ == "__main__":
    main()
