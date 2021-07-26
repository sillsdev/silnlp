from typing import Dict, Set

import pandas as pd

from ..common.canon import ALL_BOOK_IDS, book_id_to_number
from ..common.environment import SIL_NLP_ENV
from .config import get_all_book_paths

ALIGNERS = [
    "PT",
    "IBM-1",
    "IBM-2",
    "FastAlign",
    "Giza-HMM",
    "Giza-IBM-4",
    "Clear-2-IBM-1",
    "Clear-2-IBM-2",
    "Clear-2-FA",
    "Clear-2-HMM",
    "Clear-2-IBM-4",
]
METRICS = [
    "F-Score",
    "Precision",
    "Recall",
    "F-Score@1",
    "Precision@1",
    "Recall@1",
    "F-Score@3",
    "Precision@3",
    "Recall@3",
    "MAP",
    "AO@1",
    "RBO",
]
TRANSLATIONS = [
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
TESTAMENTS = ["nt", "ot", "nt+ot"]
EXP_DIR = SIL_NLP_ENV.align_experiments_dir / "pab-nlp"


def aggregate_testament_results() -> None:
    for testament in TESTAMENTS:
        data: Dict[str, pd.DataFrame] = {}
        available_books: Set[str] = set()
        available_aligners: Set[str] = set()
        for translation in TRANSLATIONS:
            translation_dir = EXP_DIR / translation
            exp_dir = translation_dir if testament == "nt+ot" else translation_dir / testament
            scores_path = exp_dir / "scores.csv"
            if scores_path.is_file():
                df = pd.read_csv(scores_path, index_col=[0, 1])
                data[translation] = df
                available_books.update(df.index.get_level_values("Book"))
                available_aligners.update(df.index.get_level_values("Model"))
        available_books.remove("ALL")

        for metric in METRICS:
            output_path = EXP_DIR / f"{testament}.all.{metric}.csv"
            with open(output_path, "w") as output_file:
                output_file.write("Model," + ",".join(filter(lambda t: t in data, TRANSLATIONS)) + "\n")
                for aligner in ALIGNERS:
                    output_file.write(aligner.replace("Giza-", ""))
                    for translation in TRANSLATIONS:
                        df = data.get(translation)
                        if df is None:
                            continue
                        output_file.write(",")
                        if ("ALL", aligner) in df.index:
                            output_file.write(str(df.at[("ALL", aligner), metric]))
                    output_file.write("\n")

            if len(available_books) > 0:
                for aligner in available_aligners:
                    output_path = EXP_DIR / f"{testament}.all.{aligner}.{metric}.csv"
                    with open(output_path, "w") as output_file:
                        output_file.write("Book," + ",".join(filter(lambda t: t in data, TRANSLATIONS)) + "\n")
                        for book_id in sorted(available_books, key=lambda b: book_id_to_number(b)):
                            output_file.write(book_id)
                            for translation in TRANSLATIONS:
                                df = data.get(translation)
                                if df is None:
                                    continue
                                output_file.write(",")
                                if (book_id, aligner) in df.index:
                                    output_file.write(str(df.at[(book_id, aligner), metric]))
                            output_file.write("\n")


def aggregate_book_results() -> None:
    for testament in TESTAMENTS:
        if testament == "nt+ot":
            continue
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        available_translations: Set[str] = set()
        available_aligners: Set[str] = set()
        for translation in TRANSLATIONS:
            exp_dir = EXP_DIR / f"{translation}-by-book" / testament
            if not exp_dir.is_dir():
                continue
            available_translations.add(translation)
            for book_id, book_dir in get_all_book_paths(exp_dir):
                scores_path = book_dir / "scores.csv"
                if scores_path.is_file():
                    df = pd.read_csv(scores_path, index_col=[0, 1])
                    book_dict = data.get(book_id, {})
                    book_dict[translation] = df
                    data[book_id] = book_dict
                    available_aligners.update(df.index.get_level_values("Model"))

        for metric in METRICS:
            for aligner in available_aligners:
                output_path = EXP_DIR / f"{testament}.single.{aligner}.{metric}.csv"
                with open(output_path, "w") as output_file:
                    output_file.write(
                        "Book," + ",".join(filter(lambda t: t in available_translations, TRANSLATIONS)) + "\n"
                    )
                    for book_id in ALL_BOOK_IDS:
                        book_dict = data.get(book_id, {})
                        if len(book_dict) == 0:
                            continue
                        output_file.write(book_id)
                        for translation in TRANSLATIONS:
                            if translation not in available_translations:
                                continue
                            df = book_dict.get(translation)
                            output_file.write(",")
                            if df is not None and ("ALL", aligner) in df.index:
                                output_file.write(str(df.at[("ALL", aligner), metric]))
                        output_file.write("\n")


def main() -> None:
    aggregate_testament_results()
    aggregate_book_results()


if __name__ == "__main__":
    main()
