import argparse
import logging
import re
from statistics import mean
from typing import List, Tuple

import pandas as pd
from machine.scripture import VerseRef, is_ot_nt
from tqdm import tqdm

from ..alignment.config import get_aligner_name
from ..alignment.utils import add_alignment_scores
from ..common.collect_verse_counts import DT_CANON, NT_CANON, OT_CANON, collect_verse_counts
from ..common.corpus import filter_parallel_corpus, get_mt_corpus_path, get_scripture_parallel_corpus, include_chapters
from ..common.environment import SIL_NLP_ENV
from ..common.script_utils import get_script, is_represented
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import Config, get_data_file_pairs

LOGGER = logging.getLogger(__package__ + ".analyze_project_pairs")

ALIGNMENT_SCORES_FILE = re.compile(r"([a-z]{2,3}-.+)_([a-z]{2,3}-.+)")


def get_corpus_stats(config: Config, force_align: bool = False, deutero: bool = False) -> None:
    stats_path = config.exp_dir / "corpus-stats.csv"
    if stats_path.is_file() and not force_align:
        stats_df = pd.read_csv(stats_path, keep_default_na=False, index_col=["src_project", "trg_project"])
    else:
        stats_df = pd.DataFrame(
            columns=[
                "src_project",
                "trg_project",
                "count",
                "src_only",
                "trg_only",
                "parallel",
                "align_score",
                "filtered_count",
                "filtered_align_score",
                "src_script",
                "src_script_in_model",
                "trg_script",
                "trg_script_in_model",
            ],
        ).set_index(["src_project", "trg_project"])

    pairs_to_process = []
    for pair in config.corpus_pairs:
        for src_file, trg_file in get_data_file_pairs(pair):
            if src_file == trg_file:
                continue

            src_project, trg_project = f"{src_file.iso}-{src_file.project}", f"{trg_file.iso}-{trg_file.project}"
            if force_align:
                (config.exp_dir / f"{src_project}_{trg_project}.csv").unlink(missing_ok=True)
                (config.exp_dir / f"{trg_project}_{src_project}.csv").unlink(missing_ok=True)

            pairs_to_process.append((src_file, trg_file))

    for src_file, trg_file in pairs_to_process:
        project_pair = (f"{src_file.iso}-{src_file.project}", f"{trg_file.iso}-{trg_file.project}")
        if project_pair not in stats_df.index:
            corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path, False)
            corpus = corpus.loc[(corpus["source"].str.len() > 0) | (corpus["target"].str.len() > 0)]
            align_corpus = corpus.loc[(corpus["source"].str.len() > 0) & (corpus["target"].str.len() > 0)]

            # Align pair
            pair_stats_path = config.exp_dir / f"{project_pair[0]}_{project_pair[1]}.csv"
            alt_pair_stats_path = config.exp_dir / f"{project_pair[1]}_{project_pair[0]}.csv"
            if not pair_stats_path.is_file() and alt_pair_stats_path.is_file():
                pair_stats_path = alt_pair_stats_path

            if pair_stats_path.is_file():
                LOGGER.info(f"Using pre-existing alignment scores from {pair_stats_path}")
                pair_stats = pd.read_csv(pair_stats_path)
                pair_stats["idx"] = align_corpus.index
                pair_stats.set_index("idx", inplace=True)
                align_corpus.insert(3, "score", pair_stats["score"])
            else:
                aligner_id = config.data["aligner"]
                LOGGER.info(f"Computing alignment scores using {get_aligner_name(aligner_id)}")
                add_alignment_scores(align_corpus, aligner_id)
                align_corpus.to_csv(pair_stats_path, index=False)

            # Filter by book/chapter configurations
            if not deutero:
                corpus = corpus.loc[[is_ot_nt(vref.book_num) for vref in corpus["vref"]]]
                align_corpus = align_corpus.loc[[is_ot_nt(vref.book_num) for vref in align_corpus["vref"]]]
            if len(pair.corpus_books) > 0:
                corpus = include_chapters(corpus, pair.corpus_books)
                align_corpus = include_chapters(align_corpus, pair.corpus_books)

            pair_count = len(corpus.index)
            parallel_count = len(align_corpus.index)
            src_only_count = len(corpus.loc[corpus["target"].str.len() == 0].index)
            trg_only_count = len(corpus.loc[corpus["source"].str.len() == 0].index)

            alignment_score = align_corpus["score"].mean()
            filtered_count = 0
            filtered_alignment_score = alignment_score
            if pair.score_threshold > 0:
                unfiltered_len = len(align_corpus)
                align_corpus = filter_parallel_corpus(align_corpus, pair.score_threshold)
                filtered_count = unfiltered_len - len(align_corpus)
                filtered_alignment_score = mean(align_corpus["score"])

            src_script = get_script("".join(corpus["source"][: min(len(corpus["source"]), 3000)]))
            src_script_in_model = (
                is_represented(src_script, config.model) if config.model != "SILTransformerBase" else None
            )
            trg_script = get_script("".join(corpus["target"][: min(len(corpus["target"]), 3000)]))
            trg_script_in_model = (
                is_represented(trg_script, config.model) if config.model != "SILTransformerBase" else None
            )

            stats_df.loc[project_pair, :] = [
                pair_count,
                src_only_count,
                trg_only_count,
                parallel_count,
                alignment_score,
                filtered_count,
                filtered_alignment_score,
                src_script,
                src_script_in_model,
                trg_script,
                trg_script_in_model,
            ]

        # Use values from the df because not all values get recalculated
        row = stats_df.loc[project_pair]
        align_score = "{:.4f}".format(row.at["align_score"])
        filtered_align_score = "{:.4f}".format(row.at["filtered_align_score"])
        LOGGER.info(
            f"{src_file.project} -> {trg_file.project} stats -"
            f" count: {row.at['count']},"
            f" source only count: {row.at['src_only']}"
            f" target only count: {row.at['trg_only']}"
            f" parallel count: {row.at['parallel']}"
            f" alignment: {align_score},"
            f" filtered count: {row.at['filtered_count']},"
            f" alignment (filtered): {filtered_align_score},"
            f" source script: {row.at['src_script']}, source script in model: {row.at['src_script_in_model']},"
            f" target script: {row.at['trg_script']}, target script in model: {row.at['trg_script_in_model']}"
        )
    stats_df.sort_index().to_csv(stats_path)


def get_extra_alignments(config: Config, deutero: bool = False) -> List[str]:
    stats_path = config.exp_dir / "corpus-stats.csv"
    if stats_path.is_file():
        stats_df = pd.read_csv(stats_path, keep_default_na=False, index_col=["src_project", "trg_project"])
    else:
        stats_df = pd.DataFrame(
            columns=[
                "src_project",
                "trg_project",
                "count",
                "src_only",
                "trg_only",
                "parallel",
                "align_score",
                "filtered_count",
                "filtered_align_score",
                "src_script",
                "src_script_in_model",
                "trg_script",
                "trg_script_in_model",
            ],
        ).set_index(["src_project", "trg_project"])

    LOGGER.info("Getting statistics from extra alignment files")
    skipped = set()
    extra_projects = []
    for filepath in config.exp_dir.glob("*.csv"):
        match = ALIGNMENT_SCORES_FILE.fullmatch(filepath.stem)
        if match is None:
            skipped.add(filepath.name)
            continue
        project_pair = match.group(1, 2)

        if project_pair not in stats_df.index:
            LOGGER.info(f"Extra alignment file found: {filepath.name}")
            src_path = get_mt_corpus_path(project_pair[0])
            if src_path.is_file():
                extra_projects.append(src_path.name)
            trg_path = get_mt_corpus_path(project_pair[1])
            if trg_path.is_file():
                extra_projects.append(trg_path.name)

            if src_path.is_file() and trg_path.is_file():
                corpus = get_scripture_parallel_corpus(src_path, trg_path, False)
                corpus = corpus.loc[(corpus["source"].str.len() > 0) | (corpus["target"].str.len() > 0)]
                if not deutero:
                    corpus = corpus.loc[[is_ot_nt(vref.book_num) for vref in corpus["vref"]]]
                pair_count = len(corpus.index)
                src_only_count = len(corpus.loc[corpus["target"].str.len() == 0].index)
                trg_only_count = len(corpus.loc[corpus["source"].str.len() == 0].index)
            else:
                LOGGER.info(
                    f"Original source or target project for {project_pair} not found. Some statistics will be missing."
                )
                pair_count = ""
                src_only_count = ""
                trg_only_count = ""

            align_corpus = pd.read_csv(filepath)
            if not deutero:
                align_corpus = align_corpus.loc[
                    [is_ot_nt(VerseRef.from_string(vref).book_num) for vref in align_corpus["vref"]]
                ]
            parallel_count = len(align_corpus.index)
            src_script = get_script("".join(align_corpus["source"][: min(len(align_corpus["source"]), 3000)]))
            src_script_in_model = (
                is_represented(src_script, config.model) if config.model != "SILTransformerBase" else None
            )
            trg_script = get_script("".join(align_corpus["target"][: min(len(align_corpus["target"]), 3000)]))
            trg_script_in_model = (
                is_represented(trg_script, config.model) if config.model != "SILTransformerBase" else None
            )

            stats_df.loc[project_pair, :] = [
                pair_count,
                src_only_count,
                trg_only_count,
                parallel_count,
                align_corpus["score"].mean(),
                "",
                "",
                src_script,
                src_script_in_model,
                trg_script,
                trg_script_in_model,
            ]
    stats_df.to_csv(stats_path)

    expected = {"corpus-stats.csv", "verse_counts.csv", "verse_percentages.csv"}
    skipped = skipped - expected
    if len(skipped) > 0:
        LOGGER.info(f"Files skipped: {skipped}")
    return extra_projects


def create_summary_file(config: Config) -> None:
    corpus_stats_df = pd.read_csv(config.exp_dir / "corpus-stats.csv", index_col=["src_project", "trg_project"])
    verse_counts_df = pd.read_csv(config.exp_dir / "verse_counts.csv", index_col="file")
    verse_percentages_df = pd.read_csv(config.exp_dir / "verse_percentages.csv", index_col="file")
    alignment_scores_df = pd.read_excel(
        config.exp_dir / f"{config.exp_dir.stem}_alignment_breakdown.xlsx",
        sheet_name="By Book",
        index_col=[0, 1],
    )

    LOGGER.info("Creating summary file")
    trg_summary_dfs = {}
    if len(corpus_stats_df.index.levels[1]) > 250:
        # Max # of sheets in an Excel spreadsheet is 255
        LOGGER.warning("Too many target projects (>250); Summary sheets for each target project will not be written.")
    else:
        for trg_project in tqdm(corpus_stats_df.index.levels[1]):
            summary_df = pd.DataFrame(columns=["file"] + list(verse_counts_df.columns)).set_index("file")
            summary_df.loc[trg_project] = verse_counts_df.loc[trg_project]

            for src_project in sorted(corpus_stats_df.swaplevel().loc[trg_project].index):
                summary_df = pd.concat([summary_df, pd.Series({col: "" for col in summary_df.columns}).to_frame().T])
                summary_df.loc[src_project] = verse_counts_df.loc[src_project]
                pair = (src_project, trg_project)
                if pair not in alignment_scores_df.index:
                    pair = (trg_project, src_project)
                summary_df.loc[f"Alignment with {src_project}"] = alignment_scores_df.loc[pair]

            trg_summary_dfs[trg_project] = summary_df

    # Write to Excel file
    with pd.ExcelWriter(config.exp_dir / f"{config.exp_dir.stem}_analysis.xlsx", engine="xlsxwriter") as writer:
        corpus_stats_df.to_excel(writer, sheet_name="corpus_stats", merge_cells=False)
        verse_counts_df.to_excel(writer, sheet_name="verse_counts")
        verse_percentages_df.to_excel(writer, sheet_name="verse_percentages")

        # Add Excel formatting
        workbook = writer.book
        int_format = workbook.add_format({"num_format": "0"})
        round4_format = workbook.add_format({"num_format": "0.0000"})

        corpus_stats_sheet = writer.sheets["corpus_stats"]
        corpus_stats_sheet.set_column(2, 8, None, int_format)
        corpus_stats_sheet.set_column(6, 6, None, round4_format)
        corpus_stats_sheet.set_column(8, 8, None, round4_format)

        counts_sheet = writer.sheets["verse_counts"]
        counts_sheet.set_column(1, len(verse_counts_df.columns), None, int_format)

        percent_sheet = writer.sheets["verse_percentages"]
        percent_sheet.set_column(1, len(verse_percentages_df.columns), None, int_format)

        # Add comparison sheets for each target project
        sheet_names = {}
        for trg_project in sorted(trg_summary_dfs.keys()):
            # Handle duplicates caused by max length of sheet names (31)
            name = trg_project[:29]
            if name in sheet_names.keys():
                sheet_names[name] += 1
                name = f"{name}_{sheet_names[name]}"
            else:
                sheet_names[name] = 0

            trg_summary_dfs[trg_project].to_excel(writer, sheet_name=name)
            trg_summary_sheet = writer.sheets[name]
            trg_summary_sheet.set_column(1, len(verse_counts_df.columns), None, int_format)
            for i in range(len(corpus_stats_df.swaplevel().loc[trg_project].index)):
                trg_summary_sheet.set_row(4 + i * 3, None, round4_format)


def split_index(vref: str) -> Tuple[str, str, str]:
    vref = VerseRef.from_string(vref)
    return (vref.book, int(vref.chapter), vref.verse)


# Sort list of verse numbers that may contain ranges, e.g. "15-16"
def sort_verse_nums(verse_nums: List[str]) -> List[str]:
    orig_lens = [len(s) for s in verse_nums]
    verse_nums = [float(n.replace("-", ".")) if "-" in n else int(n) for n in verse_nums]
    verse_nums, orig_lens = zip(*sorted(zip(verse_nums, orig_lens), key=lambda x: x[0]))
    verse_nums = [str(n) if n is int else str(n).replace(".", "-") for n in verse_nums]
    verse_nums = [
        s if len(s) == l else s + ("0" * (l - len(s))) for s, l in zip(verse_nums, orig_lens)
    ]  # restore trailing 0s
    return verse_nums


def create_alignment_breakdown_file(config: Config, deutero: bool) -> None:
    books = OT_CANON + NT_CANON + (DT_CANON if deutero else [])
    corpus_stats_df = pd.read_csv(config.exp_dir / "corpus-stats.csv", index_col=["src_project", "trg_project"])

    LOGGER.info("Creating alignment breakdown file")
    aligment_by_book = pd.DataFrame(columns=["src_project", "trg_project"] + books).set_index(
        ["src_project", "trg_project"]
    )
    alignment_by_chapter = {}
    alignment_by_verse = {}
    book_orders = {}
    num_pairs = 0
    for project_pair in tqdm(corpus_stats_df.index):
        src_project, trg_project = project_pair
        scores_path = config.exp_dir / f"{src_project}_{trg_project}.csv"
        # Only add one breakdown per pair
        if not scores_path.is_file():
            continue
        num_pairs += 1

        # Get verse alignment scores for pair
        align_scores_df = pd.read_csv(scores_path).set_index("vref").drop(columns=["source", "target"])
        align_scores_df.index = pd.MultiIndex.from_tuples(
            [split_index(vref) for vref in align_scores_df.index], names=["book", "chapter", "verse"]
        )
        existing_books = list(filter(lambda x: x in align_scores_df.index.levels[0], books))

        # Make each row the alignment scores for a chapter
        verse_df = (
            align_scores_df.reset_index()
            .pivot(index=["book", "chapter"], columns="verse", values="score")
            .loc[existing_books]
        )
        verse_df = verse_df[sort_verse_nums(verse_df.columns)]

        # Get average alignment scores at the chapter and book level and make df for chapter alignment scores
        chapter_avgs = verse_df.mean(axis=1)
        book_avgs = verse_df.unstack(level=1).mean(axis=1)
        verse_df.insert(0, "average", chapter_avgs)
        chapter_df = verse_df.reset_index().pivot(index="book", columns="chapter", values="average").loc[existing_books]
        chapter_df.insert(0, "average", book_avgs)

        aligment_by_book.loc[project_pair, :] = book_avgs
        alignment_by_chapter[project_pair] = chapter_df
        alignment_by_verse[project_pair] = verse_df

        # Create book alignment ranking
        book_order_df = pd.DataFrame(columns=existing_books)
        book_order_df.index.name = f"{src_project}  {trg_project}"
        book_order_df.loc["Verses in Common"] = [len(align_scores_df.loc[book].index) for book in existing_books]
        book_order_df.loc["align"] = book_avgs
        book_order_df = book_order_df.sort_values("align", axis=1, ascending=False).drop(index="align")
        book_order_df.loc["Cumulative Verses"] = [
            sum(book_order_df.loc["Verses in Common"].iloc[: i + 1]) for i in range(len(existing_books))
        ]
        book_order_df.loc["corpus_books", book_order_df.columns[0]] = ";".join(book_order_df.columns)
        book_orders[project_pair] = book_order_df

    LOGGER.info("Writing alignment breakdown file")
    with pd.ExcelWriter(config.exp_dir / f"{config.exp_dir.stem}_alignment_breakdown.xlsx") as writer:
        aligment_by_book.to_excel(writer, sheet_name="By Book", merge_cells=False)

        # Add Excel formatting
        workbook = writer.book
        int_format = workbook.add_format({"num_format": "0"})
        round4_format = workbook.add_format({"num_format": "0.0000"})
        book_sheet = writer.sheets["By Book"]
        book_sheet.set_column(2, len(aligment_by_book.columns) + 1, None, round4_format)
        book_order_sheet = workbook.add_worksheet("corpus_books")
        writer.sheets["corpus_books"] = book_order_sheet

        # Max # of sheets in an Excel spreadsheet is 255
        if num_pairs > 125:
            LOGGER.warning(
                "Too many project pairs (>125); Alignment breakdowns beyond the book level will not be written."
            )
            return

        # Add chapter-level and verse-level sheets for each pair
        sheet_names = {}
        i = 0
        for project_pair in tqdm(corpus_stats_df.index):
            # Only add one breakdown per pair
            if not (config.exp_dir / f"{project_pair[0]}_{project_pair[1]}.csv").is_file():
                continue

            # Add book order info to corpus_books sheet
            book_orders[project_pair].to_excel(writer, sheet_name="corpus_books", startrow=i * 5)
            book_order_sheet.set_row(i * 5 + 1, None, int_format)
            book_order_sheet.set_row(i * 5 + 2, None, int_format)
            i += 1

            # Handle duplicates caused by max length of sheet names (31)
            name = f"{project_pair[0][:9]}_{project_pair[1][:9]}"
            if name in sheet_names.keys():
                sheet_names[name] += 1
                name = f"{name}_{sheet_names[name]}"
            else:
                sheet_names[name] = 0

            alignment_by_chapter[project_pair].to_excel(writer, sheet_name=f"{name}_chapters")
            chapter_sheet = writer.sheets[f"{name}_chapters"]
            chapter_sheet.set_row(0, None, int_format)
            chapter_sheet.set_column(1, len(alignment_by_chapter[project_pair].columns), None, round4_format)
            alignment_by_verse[project_pair].to_excel(writer, sheet_name=f"{name}_verses")
            verse_sheet = writer.sheets[f"{name}_verses"]
            verse_sheet.set_column(2, len(alignment_by_verse[project_pair].columns) + 1, None, round4_format)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verse counts and compute alignment scores")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--recalculate",
        default=False,
        action="store_true",
        help="Force recalculation of all verse counts and alignment scores",
    )
    parser.add_argument("--deutero", default=False, action="store_true", help="Include deuterocanonical books")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    args = parser.parse_args()

    get_git_revision_hash()

    exp_name = args.experiment
    SIL_NLP_ENV.copy_experiment_from_bucket(exp_name)

    if args.clearml_queue is not None and "cpu" not in args.clearml_queue:
        LOGGER.warning("Running this script on a GPU queue will not speed it up. Please only use CPU queues.")
        exit()
    clearml = SILClearML(exp_name, args.clearml_queue)

    config = clearml.config
    config.set_seed()

    # Confirm that input file paths exist and make sure every corpus pair is many to many
    data_files = []
    for pair in config.corpus_pairs:
        data_files += [f.path for f in pair.src_files] + [f.path for f in pair.trg_files]
    for file in data_files:
        if not file.is_file():
            LOGGER.error(f"The source file {str(file)} does not exist")
            return

    file_patterns = ";".join([f.name for f in data_files])
    collect_verse_counts(SIL_NLP_ENV.mt_scripture_dir, config.exp_dir, file_patterns, args.deutero, args.recalculate)

    get_corpus_stats(config, args.recalculate, args.deutero)

    # Add stats about projects in extra alignment files in the experiment folder
    extra_projects = get_extra_alignments(config, args.deutero)
    all_projects = set(extra_projects) | set([f.name for f in data_files])
    if len(all_projects) > len(data_files):
        LOGGER.info("Adding verse counts for projects in extra alignment files")
        collect_verse_counts(SIL_NLP_ENV.mt_scripture_dir, config.exp_dir, ";".join(all_projects), args.deutero)

    # Create summary outputs
    create_alignment_breakdown_file(config, args.deutero)
    create_summary_file(config)

    SIL_NLP_ENV.copy_experiment_to_bucket(exp_name, overwrite=args.recalculate)


if __name__ == "__main__":
    main()
