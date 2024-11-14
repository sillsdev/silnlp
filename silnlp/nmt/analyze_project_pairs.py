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
from ..common.script_utils import is_represented, predict_script_code
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import Config, get_data_file_pairs

LOGGER = logging.getLogger(__package__ + ".analyze_project_pairs")

ALIGNMENT_SCORES_FILE = re.compile(r"([a-z]{2,3}-.+)_([a-z]{2,3}-.+)")


def get_corpus_stats(config: Config, force_align: bool = False, deutero: bool = False) -> None:
    stats_path = config.exp_dir / "corpus-stats.csv"
    if stats_path.is_file() and not force_align:
        stats_df = pd.read_csv(stats_path, dtype=str, keep_default_na=False, index_col=["src_project", "trg_project"])
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
            dtype=str,
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

            pairs_to_process.append((src_file, trg_file, pair.corpus_books, pair.score_threshold))

    for src_file, trg_file, corpus_books, score_threshold in pairs_to_process:
        project_pair = (f"{src_file.iso}-{src_file.project}", f"{trg_file.iso}-{trg_file.project}")
        if project_pair not in stats_df.index:
            # Get corpus and filter by book/chapter configurations
            corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path, False)
            if not deutero:
                corpus = corpus.loc[[is_ot_nt(vref.book_num) for vref in corpus["vref"]]]
            if len(corpus_books) > 0:
                corpus = include_chapters(corpus, corpus_books)

            pair_count = len(corpus.index)
            src_only_count = len(corpus.loc[corpus["target"].str.len() == 0].index)
            trg_only_count = len(corpus.loc[corpus["source"].str.len() == 0].index)

            corpus = corpus.loc[(corpus["source"].str.len() > 0) & (corpus["target"].str.len() > 0)]
            parallel_count = len(corpus.index)

            # Align pair
            pair_stats_path = config.exp_dir / f"{project_pair[0]}_{project_pair[1]}.csv"
            alt_pair_stats_path = config.exp_dir / f"{project_pair[1]}_{project_pair[0]}.csv"
            if not pair_stats_path.is_file() and alt_pair_stats_path.is_file():
                pair_stats_path = alt_pair_stats_path

            if pair_stats_path.is_file():
                LOGGER.info(f"Using pre-existing alignment scores from {pair_stats_path}")
                pair_stats = pd.read_csv(pair_stats_path)
                pair_stats["idx"] = corpus.index
                pair_stats.set_index("idx", inplace=True)
                corpus.insert(3, "score", pair_stats["score"])
            else:
                aligner_id = config.data["aligner"]
                LOGGER.info(f"Computing alignment scores using {get_aligner_name(aligner_id)}")
                add_alignment_scores(corpus, aligner_id)
                corpus.to_csv(pair_stats_path, index=False)
                SIL_NLP_ENV.copy_experiment_to_bucket(config.exp_dir, pair_stats_path.name, overwrite=True)

            alignment_score = corpus["score"].mean()
            filtered_count = 0
            filtered_alignment_score = alignment_score
            if score_threshold > 0:
                corpus = filter_parallel_corpus(corpus, score_threshold)
                filtered_count = parallel_count - len(corpus)
                filtered_alignment_score = mean(corpus["score"])

            src_script = predict_script_code("".join(corpus["source"][: min(len(corpus["source"]), 3000)]))
            trg_script = predict_script_code("".join(corpus["target"][: min(len(corpus["target"]), 3000)]))
            src_script_in_model = is_represented(src_script, config.model)
            trg_script_in_model = is_represented(trg_script, config.model)

            stats_df.loc[project_pair, :] = [
                pair_count,
                src_only_count,
                trg_only_count,
                parallel_count,
                "{:.4f}".format(alignment_score),
                filtered_count,
                "{:.4f}".format(filtered_alignment_score),
                src_script,
                src_script_in_model,
                trg_script,
                trg_script_in_model,
            ]

        # Use values from the df because not all values get recalculated
        row = stats_df.loc[project_pair]
        LOGGER.info(
            f"{src_file.project} -> {trg_file.project} stats -"
            f" count: {row.at['count']},"
            f" source only count: {row.at['src_only']}"
            f" target only count: {row.at['trg_only']}"
            f" parallel count: {row.at['parallel']}"
            f" alignment: {row.at['align_score']},"
            f" filtered count: {row.at['filtered_count']},"
            f" alignment (filtered): {row.at['filtered_align_score']},"
            f" source script: {row.at['src_script']}, source script in model: {row.at['src_script_in_model']},"
            f" target script: {row.at['trg_script']}, target script in model: {row.at['trg_script_in_model']}"
        )
    stats_df.sort_values("filtered_align_score", ascending=False).to_csv(stats_path)


def get_extra_alignments(config: Config, deutero: bool = False) -> List[str]:
    stats_path = config.exp_dir / "corpus-stats.csv"
    if stats_path.is_file():
        stats_df = pd.read_csv(stats_path, dtype=str, keep_default_na=False, index_col=["src_project", "trg_project"])
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
            dtype=str,
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

        if project_pair not in stats_df.index and project_pair not in stats_df.swaplevel().index:
            LOGGER.info(f"Extra alignment file found: {filepath.name}")
            src_path = get_mt_corpus_path(project_pair[0])
            if src_path.is_file():
                extra_projects.append(src_path.name)
            trg_path = get_mt_corpus_path(project_pair[1])
            if trg_path.is_file():
                extra_projects.append(trg_path.name)

            if src_path.is_file() and trg_path.is_file():
                corpus = get_scripture_parallel_corpus(src_path, trg_path, False)
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
            src_script = predict_script_code("".join(align_corpus["source"][: min(len(align_corpus["source"]), 3000)]))
            trg_script = predict_script_code("".join(align_corpus["target"][: min(len(align_corpus["target"]), 3000)]))
            src_script_in_model = is_represented(src_script, config.model)
            trg_script_in_model = is_represented(trg_script, config.model)

            stats_df.loc[project_pair, :] = [
                pair_count,
                src_only_count,
                trg_only_count,
                parallel_count,
                "{:.4f}".format(align_corpus["score"].mean()),
                "",
                "",
                src_script,
                src_script_in_model,
                trg_script,
                trg_script_in_model,
            ]
        elif project_pair in stats_df.index and len(stats_df.loc[project_pair, "filtered_align_score"]) == 0:
            src_path = get_mt_corpus_path(project_pair[0])
            if src_path.is_file():
                extra_projects.append(src_path.name)
            trg_path = get_mt_corpus_path(project_pair[1])
            if trg_path.is_file():
                extra_projects.append(trg_path.name)
    stats_df.sort_values("filtered_align_score", ascending=False).to_csv(stats_path)

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
        for trg_project in tqdm(sorted(corpus_stats_df.index.levels[1])):
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
        corpus_stats_sheet.autofit()
        corpus_stats_sheet.freeze_panes(1, 0)

        counts_sheet = writer.sheets["verse_counts"]
        counts_sheet.autofit()
        counts_sheet.set_column(1, len(verse_counts_df.columns), None, int_format)
        counts_sheet.freeze_panes(1, 1)

        percent_sheet = writer.sheets["verse_percentages"]
        percent_sheet.autofit()
        percent_sheet.set_column(1, len(verse_percentages_df.columns), None, int_format)
        percent_sheet.freeze_panes(1, 1)

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
            trg_summary_sheet.autofit()
            trg_summary_sheet.set_column(1, len(verse_counts_df.columns), None, int_format)
            for i in range(len(corpus_stats_df.swaplevel().loc[trg_project].index)):
                trg_summary_sheet.set_row(4 + i * 3, None, round4_format)
            trg_summary_sheet.freeze_panes(1, 1)


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
    summary_cols = ["Total", "OT", "NT"] + (["DT"] if deutero else [])
    alignment_by_book = pd.DataFrame(columns=["src_project", "trg_project"] + summary_cols + books).set_index(
        ["src_project", "trg_project"]
    )
    alignment_by_verse = {}
    book_orders = {}
    for project_pair in tqdm(corpus_stats_df.sort_index().index):
        src_project, trg_project = project_pair
        scores_path = config.exp_dir / f"{src_project}_{trg_project}.csv"
        # Only add one breakdown per pair
        if project_pair in alignment_by_book.index or project_pair in alignment_by_book.swaplevel().index:
            continue
        if not scores_path.is_file():
            scores_path = config.exp_dir / f"{trg_project}_{src_project}.csv"

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
        alignment_by_book.loc[project_pair, :] = book_avgs
        alignment_by_book.loc[project_pair, "Total"] = verse_df.unstack(level=[0, 1]).mean()
        alignment_by_book.loc[project_pair, "OT"] = (
            verse_df[verse_df.apply(lambda x: x.name[0] in OT_CANON, axis=1)].unstack(level=[0, 1]).mean()
        )
        alignment_by_book.loc[project_pair, "NT"] = (
            verse_df[verse_df.apply(lambda x: x.name[0] in NT_CANON, axis=1)].unstack(level=[0, 1]).mean()
        )
        if deutero:
            alignment_by_book.loc[project_pair, "DT"] = (
                verse_df[verse_df.apply(lambda x: x.name[0] in DT_CANON, axis=1)].unstack(level=[0, 1]).mean()
            )
        verse_df.insert(0, "average", chapter_avgs)

        alignment_by_verse[project_pair] = verse_df

        # Create book alignment ranking
        book_order_df = pd.DataFrame(columns=existing_books)
        book_order_df.index.name = f"{src_project}  {trg_project}"
        book_order_df.loc["Alignment Score"] = book_avgs
        book_order_df.loc["Verses in Common"] = [len(align_scores_df.loc[book].index) for book in existing_books]
        book_order_df = book_order_df.sort_values("Alignment Score", axis=1, ascending=False)
        book_order_df.loc["Cumulative Verses"] = [
            sum(book_order_df.loc["Verses in Common"].iloc[: i + 1]) for i in range(len(existing_books))
        ]
        book_order_df.loc["corpus_books", book_order_df.columns[0]] = ";".join(book_order_df.columns)
        book_orders[project_pair] = book_order_df

    LOGGER.info("Writing alignment breakdown file")
    with pd.ExcelWriter(config.exp_dir / f"{config.exp_dir.stem}_alignment_breakdown.xlsx") as writer:
        alignment_by_book.to_excel(writer, sheet_name="By Book", merge_cells=False)

        # Add Excel formatting
        workbook = writer.book
        int_format = workbook.add_format({"num_format": "0"})
        round4_format = workbook.add_format({"num_format": "0.0000"})
        book_sheet = writer.sheets["By Book"]
        book_sheet.autofit()
        book_sheet.set_column(2, len(alignment_by_book.columns) + 1, None, round4_format)
        book_sheet.freeze_panes(1, 2)
        book_order_sheet = workbook.add_worksheet("corpus_books")
        writer.sheets["corpus_books"] = book_order_sheet
        book_order_sheet.set_column(0, 0, max(19, max([3 + len(src) + len(trg) for src, trg in corpus_stats_df.index])))
        book_order_sheet.freeze_panes(0, 1)

        # Max # of sheets in an Excel spreadsheet is 255
        if len(alignment_by_book.index) > 250:
            LOGGER.warning(
                "Too many project pairs (>250); Alignment breakdowns beyond the book level will not be written."
            )
            return

        sheet_names = {}
        for i, project_pair in enumerate(tqdm(alignment_by_book.index)):
            # Add book order info to corpus_books sheet
            book_orders[project_pair].to_excel(writer, sheet_name="corpus_books", startrow=i * 6)
            book_order_sheet.set_row(i * 6 + 1, None, round4_format)
            book_order_sheet.set_row(i * 6 + 2, None, int_format)
            book_order_sheet.set_row(i * 6 + 3, None, int_format)

            # Handle duplicates caused by max length of sheet names (31)
            name = f"{project_pair[0][:14]}_{project_pair[1][:14]}"
            if name in sheet_names.keys():
                sheet_names[name] += 1
                name = f"{name}_{sheet_names[name]}"
            else:
                sheet_names[name] = 0

            # Add sheet with verse- and chapter-level scores
            alignment_by_verse[project_pair].to_excel(writer, sheet_name=name)
            verse_sheet = writer.sheets[name]
            verse_sheet.set_column(2, len(alignment_by_verse[project_pair].columns) + 1, None, round4_format)
            verse_sheet.freeze_panes(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verse counts and compute alignment scores")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--create-summaries", default=False, action="store_true", help="Create summary Excel files")
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

    if args.clearml_queue is not None and "cpu" not in args.clearml_queue:
        LOGGER.warning("Running this script on a GPU queue will not speed it up. Please only use CPU queues.")
        exit()
    clearml = SILClearML(exp_name, args.clearml_queue)

    SIL_NLP_ENV.copy_experiment_from_bucket(exp_name)

    config = clearml.config
    config.set_seed()

    # Confirm that input file paths exist and make sure every corpus pair is many to many
    data_files = set()
    for pair in config.corpus_pairs:
        data_files.update([f.path for f in pair.src_files] + [f.path for f in pair.trg_files])
    for file in data_files:
        if not file.is_file():
            LOGGER.error(f"The source file {str(file)} does not exist")
            return

    file_patterns = ";".join([f.name for f in data_files])
    collect_verse_counts(SIL_NLP_ENV.mt_scripture_dir, config.exp_dir, file_patterns, args.deutero, args.recalculate)
    SIL_NLP_ENV.copy_experiment_to_bucket(exp_name, "*_detailed_percentages.csv", overwrite=args.recalculate)

    get_corpus_stats(config, args.recalculate, args.deutero)

    # Add stats about projects in extra alignment files in the experiment folder
    extra_projects = get_extra_alignments(config, args.deutero)
    all_projects = set(extra_projects) | set([f.name for f in data_files])
    if len(all_projects) > len(data_files):
        LOGGER.info("Adding verse counts for projects in extra alignment files")
        collect_verse_counts(SIL_NLP_ENV.mt_scripture_dir, config.exp_dir, ";".join(all_projects), args.deutero)
        SIL_NLP_ENV.copy_experiment_to_bucket(exp_name, "*_detailed_percentages.csv", overwrite=args.recalculate)

    # Create summary outputs
    if args.create_summaries:
        create_alignment_breakdown_file(config, args.deutero)
        create_summary_file(config)

    patterns = [
        "verse_counts.csv",
        "verse_percentages.csv",
        "corpus-stats.csv",
        "*_alignment_breakdown.xlsx",
        "*_analysis.xlsx",
    ]
    SIL_NLP_ENV.copy_experiment_to_bucket(exp_name, patterns, overwrite=args.recalculate)


if __name__ == "__main__":
    main()
