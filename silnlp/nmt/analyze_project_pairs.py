import argparse
import logging
from statistics import mean

import pandas as pd
from machine.scripture import is_ot_nt

from ..alignment.config import get_aligner_name
from ..alignment.utils import add_alignment_scores
from ..common.collect_verse_counts import collect_verse_counts
from ..common.corpus import exclude_chapters, filter_parallel_corpus, get_scripture_parallel_corpus, include_chapters
from ..common.environment import SIL_NLP_ENV
from ..common.script_utils import get_script, is_represented
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import Config, get_data_file_pairs
from .config_utils import load_config

LOGGER = logging.getLogger(__package__ + ".analyze_project_pairs")


def preprocess_stats(config: Config, force_align: bool = False, deutero: bool = False):
    stats_path = config.exp_dir / "corpus-stats.csv"
    if stats_path.is_file():
        stats_df = pd.read_csv(stats_path, dtype=str, keep_default_na=False).set_index(["src_project", "trg_project"])
    else:
        stats_df = pd.DataFrame(
            columns=[
                "src_project",
                "trg_project",
                "count",
                "parallel",
                "src_only",
                "trg_only",
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
    stats_df.sort_index(inplace=True)

    for pair in config.corpus_pairs:
        for src_file, trg_file in get_data_file_pairs(pair):
            project_pair = (f"{src_file.iso}-{src_file.project}", f"{trg_file.iso}-{trg_file.project}")
            corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path, False)
            corpus = corpus.loc[(corpus["source"].str.len() > 0) | (corpus["target"].str.len() > 0)]
            if not deutero:
                corpus = corpus.loc[[is_ot_nt(vref.book_num) for vref in corpus["vref"]]]

            if len(pair.corpus_books) > 0:
                corpus = include_chapters(corpus, pair.corpus_books)

            # Align pair
            pair_stats_path = config.exp_dir / f"{project_pair[0]}_{project_pair[1]}.csv"
            if pair_stats_path.is_file() and not force_align:
                LOGGER.info(f"Using pre-existing alignment scores from {pair_stats_path}")
                pair_stats = pd.read_csv(pair_stats_path)
                pair_stats["idx"] = corpus.index
                pair_stats.set_index("idx", inplace=True)
                corpus["score"] = pair_stats["score"]
            else:
                aligner_id = config.data["aligner"]
                LOGGER.info(f"Computing alignment scores using {get_aligner_name(aligner_id)}")
                add_alignment_scores(corpus, aligner_id)
                corpus.to_csv(pair_stats_path, index=False)

            # Compute stats
            if project_pair not in stats_df.index or force_align:
                pair_count = len(corpus.index)
                src_only_count = 0
                trg_only_count = 0
                for _, row in corpus.iterrows():
                    src_only_count += len(row["target"]) == 0
                    trg_only_count += len(row["source"]) == 0
                parallel_count = pair_count - src_only_count - trg_only_count

                alignment_score = corpus["score"].mean()
                filtered_count = 0
                filtered_alignment_score = alignment_score
                if pair.score_threshold > 0:
                    unfiltered_len = len(corpus)
                    corpus = filter_parallel_corpus(corpus, pair.score_threshold)
                    filtered_count = unfiltered_len - len(corpus)
                    filtered_alignment_score = mean(corpus["score"])

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
                    parallel_count,
                    src_only_count,
                    trg_only_count,
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
                f" parallel count: {row.at['parallel']}"
                f" source only count: {row.at['src_only']}"
                f" target only count: {row.at['trg_only']}"
                f" alignment: {row.at['align_score']},"
                f" filtered count: {row.at['filtered_count']},"
                f" alignment (filtered): {row.at['filtered_align_score']},"
                f" source script: {row.at['src_script']}, source script in model: {row.at['src_script_in_model']},"
                f" target script: {row.at['trg_script']}, target script in model: {row.at['trg_script_in_model']}"
            )
    stats_df.to_csv(stats_path, index=True)


def aggregate_data(config: Config):
    pass


def aggregate_alignments(config: Config):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verse counts and compute alignment scores")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--force-align", default=False, action="store_true", help="Force recalculation of all alignment scores"
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
    clearml = SILClearML(exp_name, args.clearml_queue)

    config = clearml.config
    config.set_seed()

    # Confirm that input file paths exist
    for file in config.src_file_paths | config.trg_file_paths:
        if not file.is_file():
            LOGGER.error(f"The source file {str(file)} does not exist.")
            return

    # TODO: use caching
    collect_verse_counts(SIL_NLP_ENV.mt_scripture_dir, config.exp_dir, args.deutero)

    preprocess_stats(config, args.force_align, args.deutero)

    """
    at this point there will be these files:
    - detailed_percentages.csv for each project w/ incomplete books: verse percentages per chapter for each incomplete book (>0 verses)
    - verse_counts.csv: verse counts per book
    - verse_percentages.csv: verse percentages per book
    - corpus-stats.csv
    - alignment files for each pair
    """

    # aggregate info into single excel file
    aggregate_data(config)

    # create excel file with alignment scores at each level
    aggregate_alignments(config)

    SIL_NLP_ENV.copy_experiment_to_bucket(exp_name)


if __name__ == "__main__":
    main()
