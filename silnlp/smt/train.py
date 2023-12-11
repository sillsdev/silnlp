import argparse
import logging
from datetime import timedelta
from pathlib import Path
from time import perf_counter

from machine.corpora import TextFileTextCorpus
from machine.translation.thot import ThotSmtModelTrainer, ThotWordAlignmentModelType
from machine.utils import Phase, PhasedProgressReporter, ProgressStatus
from tqdm import tqdm

from ..common.environment import SIL_NLP_ENV
from ..common.utils import get_git_revision_hash, get_mt_exp_dir
from .config import create_word_tokenizer, get_thot_word_alignment_type, load_config

LOGGER = logging.getLogger(__package__ + ".train")


def create_thot_smt_config_file(config_file_path: Path, model_type: ThotWordAlignmentModelType) -> None:
    em_iters = 5
    if model_type is ThotWordAlignmentModelType.FAST_ALIGN:
        em_iters = 4
    config = f"""# Translation model prefix
-tm tm/src_trg

# Language model
-lm lm/trg.lm

# W parameter (maximum number of translation options to be considered per each source phrase)
-W 10

# S parameter (maximum number of hypotheses that can be stored in each stack)
-S 10

# A parameter (Maximum length in words of the source phrases to be translated)
-A 7

# Degree of non-monotonicity
-nomon 0

# Heuristic function used
-h 6

# Best-first search flag
-be

# Translation model weights
-tmw 0 0.5 1 1 1 1 0 1

# Set online learning parameters (ol_alg, lr_policy, l_stepsize, em_iters, e_par, r_par)
-olp 0 0 1 {em_iters} 1 0
"""

    config_file_path.write_text(config, encoding="utf-8")


def train(exp_name: str) -> None:
    exp_dir = get_mt_exp_dir(exp_name)
    SIL_NLP_ENV.copy_experiment_from_bucket(exp_name)
    config = load_config(exp_name)

    src_file_path = exp_dir / "train.src.txt"
    trg_file_path = exp_dir / "train.trg.txt"
    engine_dir = exp_dir / "engine"

    model_type = get_thot_word_alignment_type(config["model"])
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_config_file_path = engine_dir / "smt.cfg"
    create_thot_smt_config_file(engine_config_file_path, model_type)

    src_corpus = (
        TextFileTextCorpus(src_file_path).tokenize(create_word_tokenizer(config["src_tokenizer"])).unescape_spaces()
    )
    trg_corpus = (
        TextFileTextCorpus(trg_file_path).tokenize(create_word_tokenizer(config["trg_tokenizer"])).unescape_spaces()
    )
    parallel_corpus = src_corpus.align_rows(trg_corpus).lowercase()

    with ThotSmtModelTrainer(
        model_type,
        parallel_corpus,
        engine_config_file_path,
        lowercase_source=True,
        lowercase_target=True,
    ) as trainer, tqdm(total=1.0, bar_format="{percentage:3.0f}%|{bar:40}|{desc}", leave=False) as pbar:

        def progress(status: ProgressStatus) -> None:
            pbar.update(status.percent_completed - pbar.n)
            pbar.set_description_str(status.message)

        start = perf_counter()
        reporter = PhasedProgressReporter(progress, [Phase("Training model", 0.99), Phase("Saving model")])
        with reporter.start_next_phase() as phase_progress:
            trainer.train(phase_progress)
        with reporter.start_next_phase():
            trainer.save()
        end = perf_counter()

        LOGGER.info(f"Execution time: {timedelta(seconds=end - start)}")
        LOGGER.info(f"# of Segments Trained: {trainer.stats.train_corpus_size}")
        perplexity = trainer.stats.metrics["perplexity"]
        LOGGER.info(f"LM Perplexity: {perplexity:.4f}")
        bleu = trainer.stats.metrics["bleu"] * 100
        LOGGER.info(f"TM BLEU: {bleu:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an SMT model using the Machine library")
    parser.add_argument("experiments", nargs="+", help="Experiment names")
    args = parser.parse_args()

    get_git_revision_hash()

    for exp_name in args.experiments:
        LOGGER.info(f"Training {exp_name}")
        train(exp_name)
        LOGGER.info(f"Finished training {exp_name}")


if __name__ == "__main__":
    main()
