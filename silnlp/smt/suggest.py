import argparse
from io import TextIOWrapper
import logging
import os.path
from pathlib import Path
import time
from typing import List, Optional

from machine.corpora import ParallelTextCorpus
from machine.translation import (
    PhraseTranslationSuggester,
    InteractiveTranslatorFactory,
    InteractiveTranslator,
    TranslationSuggestion,
)
from machine.translation.thot import ThotSmtModel, ThotWordAlignmentModelType

from ..common.corpus import get_scripture_parallel_corpus
from ..common.environment import SIL_NLP_ENV
from ..common.utils import get_git_revision_hash, get_mt_exp_dir
from .config import load_config, create_word_tokenizer, create_word_detokenizer

LOGGER = logging.getLogger(__package__ + ".suggest")


def suggest(
    model_path: Path,
    src: Path,
    trg: Path,
    aligner: str,
    source_tokenizer: str,
    target_tokenizer: str,
    confidence: float,
    num_suggestions: int,
    trace: Optional[str],
    approve_aligned: bool,
    quiet: bool,
):
    action_count = 0
    char_count = 0
    total_accepted_suggestion_count = 0
    total_suggestion_count = 0
    full_suggestion_count = 0
    init_suggestion_count = 0
    final_suggestion_count = 0
    middle_suggestion_count = 0
    accepted_suggestion_counts = [0] * num_suggestions

    trace_file = None
    if trace is not None:
        trace_file = open(trace, "w", encoding="utf-8")

    suggester = PhraseTranslationSuggester(confidence)

    if not quiet:
        LOGGER.info("Loading model...")

    source_tokenizer = create_word_tokenizer(source_tokenizer)
    target_detokenizer = create_word_detokenizer(target_tokenizer)
    target_tokenizer = create_word_tokenizer(target_tokenizer)

    model = ThotSmtModel(ThotWordAlignmentModelType[aligner.upper()], model_path / "smt.cfg")

    if not quiet:
        LOGGER.info("done.")
        LOGGER.info("Suggesting...")
    start = time.time()

    corpus_df = get_scripture_parallel_corpus(src, trg)
    corpus = ParallelTextCorpus.from_pandas(corpus_df, None, "vref", "source", "target", None)
    corpus = corpus.tokenize(source_tokenizer, target_tokenizer)
    corpus = corpus.escape_spaces()
    corpus = corpus.lowercase()

    translator_factory = InteractiveTranslatorFactory(model, target_tokenizer, target_detokenizer)

    for segment_count, row in enumerate(corpus):
        if trace_file is not None:
            trace_file.write(f"Segment: {row.ref}\n")
            trace_file.write(f"Source: {row.source_text}\n")
            trace_file.write(f"Target: {row.target_text}\n")
            trace_file.write("=" * 120 + "\n")

        prev_suggestion_words: List[List[str]] = None
        is_last_word_suggestion = False
        suggestion_result = None

        translator: InteractiveTranslator = translator_factory.create(row.source_text)
        while len(translator.prefix_word_ranges) < len(row.target_segment) or not translator.is_last_word_complete:
            target_index = len(translator.prefix_word_ranges)
            if not translator.is_last_word_complete:
                target_index -= 1

            match = False
            suggestions: List[TranslationSuggestion] = suggester.get_suggestions(
                num_suggestions,
                len(translator.prefix_word_ranges),
                translator.is_last_word_complete,
                translator.get_current_results(),
            )

            suggestion_words: List[List[str]] = [list(s.target_words) for s in suggestions]
            if prev_suggestion_words == None or not suggestions_are_equal(prev_suggestion_words, suggestion_words):
                write_prefix(trace_file, suggestion_result, translator.prefix)
                write_suggestions(trace_file, suggestions)
                suggestion_result = None

                if any(len(s.target_word_indices) > 0 for s in suggestions):
                    total_suggestion_count += 1

            for k, suggestion in enumerate(suggestions):
                accepted = []
                j = target_index
                for i in range(len(suggestion_words[k])):
                    if j >= len(row.target_segment):
                        break

                    if suggestion_words[k][i] == row.target_segment[j]:
                        accepted.append(suggestion.target_word_indices[i])
                        j += 1
                    elif len(accepted) == 0:
                        j = target_index
                    else:
                        break

                if len(accepted) > 0:
                    translator.append_to_prefix(" ".join([suggestion.result.target_tokens[j] for j in accepted]) + " ")
                    is_last_word_suggestion = True
                    action_count += 1
                    total_accepted_suggestion_count += 1

                    if len(accepted) == len(suggestion.target_word_indices):
                        suggestion_result = "ACCEPT_FULL"
                        full_suggestion_count += 1
                    elif accepted[0] == suggestion.target_word_indices[0]:
                        suggestion_result = "ACCEPT_INIT"
                        init_suggestion_count += 1
                    elif accepted[-1] == suggestion.target_word_indices[-1]:
                        suggestion_result = "ACCEPT_FIN"
                        final_suggestion_count += 1
                    else:
                        suggestion_result = "ACCEPT_MID"
                        middle_suggestion_count += 1

                    accepted_suggestion_counts[k] += 1
                    match = True
                    break

            if not match:
                if is_last_word_suggestion:
                    action_count += 1
                    is_last_word_suggestion = False
                    write_prefix(trace_file, suggestion_result, translator.prefix)
                    suggestion_result = None

                length = 0 if translator.is_last_word_complete else len(translator.prefix_word_ranges[-1])
                target_word = row.target_segment[target_index]
                if length == len(target_word):
                    translator.append_to_prefix(" ")
                elif length + 1 == len(target_word):  # word will be complete after added character
                    translator.append_to_prefix(target_word[length : length + 1] + " ")
                else:
                    translator.append_to_prefix(target_word[length : length + 1])

                suggestion_result = "REJECT" if any(len(s.target_word_indices) > 0 for s in suggestions) else "NONE"
                action_count += 1

            prev_suggestion_words = suggestion_words

        write_prefix(trace_file, suggestion_result, translator.prefix)

        translator.approve(approve_aligned)

        char_count += sum(len(w) + 1 for w in row.target_segment)

        if trace_file is not None:
            trace_file.write("\n")

    if not quiet:
        LOGGER.info("done.")

    if trace_file is not None:
        trace_file.close()

    # Print stats
    LOGGER.info(f"Execution time: {time.time() - start}s")
    LOGGER.info(f"# of Segments: {segment_count + 1}")
    LOGGER.info(f"# of Suggestions: {total_suggestion_count}")
    LOGGER.info(f"# of Correct Suggestions: {total_accepted_suggestion_count}")
    LOGGER.info("Correct Suggestion Types")
    full_pcnt = 0 if full_suggestion_count == 0 else float(full_suggestion_count) / total_accepted_suggestion_count
    LOGGER.info(f"- Full: {full_pcnt:.4f}")
    init_pcnt = 0 if init_suggestion_count == 0 else float(init_suggestion_count) / total_accepted_suggestion_count
    LOGGER.info(f"- Initial: {init_pcnt:.4f}")
    final_pcnt = 0 if final_suggestion_count == 0 else float(final_suggestion_count) / total_accepted_suggestion_count
    LOGGER.info(f"- Final: {final_pcnt:.4f}")
    middle_pcnt = (
        0 if middle_suggestion_count == 0 else float(middle_suggestion_count) / total_accepted_suggestion_count
    )
    LOGGER.info(f"- Middle: {middle_pcnt:.4f}")
    LOGGER.info("Correct Suggestion N")
    for i in range(len(accepted_suggestion_counts)):
        pcnt = (
            0
            if accepted_suggestion_counts[i] == 0
            else float(accepted_suggestion_counts[i]) / total_accepted_suggestion_count
        )
        LOGGER.info(f"- {i+1}: {pcnt:.4f}")
    ksmr = 0 if action_count == 0 else float(action_count) / char_count
    LOGGER.info(f"KSMR: {ksmr:.4f}")
    precision = (
        0 if total_accepted_suggestion_count == 0 else float(total_accepted_suggestion_count) / total_suggestion_count
    )
    LOGGER.info(f"Precision: {precision:.4f}")


def write_prefix(trace_file: TextIOWrapper, suggestion_result: str, prefix: str):
    if trace_file == None or suggestion_result == None:
        return

    trace_file.write(f"- {suggestion_result}\n")
    trace_file.write(f"{prefix}\n")


def write_suggestions(trace_file: TextIOWrapper, suggestions: List[TranslationSuggestion]):
    if trace_file == None:
        return

    for i, suggestion in enumerate(suggestions):
        in_suggestion = False
        trace_file.write(f"SUGGESTION {i+1} ")

        for j, token in enumerate(suggestion.result.target_tokens):
            if j in suggestion.target_word_indices:
                if j > 0:
                    trace_file.write(" ")
                if not in_suggestion:
                    trace_file.write("[")
                in_suggestion = True
            elif in_suggestion:
                trace_file.write("] ")
                in_suggestion = False
            elif j > 0:
                trace_file.write(" ")

            trace_file.write(token)

        if in_suggestion:
            trace_file.write("]")
        trace_file.write("\n")


def suggestions_are_equal(x: List[List[str]], y: List[List[str]]):
    if len(x) != len(y):
        return False

    for i in range(len(x)):
        if len(x[i]) != len(y[i]):
            return False

        for j in range(len(x[i])):
            if x[i][j] != y[i][j]:
                return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulates the generation of translation suggestions during an interactive translation session."
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("-c", "--confidence", default=0.2, type=float, help="Confidence threshold")
    parser.add_argument("-n", default=1, type=int, help="Number of suggestions to generate")
    parser.add_argument("-t", "--trace", default=None, type=str, help="Trace file")
    parser.add_argument(
        "-aa",
        "--approve-aligned",
        default=False,
        action="store_true",
        help="Approve aligned part of source segment",
    )
    parser.add_argument("-q", "--quiet", default=False, action="store_true", help="Only display results")
    args = parser.parse_args()

    get_git_revision_hash()

    exp_name = args.experiment
    exp_dir = get_mt_exp_dir(exp_name)
    SIL_NLP_ENV.copy_experiment_from_bucket(exp_name)
    config = load_config(exp_name)

    engine_dir = exp_dir / f"engine{os.sep}"
    src_file_path = exp_dir / "test.src.txt"
    trg_file_path = exp_dir / "test.trg.txt"

    suggest(
        engine_dir,
        src_file_path,
        trg_file_path,
        config["model"],
        config["src_tokenizer"],
        config["trg_tokenizer"],
        args.confidence,
        args.n,
        args.trace,
        args.approve_aligned,
        args.quiet,
    )


if __name__ == "__main__":
    main()
