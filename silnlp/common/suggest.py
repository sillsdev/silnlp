import argparse
from io import TextIOWrapper
import os.path
from pathlib import Path
import time
from typing import List

from machine.corpora import ParallelTextCorpus
from machine.tokenization import (
    WhitespaceTokenizer,
    WhitespaceDetokenizer,
    LatinWordTokenizer,
    LatinWordDetokenizer,
    NullTokenizer,
    ZwspWordTokenizer,
    ZwspWordDetokenizer,
)
from machine.translation import (
    PhraseTranslationSuggester,
    InteractiveTranslatorFactory,
    InteractiveTranslator,
    TranslationSuggestion,
)
from machine.translation.thot import ThotSmtModel, ThotWordAlignmentModelType

from .corpus import get_scripture_parallel_corpus

NORMALIZATION_FORMS = ["nfc, nfd, nfkc, nfkd"]


def get_tokenizer(tokenizer: str):
    if tokenizer is None:
        return NullTokenizer()
    elif tokenizer == "whitespace":
        return WhitespaceTokenizer()
    elif tokenizer == "latin":
        return LatinWordTokenizer()
    elif tokenizer == "zwsp":
        return ZwspWordTokenizer()
    else:
        raise ValueError(f"Invalid tokenizer {tokenizer}")


def get_detokenizer(detokenizer: str):
    if detokenizer is None:
        return WhitespaceDetokenizer()
    elif detokenizer == "whitespace":
        return WhitespaceDetokenizer()
    elif detokenizer == "latin":
        return LatinWordDetokenizer()
    elif detokenizer == "zwsp":
        return ZwspWordDetokenizer()
    else:
        raise ValueError(f"Invalid detokenizer {detokenizer}")


def suggest(
    model_path: str,
    src: str,
    trg: str,
    aligner: str,
    source_tokenizer: str,
    target_tokenizer: str,
    confidence: float,
    num_suggestions: int,
    trace: str,
    approve_aligned: bool,
    normalization_form: str,
    escape_spaces: bool,
    cased: bool,
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

    if os.path.isfile(trace):
        trace_file = None
    else:
        trace_file = open(trace, "w", encoding="utf-8")

    suggester = PhraseTranslationSuggester(confidence)

    if not quiet:
        print("Loading model...")

    source_tokenizer = get_tokenizer(source_tokenizer)
    target_detokenizer = get_detokenizer(target_tokenizer)
    target_tokenizer = get_tokenizer(target_tokenizer)

    model = ThotSmtModel(
        ThotWordAlignmentModelType[aligner.upper()],
        Path(model_path) / "smt.cfg",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        target_detokenizer=target_detokenizer,
        lowercase_source=(not cased),
        lowercase_target=(not cased),
    )

    if not quiet:
        print("done.")
        print("Suggesting...")
    start = time.time()

    corpus_df = get_scripture_parallel_corpus(Path(src), Path(trg))
    corpus = ParallelTextCorpus.from_pandas(corpus_df, None, "vref", "source", "target", None)
    corpus = corpus.tokenize(source_tokenizer, target_tokenizer)
    if normalization_form is not None and normalization_form in NORMALIZATION_FORMS:
        corpus = corpus.normalize(normalization_form)
    if escape_spaces:
        corpus = corpus.escape_spaces()
    if not cased:
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
        print("done.")

    if trace_file is not None:
        trace_file.close()

    # Print stats
    print(f"Execution time: {time.time() - start}s")
    print(f"# of Segments: {segment_count + 1}")
    print(f"# of Suggestions: {total_suggestion_count}")
    print(f"# of Correct Suggestions: {total_accepted_suggestion_count}")
    print("Correct Suggestion Types")
    full_pcnt = 0 if full_suggestion_count == 0 else float(full_suggestion_count) / total_accepted_suggestion_count
    print(f"- Full: {full_pcnt:.4f}")
    init_pcnt = 0 if init_suggestion_count == 0 else float(init_suggestion_count) / total_accepted_suggestion_count
    print(f"- Initial: {init_pcnt:.4f}")
    final_pcnt = 0 if final_suggestion_count == 0 else float(final_suggestion_count) / total_accepted_suggestion_count
    print(f"- Final: {final_pcnt:.4f}")
    middle_pcnt = (
        0 if middle_suggestion_count == 0 else float(middle_suggestion_count) / total_accepted_suggestion_count
    )
    print(f"- Middle: {middle_pcnt:.4f}")
    print("Correct Suggestion N")
    for i in range(len(accepted_suggestion_counts)):
        pcnt = (
            0
            if accepted_suggestion_counts[i] == 0
            else float(accepted_suggestion_counts[i]) / total_accepted_suggestion_count
        )
        print(f"- {i+1}: {pcnt:.4f}")
    ksmr = 0 if action_count == 0 else float(action_count) / char_count
    print(f"KSMR: {ksmr:.4f}")
    precision = (
        0 if total_accepted_suggestion_count == 0 else float(total_accepted_suggestion_count) / total_suggestion_count
    )
    print(f"Precision: {precision:.4f}")


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
    parser.add_argument("model", help="Path to model")
    parser.add_argument("src", type=str, help="Source corpus")
    parser.add_argument("trg", type=str, help="Target corpus")
    parser.add_argument("aligner", default=None, type=str, help="The word alignment model type")
    parser.add_argument("-st", "--source-tokenizer", default=None, type=str, help="Source tokenizer")
    parser.add_argument("-tt", "--target-tokenizer", default=None, type=str, help="Target tokenizer")
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
    parser.add_argument(
        "-nf",
        "--normalization-form",
        default=None,
        type=str,
        help="Normalizes text to the specified form. Forms: nfc, nfd, nfkc, nfkd",
    )
    parser.add_argument("--escape-spaces", default=False, action="store_true", help="Escape spaces")
    parser.add_argument("--cased", default=False, action="store_true", help="Do not make corpus lowercase")
    parser.add_argument("-q", "--quiet", default=False, action="store_true", help="Only display results")
    args = parser.parse_args()

    suggest(
        args.model,
        args.src,
        args.trg,
        args.aligner,
        args.source_tokenizer,
        args.target_tokenizer,
        args.confidence,
        args.n,
        args.trace,
        args.approve_aligned,
        args.normalization_form,
        args.escape_spaces,
        args.cased,
        args.quiet,
    )


if __name__ == "__main__":
    main()
