"""Score inference output produced outside silnlp using the exact same sacrebleu calls and
parameters as ``silnlp.nmt.test`` does during the ``test`` stage.

This module is deliberately self-contained: it duplicates the sacrebleu calls in
``silnlp/nmt/test.py`` (``score_pair`` / ``write_pair_verse_scores``) rather than importing
them, so that running ``silnlp-nmt-test`` is never affected by changes here. Every metric call
below carries a comment pointing at the line(s) in ``test.py`` it mirrors, and
``tests/unit_tests/test_score.py`` asserts the two produce identical numbers on a fixed corpus. If you
change a metric call here, change the drift test too - that test is what keeps this file honest.

Known, deliberate differences from ``test.py``:

* silnlp normalizes reference text at *preprocess* time (Moses punctuation normalization plus
  the HuggingFace backend tokenizer's own normalization, see ``config.py``). References read
  from silnlp's own ``test.trg.detok*.txt`` files are already normalized, so scoring them here
  is exact. References from any other source are not, and scores may differ slightly for
  punctuation-heavy text. This script does not attempt to re-implement that normalization.
* The seeded, random-reference-per-line reconstruction silnlp performs when several reference
  projects exist and none are named (``test.py:510-519``) is not reproduced, since it depends on
  ``config.set_seed()`` and is not reproducible outside a live experiment. Pass explicit
  reference files/columns instead.
* Two bugs in silnlp's own ``--by-book`` path are not reproduced: the confidences file is not
  unconditionally required for per-book scoring here, and the random-reference-per-book indexing
  off-by-three (``test.py:473-475``) has no equivalent since random-reference mode isn't
  supported here at all.
"""

import argparse
import csv
import importlib.metadata
import logging
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, TextIO, Tuple

import sacrebleu
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, book_number_to_id, get_chapters
from sacrebleu.metrics import BLEUScore
from scipy.stats import gmean

LOGGER = logging.getLogger((__package__ or "") + ".score")

logging.getLogger("sacrebleu").setLevel(logging.ERROR)

# Pinned in poetry.lock; scores are only guaranteed to match silnlp at these versions.
EXPECTED_SACREBLEU_VERSION = "2.6.0"
EXPECTED_SIL_MACHINE_VERSION = "1.8.11"

SUPPORTED_SCORERS = [
    "bleu",
    "chrf3",
    "chrf3+",
    "chrf3++",
    "spbleu",
    "m-bleu",
    "m-chrf3",
    "m-chrf3+",
    "m-chrf3++",
    "ter",
    "confidence",
]

SUPPORTED_SENTENCE_SCORERS = [
    "bleu",
    "chrf3",
    "chrf3+",
    "chrf3++",
    "spbleu",
    "ter",
    "confidence",
]

# silnlp/nmt/experiment.py:234-245's default scorer set. Note this is NOT the same as
# silnlp/nmt/test.py's own default when run standalone with no --scorers, which is {"bleu"}
# alone (test.py:788-789) - there is no single "silnlp default", it depends on entry point.
DEFAULT_SCORERS = {"bleu", "chrf3", "chrf3+", "chrf3++", "spbleu", "m-bleu", "m-chrf3", "m-chrf3+", "m-chrf3++"}

CONFIDENCE_SUFFIX = ".confidences.tsv"
VERSE_SCORES_SUFFIX = ".scores.tsv"

_LANG_CODE_RE = re.compile(r"^[a-z]{2,3}_[A-Z][a-z]{3}\b")


# Verbatim port of silnlp/nmt/test.py::PairScore - same columns, same rounding, same method
# name (writeHeader, not write_header) so this class diffs cleanly against test.py's.
class PairScore:
    def __init__(
        self,
        book: str,
        src_iso: str,
        trg_iso: str,
        bleu: Optional[BLEUScore],
        sent_len: int,
        projects: Set[str],
        other_scores: Dict[str, float] = {},
        draft_index: int = 1,
    ) -> None:
        self.src_iso = src_iso
        self.trg_iso = trg_iso
        self.bleu = bleu
        self.sent_len = sent_len
        self.num_refs = len(projects)
        self.refs = "_".join(sorted(projects))
        self.other_scores = other_scores
        self.book = book
        self.draft_index = draft_index

    def writeHeader(self, file: TextIO) -> None:
        header = (
            "book,draft_index,src_iso,trg_iso,num_refs,references,sent_len"
            + (
                ",BLEU,BLEU_1gram_prec,BLEU_2gram_prec,BLEU_3gram_prec,BLEU_4gram_prec,BLEU_brevity_penalty,BLEU_total_sys_len,BLEU_total_ref_len"
                if self.bleu is not None
                else ""
            )
            + ("," if len(self.other_scores) > 0 else "")
            + ",".join(self.other_scores.keys())
            + "\n"
        )
        file.write(header)

    def write(self, file: TextIO) -> None:
        file.write(
            f"{self.book},{self.draft_index},{self.src_iso},{self.trg_iso},"
            f"{self.num_refs},{self.refs},{self.sent_len:d}"
        )
        if self.bleu is not None:
            file.write(
                f",{self.bleu.score:.2f},{self.bleu.precisions[0]:.2f},{self.bleu.precisions[1]:.2f}"
                f",{self.bleu.precisions[2]:.2f},{self.bleu.precisions[3]:.2f},{self.bleu.bp:.3f}"
                f",{self.bleu.sys_len:d},{self.bleu.ref_len:d}"
            )
        for scorer, val in self.other_scores.items():
            if scorer.lower() == "confidence":
                file.write(f",{val:.8f}")
            else:
                file.write(f",{val:.2f}")
        file.write("\n")


def check_versions(allow_mismatch: bool) -> None:
    actual_sacrebleu = sacrebleu.__version__
    try:
        actual_machine = importlib.metadata.version("sil-machine")
    except importlib.metadata.PackageNotFoundError:
        actual_machine = "unknown"

    mismatches = []
    if actual_sacrebleu != EXPECTED_SACREBLEU_VERSION:
        mismatches.append(f"sacrebleu {actual_sacrebleu} (silnlp's poetry.lock pins {EXPECTED_SACREBLEU_VERSION})")
    if actual_machine != EXPECTED_SIL_MACHINE_VERSION:
        mismatches.append(f"sil-machine {actual_machine} (silnlp's poetry.lock pins {EXPECTED_SIL_MACHINE_VERSION})")

    if mismatches:
        message = "Version mismatch - scores may not exactly match silnlp:\n  " + "\n  ".join(mismatches)
        if allow_mismatch:
            LOGGER.warning(message)
        else:
            raise SystemExit(message + "\n\nPass --allow-version-mismatch to score anyway.")
    LOGGER.info(f"sacrebleu {actual_sacrebleu}, sil-machine {actual_machine}")


def warn_if_spbleu_model_missing() -> None:
    import os

    sacrebleu_dir = os.environ.get("SACREBLEU", str(Path.home() / ".sacrebleu"))
    model_path = Path(sacrebleu_dir) / "models" / "flores200sacrebleuspm"
    if not model_path.exists():
        LOGGER.warning(
            f"FLORES-200 SPM model not found at {model_path}. sacrebleu will try to download it from "
            "https://tinyurl.com/flores200sacrebleuspm the first time spBLEU is computed - this needs "
            "network access."
        )


# Mirrors silnlp/nmt/test.py::score_pair's metric computation (test.py:127-236). The block
# structure and variable names below match test.py's so the two diff cleanly - if you change a
# metric call here, change the drift test in tests/unit_tests/test_score.py too.
def compute_scores(
    pair_sys: List[str],
    pair_refs: List[List[str]],
    scorers: Set[str],
    sacrebleu_tokenize: str = "13a",
    confidences: Optional[List[float]] = None,
) -> Tuple[Optional[BLEUScore], Dict[str, float]]:
    bleu_score = None
    if "bleu" in scorers:
        # test.py:129-134
        bleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize=sacrebleu_tokenize,
        )

    other_scores: Dict[str, float] = {}
    if "chrf3" in scorers:
        # test.py:138 - note: no eps_smoothing, unlike chrf3+/chrf3++ below
        chrf3_score = sacrebleu.corpus_chrf(pair_sys, pair_refs, char_order=6, beta=3, remove_whitespace=True)
        other_scores["chrF3"] = chrf3_score.score

    if "chrf3+" in scorers:
        # test.py:142-144
        chrfp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
        )
        other_scores["chrF3+"] = chrfp_score.score

    if "chrf3++" in scorers:
        # test.py:148-150
        chrfpp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
        )
        other_scores["chrF3++"] = chrfpp_score.score

    if "spbleu" in scorers:
        # test.py:154-159
        spbleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize="flores200",
        )
        other_scores["spBLEU"] = spbleu_score.score

    # m-bleu / m-chrf3(+/++): arithmetic mean of per-sentence scores, from
    # https://arxiv.org/pdf/2407.12832 (test.py:162-217). sentence_bleu defaults to
    # use_effective_order=True, so m-BLEU is not the same computation as corpus BLEU above.
    if "m-bleu" in scorers:
        sentence_bleu_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_bleu_score = sacrebleu.sentence_bleu(
                sentence,
                references,
                lowercase=True,
                tokenize=sacrebleu_tokenize,
            )
            sentence_bleu_scores.append(sentence_bleu_score.score)
        if len(sentence_bleu_scores) == 0:
            other_scores["m-BLEU"] = 0
        else:
            other_scores["m-BLEU"] = sum(sentence_bleu_scores) / len(sentence_bleu_scores)

    if "m-chrf3" in scorers:
        sentence_chrf3_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_chrf3_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, remove_whitespace=True
            )
            sentence_chrf3_scores.append(sentence_chrf3_score.score)
        if len(sentence_chrf3_scores) == 0:
            other_scores["m-chrf3"] = 0
        else:
            other_scores["m-chrf3"] = sum(sentence_chrf3_scores) / len(sentence_chrf3_scores)

    if "m-chrf3+" in scorers:
        sentence_chrfp_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_chrfp_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
            )
            sentence_chrfp_scores.append(sentence_chrfp_score.score)
        if len(sentence_chrfp_scores) == 0:
            other_scores["m-chrf3+"] = 0
        else:
            other_scores["m-chrf3+"] = sum(sentence_chrfp_scores) / len(sentence_chrfp_scores)

    if "m-chrf3++" in scorers:
        sentence_chrfpp_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_chrfpp_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
            )
            sentence_chrfpp_scores.append(sentence_chrfpp_score.score)
        if len(sentence_chrfpp_scores) == 0:
            other_scores["m-chrf3++"] = 0
        else:
            other_scores["m-chrf3++"] = sum(sentence_chrfpp_scores) / len(sentence_chrfpp_scores)

    if "ter" in scorers:
        # test.py:219-222 - all sacrebleu TER defaults (case-insensitive, not normalized)
        ter_score = sacrebleu.corpus_ter(pair_sys, pair_refs)
        if ter_score.score >= 0:
            other_scores["TER"] = ter_score.score

    if "confidence" in scorers:
        # test.py:224-236
        if confidences is None:
            raise ValueError(
                "The 'confidence' scorer requires per-sentence confidences: pass --conf/--conf-col, "
                "or use --exp-dir with a matching *.confidences.tsv file present."
            )
        other_scores["Confidence"] = gmean(confidences)

    return bleu_score, other_scores


# Mirrors silnlp/nmt/test.py::write_pair_verse_scores (test.py:254-364); block structure and
# variable names match test.py's so the two diff cleanly. Omits the confidence/chrF3 linear-
# regression side output, which is orthogonal to matching scores. Note (test.py:287): spbleu_metric
# is one shared BLEU(tokenize="flores200") object with effective_order=False, so per-verse spBLEU
# is 0 for any verse with no 4-gram match, unlike per-verse BLEU - this asymmetry is silnlp's own
# and is intentionally preserved here, not a bug in this port.
def write_verse_scores(
    path: Path,
    pair_sys: List[str],
    pair_refs: List[List[str]],
    scorers: Set[str],
    other_scores: Dict[str, float],
    sacrebleu_tokenize: str,
    confidences: Optional[List[float]],
) -> None:
    scorers = scorers.intersection(SUPPORTED_SENTENCE_SCORERS)
    other_scores = {k: v for k, v in other_scores.items() if k.lower() in scorers}

    with open(path, "w", encoding="utf-8", newline="") as scores_file:
        writer = csv.writer(scores_file, delimiter="\t")
        header = ["Verse"]
        if "bleu" in scorers:
            header += [
                "BLEU",
                "BLEU_1gram_prec",
                "BLEU_2gram_prec",
                "BLEU_3gram_prec",
                "BLEU_4gram_prec",
                "BLEU_brevity_penalty",
            ]
        header += list(other_scores.keys())
        header.append("Prediction")
        for _ in pair_refs:
            header.append("Reference")
        writer.writerow(header)
        spbleu_metric = sacrebleu.metrics.BLEU(tokenize="flores200", lowercase=True) if "spbleu" in scorers else None
        for index, pred in enumerate(pair_sys):
            sentences: List[str] = []
            for ref in pair_refs:
                sentences.append(ref[index])
            if "bleu" in scorers:
                bleu_verse_score = sacrebleu.sentence_bleu(
                    pred,
                    sentences,
                    lowercase=True,
                    tokenize=sacrebleu_tokenize,
                )
            other_verse_scores: Dict[str, float] = {}
            if "chrf3" in scorers:
                chrf3_verse_score = sacrebleu.sentence_chrf(
                    pred, sentences, char_order=6, beta=3, remove_whitespace=True
                )
                other_verse_scores["chrF3"] = chrf3_verse_score.score

            if "chrf3+" in scorers:
                chrfp_verse_score = sacrebleu.sentence_chrf(
                    pred, sentences, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
                )
                other_verse_scores["chrF3+"] = chrfp_verse_score.score

            if "chrf3++" in scorers:
                chrfpp_verse_score = sacrebleu.sentence_chrf(
                    pred, sentences, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
                )
                other_verse_scores["chrF3++"] = chrfpp_verse_score.score

            if "spbleu" in scorers and spbleu_metric is not None:
                spbleu_verse_score = spbleu_metric.sentence_score(pred, sentences)
                other_verse_scores["spBLEU"] = spbleu_verse_score.score

            if "ter" in scorers:
                ter_verse_score = sacrebleu.sentence_ter(pred, sentences)
                if ter_verse_score.score >= 0:
                    other_verse_scores["TER"] = ter_verse_score.score

            if "confidence" in scorers and confidences is not None:
                other_verse_scores["Confidence"] = confidences[index]

            row: List[str] = [f"{index + 1}"]

            if "bleu" in scorers:
                row += [
                    f"{bleu_verse_score.score:.2f}",
                    f"{bleu_verse_score.precisions[0]:.2f}",
                    f"{bleu_verse_score.precisions[1]:.2f}",
                    f"{bleu_verse_score.precisions[2]:.2f}",
                    f"{bleu_verse_score.precisions[3]:.2f}",
                    f"{bleu_verse_score.bp:.3f}",
                ]
            for scorer, val in other_verse_scores.items():
                if scorer.lower() == "confidence":
                    row.append(f"{val:.8f}")
                else:
                    row.append(f"{val:.2f}")

            row.append(pred.rstrip("\n"))
            for sentence in sentences:
                row.append(sentence.rstrip("\n"))
            writer.writerow(row)


class LoadedData(NamedTuple):
    hyp_lines: List[str]
    ref_lines_list: List[List[str]]
    vref_lines: Optional[List[str]]
    conf_list: Optional[List[float]]


def _read_all_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def _read_confidences(path: Path) -> List[float]:
    if path.name.endswith(CONFIDENCE_SUFFIX):
        # silnlp's native format: 3 header/legend rows, then alternating token-string /
        # token-score rows per sequence (test.py:230, test.py:448)
        with open(path, "r", encoding="utf-8") as f:
            return [float(line.split("\t")[0]) for line in list(f)[3::2]]
    with open(path, "r", encoding="utf-8") as f:
        return [float(line.strip()) for line in f if line.strip() != ""]


def _looks_tokenized(lines: Sequence[str]) -> Optional[str]:
    for line in lines[:20]:
        if "▁" in line or "</s>" in line:
            return line
        if _LANG_CODE_RE.match(line):
            return line
    return None


def _project_name(path: Path) -> Optional[str]:
    # silnlp/nmt/config.py::Config._parse_ref_file_path, single-pair case only
    parts = path.name.split(".")
    if len(parts) == 5:
        return parts[3]
    return None


def _reconcile_lengths(data: LoadedData) -> LoadedData:
    """Truncates every list to the shortest one, as zip() would, but warns first - outside a
    live silnlp run, mismatched lengths usually mean misaligned files rather than intent."""
    lengths = {len(data.hyp_lines)} | {len(r) for r in data.ref_lines_list}
    if data.vref_lines is not None:
        lengths.add(len(data.vref_lines))
    if data.conf_list is not None:
        lengths.add(len(data.conf_list))
    if len(lengths) == 1:
        return data

    n = min(lengths)
    LOGGER.warning(f"Input files have different line counts {sorted(lengths)}; truncating to the shortest ({n}).")
    return LoadedData(
        data.hyp_lines[:n],
        [r[:n] for r in data.ref_lines_list],
        data.vref_lines[:n] if data.vref_lines is not None else None,
        data.conf_list[:n] if data.conf_list is not None else None,
    )


def load_text_mode(args: argparse.Namespace) -> LoadedData:
    if not args.ref:
        raise SystemExit("--ref is required in text mode (one or more reference files).")
    hyp_lines = _read_all_lines(Path(args.hyp))
    ref_lines_list = [_read_all_lines(Path(p)) for p in args.ref]
    vref_lines = _read_all_lines(Path(args.vref)) if args.vref else None
    conf_list = _read_confidences(Path(args.conf)) if args.conf else None
    return _reconcile_lengths(LoadedData(hyp_lines, ref_lines_list, vref_lines, conf_list))


def load_table_mode(args: argparse.Namespace) -> LoadedData:
    import pandas as pd

    if not args.hyp_col:
        raise SystemExit("--hyp-col is required in table mode.")
    if not args.ref_col:
        raise SystemExit("--ref-col is required in table mode (one or more columns).")

    sep = "\t" if Path(args.hyp).suffix.lower() == ".tsv" else ","
    df = pd.read_csv(args.hyp, sep=sep, dtype=str, keep_default_na=False)

    for col in [args.hyp_col, *args.ref_col, args.vref_col, args.conf_col]:
        if col is not None and col not in df.columns:
            raise SystemExit(f"Column {col!r} not found in {args.hyp}. Available columns: {list(df.columns)}")

    hyp_lines = [v.strip() for v in df[args.hyp_col].tolist()]
    ref_lines_list = [[v.strip() for v in df[col].tolist()] for col in args.ref_col]
    vref_lines = [v.strip() for v in df[args.vref_col].tolist()] if args.vref_col else None
    conf_list = [float(v) for v in df[args.conf_col].tolist()] if args.conf_col else None

    return LoadedData(hyp_lines, ref_lines_list, vref_lines, conf_list)


def load_exp_dir_mode(args: argparse.Namespace) -> Tuple[LoadedData, str, Set[str]]:
    exp_dir = Path(args.exp_dir)
    if not (exp_dir / "test.src.txt").is_file():
        raise SystemExit(
            f"{exp_dir} does not have a single test.src.txt - it looks like a multi-language-pair "
            "experiment directory, which --exp-dir discovery doesn't support. Use --hyp/--ref (text "
            "mode) or --hyp/--hyp-col (table mode) instead."
        )

    step_token = args.step
    hyp_path = exp_dir / f"test.trg-predictions.detok.txt.{step_token}"
    if not hyp_path.is_file():
        candidates = sorted(p.name for p in exp_dir.glob("test.trg-predictions.detok.txt.*"))
        raise SystemExit(f"{hyp_path} not found. Available prediction files: {candidates or '(none)'}")

    ref_paths = sorted(exp_dir.glob("test.trg.detok*.txt"))
    if len(ref_paths) == 0:
        raise SystemExit(f"No reference files matching test.trg.detok*.txt found in {exp_dir}.")

    ref_projects = set(args.ref_projects)
    if len(ref_paths) > 1:
        if not ref_projects:
            raise SystemExit(
                f"Multiple reference files found ({[p.name for p in ref_paths]}) and no --ref-projects "
                "given. silnlp would build a single reference by randomly sampling a project per line "
                "using its own training seed (test.py:510-519), which this script cannot reproduce. "
                "Pass --ref-projects to select specific reference projects explicitly."
            )
        ref_paths = [p for p in ref_paths if _project_name(p) in ref_projects]
        if not ref_paths:
            raise SystemExit(f"None of --ref-projects {sorted(ref_projects)} matched reference files in {exp_dir}.")

    vref_path = exp_dir / "test.vref.txt"
    vref_lines = _read_all_lines(vref_path) if vref_path.is_file() else None

    conf_path = exp_dir / f"test.trg-predictions.txt.{step_token}{CONFIDENCE_SUFFIX}"
    conf_list = _read_confidences(conf_path) if conf_path.is_file() else None

    data = _reconcile_lengths(
        LoadedData(
            _read_all_lines(hyp_path),
            [_read_all_lines(p) for p in ref_paths],
            vref_lines,
            conf_list,
        )
    )
    return data, step_token, ref_projects


def _filter_by_books(
    hyp_lines: List[str],
    ref_lines_list: List[List[str]],
    vref_lines: Optional[List[str]],
    conf_list: Optional[List[float]],
    books: Dict[int, List[int]],
) -> Tuple[List[str], List[List[str]], Optional[List[float]]]:
    """Mirrors test.py:527-533: filtering only happens when both a book selection and a vref
    file are present; a blank vref line is kept regardless of the filter. main() already
    rejects --books with no vref before calling this, so the vref_lines is None branch here
    is just a defensive no-op for direct callers (e.g. tests)."""
    if len(books) == 0 or vref_lines is None:
        return hyp_lines, ref_lines_list, conf_list

    keep: List[int] = []
    for i in range(len(hyp_lines)):
        vref_line = vref_lines[i]
        if vref_line != "":
            vref = VerseRef.from_string(vref_line, ORIGINAL_VERSIFICATION)
            if vref.book_num not in books:
                continue
        keep.append(i)

    out_sys = [hyp_lines[i] for i in keep]
    out_refs = [[ref_lines[i] for i in keep] for ref_lines in ref_lines_list]
    out_conf = [conf_list[i] for i in keep] if conf_list is not None else None
    return out_sys, out_refs, out_conf


def _build_book_dict(
    hyp_lines: List[str],
    ref_lines_list: List[List[str]],
    vref_lines: List[str],
    conf_list: Optional[List[float]],
    books: Dict[int, List[int]],
) -> Dict[str, Tuple[List[str], List[List[str]], List[float]]]:
    """Mirrors test.py::process_individual_books (test.py:428-486), without the unconditional
    confidences-file requirement and without random-reference support (both intentional -
    see the module docstring)."""
    book_dict: Dict[str, Tuple[List[str], List[List[str]], List[float]]] = {}
    for i, pred_line in enumerate(hyp_lines):
        vref_line = vref_lines[i]
        if vref_line == "":
            continue
        vref = VerseRef.from_string(vref_line, ORIGINAL_VERSIFICATION)
        if len(books) > 0 and vref.book_num not in books:
            continue

        if vref.book not in book_dict:
            book_dict[vref.book] = ([], [[] for _ in ref_lines_list], [])
        book_pred, book_refs, book_conf = book_dict[vref.book]
        book_pred.append(pred_line)
        for j, ref_lines in enumerate(ref_lines_list):
            book_refs[j].append(ref_lines[i])
        if conf_list is not None:
            book_conf.append(conf_list[i])
    return book_dict


def _parse_books(books_arg: List[str]) -> Dict[int, List[int]]:
    # mirrors test.py main(): a single semicolon-joined argument is split, else nargs="*" is used as-is
    if len(books_arg) == 1:
        selections: List[str] = books_arg[0].split(";")
    else:
        selections = books_arg
    return get_chapters(selections)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scores translation output outside of silnlp using the exact sacrebleu calls and "
            "parameters silnlp's test stage uses (silnlp/nmt/test.py)."
        )
    )

    source = parser.add_argument_group("input")
    source.add_argument("--hyp", help="Hypothesis file: plain text, or a .csv/.tsv table. Not used with --exp-dir.")
    source.add_argument("--ref", nargs="*", default=[], help="Reference file(s) (text mode).")
    source.add_argument("--vref", help="Verse reference file, one silnlp vref per line (text mode).")
    source.add_argument("--conf", help="Confidence file (text mode): a *.confidences.tsv or one float per line.")
    source.add_argument("--hyp-col", help="Hypothesis column name (table mode).")
    source.add_argument("--ref-col", nargs="*", default=[], help="Reference column name(s) (table mode).")
    source.add_argument("--vref-col", help="Verse reference column name (table mode).")
    source.add_argument("--conf-col", help="Confidence column name (table mode).")
    source.add_argument(
        "--exp-dir",
        help="A silnlp experiment directory; discovers test.trg-predictions.detok.txt.<step>, "
        "test.trg.detok*.txt, test.vref.txt, and the confidences file automatically.",
    )
    source.add_argument(
        "--step",
        default="avg",
        help="Checkpoint step suffix used in silnlp's file names, e.g. 5000, or 'avg' for the "
        "averaged checkpoint (--exp-dir mode only). Default: avg.",
    )
    source.add_argument(
        "--ref-projects",
        nargs="*",
        metavar="project",
        default=[],
        help="Reference project name(s). In --exp-dir mode, selects which test.trg.detok.<project>.txt "
        "files to score against when more than one exists. In all modes, also fills the CSV's "
        "num_refs/references columns, matching silnlp's --ref-projects.",
    )
    source.add_argument(
        "--allow-tokenized-input",
        default=False,
        action="store_true",
        help="Skip the check for subword-tokenized hypothesis text (e.g. containing '▁' or '</s>').",
    )

    metrics = parser.add_argument_group("metrics")
    metrics.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        choices=SUPPORTED_SCORERS,
        default=[],
        help=f"List of scorers - {SUPPORTED_SCORERS}. Default: {sorted(DEFAULT_SCORERS)} "
        "(silnlp/nmt/experiment.py's default scorer set - not the same as silnlp-nmt-test's own "
        "standalone default of just bleu).",
    )
    metrics.add_argument(
        "--sacrebleu-tokenize",
        default="13a",
        help="Tokenizer passed to sacrebleu for BLEU/m-BLEU (matches silnlp's config.data['sacrebleu_tokenize'], "
        "default 13a).",
    )
    metrics.add_argument("--src-iso", default="", help="Source language ISO code, for the CSV's src_iso column only.")
    metrics.add_argument("--trg-iso", default="", help="Target language ISO code, for the CSV's trg_iso column only.")
    metrics.add_argument(
        "--allow-version-mismatch",
        default=False,
        action="store_true",
        help="Warn instead of exiting when sacrebleu/sil-machine don't match the versions silnlp is pinned to.",
    )

    books = parser.add_argument_group("book selection")
    books.add_argument("--books", nargs="*", metavar="book", default=[], help="Books to score, e.g. MAT MRK.")
    books.add_argument(
        "--by-book",
        default=False,
        action="store_true",
        help="Also emit one score row per book (requires a vref file/column).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out",
        help="Output scores CSV path. Default: scores-<step>-external.csv next to --exp-dir (deliberately "
        "not scores-<step>.csv, so this never overwrites silnlp's own test-stage output), or ./scores.csv "
        "otherwise.",
    )
    output.add_argument(
        "--verse-scores",
        default=False,
        action="store_true",
        help="Also write a per-verse *.scores.tsv file, matching silnlp's write_pair_verse_scores output.",
    )
    output.add_argument("--verse-scores-out", help="Path for the per-verse TSV. Default: <out>.scores.tsv.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.exp_dir and not args.hyp:
        parser.error("one of --exp-dir or --hyp is required")

    check_versions(args.allow_version_mismatch)

    scorers = set(s.lower() for s in args.scorers) if args.scorers else set(DEFAULT_SCORERS)
    if "spbleu" in scorers:
        warn_if_spbleu_model_missing()

    books = _parse_books(args.books)

    ref_projects = set(args.ref_projects)
    if args.exp_dir:
        data, step_token, ref_projects = load_exp_dir_mode(args)
    elif args.hyp_col or Path(args.hyp).suffix.lower() in (".csv", ".tsv"):
        data = load_table_mode(args)
        step_token = "table"
    else:
        data = load_text_mode(args)
        step_token = "text"

    tokenized_line = _looks_tokenized(data.hyp_lines)
    if tokenized_line is not None and not args.allow_tokenized_input:
        raise SystemExit(
            f"Hypothesis lines look subword-tokenized (found e.g. {tokenized_line!r}). silnlp scores "
            "detokenized text - point --hyp at test.trg-predictions.detok.txt.<step> (--exp-dir mode "
            "does this automatically), or pass --allow-tokenized-input if this is intentional."
        )

    if books and data.vref_lines is None:
        raise SystemExit(
            "--books was given but no vref file (--vref) or column (--vref-col) was found - there is "
            "nothing to filter on, so this would silently score the whole corpus instead of the "
            "requested books."
        )

    all_sys, all_refs, all_confs = _filter_by_books(
        data.hyp_lines, data.ref_lines_list, data.vref_lines, data.conf_list, books
    )

    bleu_score, other_scores = compute_scores(
        all_sys,
        all_refs,
        scorers,
        sacrebleu_tokenize=args.sacrebleu_tokenize,
        confidences=all_confs if "confidence" in scorers else None,
    )
    scores = [PairScore("ALL", args.src_iso, args.trg_iso, bleu_score, len(all_sys), ref_projects, other_scores)]

    if args.by_book:
        if data.vref_lines is None:
            raise SystemExit("--by-book requires a vref file (--vref) or column (--vref-col).")
        book_dict = _build_book_dict(data.hyp_lines, data.ref_lines_list, data.vref_lines, data.conf_list, books)
        if len(book_dict) == 0:
            LOGGER.warning("No verses matched a book; not emitting per-book rows.")
        for book, (book_sys, book_refs, book_conf) in book_dict.items():
            book_bleu, book_other = compute_scores(
                book_sys,
                book_refs,
                scorers,
                sacrebleu_tokenize=args.sacrebleu_tokenize,
                confidences=book_conf if "confidence" in scorers else None,
            )
            scores.append(
                PairScore(book, args.src_iso, args.trg_iso, book_bleu, len(book_sys), ref_projects, book_other)
            )

    if args.out:
        out_path = Path(args.out)
    elif args.exp_dir:
        suffix = step_token
        if books:
            book_ids = "_".join(sorted(book_number_to_id(n) for n in books.keys()))
            suffix = f"{book_ids}-{step_token}"
        root = f"scores-{suffix}"
        if ref_projects:
            root += f"-{'_'.join(sorted(ref_projects))}"
        # "-external" keeps this from ever landing on the same path silnlp's own test stage
        # writes (test.py:750-753: scores_file_root = f"scores-{suffix_str}[-{ref_projects}]").
        out_path = Path(args.exp_dir) / f"{root}-external.csv"
    else:
        out_path = Path("scores.csv")

    with out_path.open("w", encoding="utf-8") as scores_file:
        scores[0].writeHeader(scores_file)
        for score in scores:
            score.write(scores_file)
    LOGGER.info(f"Wrote {out_path}")

    header = "book,draft_index,src_iso,trg_iso,num_refs,references,sent_len"
    if scores[0].bleu is not None:
        header += (
            ",BLEU,BLEU_1gram_prec,BLEU_2gram_prec,BLEU_3gram_prec,BLEU_4gram_prec,BLEU_brevity_penalty,"
            "BLEU_total_sys_len,BLEU_total_ref_len"
        )
    header += ("," if len(scores[0].other_scores) > 0 else "") + ",".join(scores[0].other_scores.keys())
    LOGGER.info(header)
    for score in scores:
        buf = StringIO()
        score.write(buf)
        LOGGER.info(buf.getvalue().strip())

    if args.verse_scores:
        if args.verse_scores_out:
            verse_path = Path(args.verse_scores_out)
        else:
            verse_path = out_path.with_name(out_path.name + VERSE_SCORES_SUFFIX)
        write_verse_scores(
            verse_path,
            all_sys,
            all_refs,
            scorers,
            other_scores,
            args.sacrebleu_tokenize,
            all_confs if "confidence" in scorers else None,
        )
        LOGGER.info(f"Wrote {verse_path}")


if __name__ == "__main__":
    main()
