import argparse
import logging
import random
from contextlib import ExitStack
from io import StringIO
from pathlib import Path
from typing import IO, Dict, List, Optional, Set, TextIO, Tuple

import sacrebleu
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, book_number_to_id, get_chapters
from sacrebleu.metrics import BLEU, BLEUScore
from scipy.stats import gmean

from ..common.metrics import compute_meteor_score
from ..common.utils import get_git_revision_hash
from .config import CheckpointType, Config, NMTModel
from .config_utils import load_config
from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__package__ + ".test")

logging.getLogger("sacrebleu").setLevel(logging.ERROR)

_SUPPORTED_SCORERS = ["bleu", "sentencebleu", "chrf3", "chrf3+", "chrf3++", "spbleu", "m-bleu", "m-chrf3", "m-chrf3+", "m-chrf3++", "meteor", "ter", "confidence"]


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

    def writeHeader(self, file: IO) -> None:
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

    def write(self, file: IO) -> None:
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
            if scorer == "confidence":
                file.write(f",{val:.8f}")
            else:
                file.write(f",{val:.2f}")
        file.write("\n")


def score_pair(
    pair_sys: List[str],
    pair_refs: List[List[str]],
    book: str,
    src_iso: str,
    trg_iso: str,
    predictions_detok_file_name: str,
    predictions_conf_file_name: str,
    scorers: Set[str],
    config: Config,
    ref_projects: Set[str],
    draft_index: int = 1,
    pair_confs: Optional[List[float]] = None,
) -> PairScore:
    bleu_score = None
    if "bleu" in scorers:
        bleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize=config.data.get("sacrebleu_tokenize", "13a"),
        )

    if "sentencebleu" in scorers:
        write_sentence_bleu(
            config.exp_dir / (predictions_detok_file_name + ".scores.tsv"),
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize=config.data.get("sacrebleu_tokenize", "13a"),
        )

    other_scores: Dict[str, float] = {}
    if "chrf3" in scorers:
        chrf3_score = sacrebleu.corpus_chrf(pair_sys, pair_refs, char_order=6, beta=3, remove_whitespace=True)
        other_scores["chrF3"] = chrf3_score.score

    if "chrf3+" in scorers:
        chrfp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
        )
        other_scores["chrF3+"] = chrfp_score.score

    if "chrf3++" in scorers:
        chrfpp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
        )
        other_scores["chrF3++"] = chrfpp_score.score

    if "spbleu" in scorers:
        spbleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize="flores200",
        )
        other_scores["spBLEU"] = spbleu_score.score

    # m-bleu, m-chrf3, m-chrf3+, and m-chrf3++ are from the paper https://arxiv.org/pdf/2407.12832
    # These metrics are implemented at the verse-level, rather than the sentence-level
    if "m-bleu" in scorers:
        bleu_scores = []
        for sentence, references in zip(pair_sys, pair_refs):
            bleu_score = sentence_bleu(
                sentence,
                references,
                lowercase=True,
                tokenize=config.data.get("sacrebleu_tokenize", "13a"),
            )
            bleu_scores.append(bleu_score)
        if len(bleu_scores) == 0:
            other_scores["m-BLEU"] = 0
        else:
            other_scores["m-BLEU"] = sum(bleu_scores) / len(bleu_scores)

    if "m-chrf3" in scorers:
        chrf3_scores = []
        for sentence, references in zip(pair_sys, pair_refs):
            chrf3_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, remove_whitespace=True
            )
            chrf3_scores.append(chrf3_score.score)
        if len(chrf3_scores) == 0:
            other_scores["m-chrf3"] = 0
        else:
            other_scores["m-chrf3"] = sum(chrf3_scores) / len(chrf3_scores)

    if "m-chrf3+" in scorers:
        chrfp_scores = []
        for sentence, references in zip(pair_sys, pair_refs):
            chrfp_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
            )
            chrfp_scores.append(chrfp_score.score)
        if len(chrfp_scores) == 0:
            other_scores["m-chrf3+"] = 0
        else:
            other_scores["m-chrf3+"] = sum(chrfp_scores) / len(chrfp_scores)

    if "m-chrf3++" in scorers:
        chrfpp_scores = []
        for sentence, references in zip(pair_sys, pair_refs):
            chrfpp_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
            )
            chrfpp_scores.append(chrfpp_score.score)
        if len(chrfpp_scores) == 0:
            other_scores["m-chrf3+"] = 0
        else:
            other_scores["m-chrf3+"] = sum(chrfpp_scores) / len(chrfpp_scores)

    if "meteor" in scorers:
        meteor_score = compute_meteor_score(trg_iso, pair_sys, pair_refs)
        if meteor_score is not None:
            other_scores["METEOR"] = meteor_score

    if "ter" in scorers:
        ter_score = sacrebleu.corpus_ter(pair_sys, pair_refs)
        if ter_score.score >= 0:
            other_scores["TER"] = ter_score.score

    if "confidence" in scorers:
        if pair_confs is not None:
            confidences = pair_confs
        else:
            try:
                with open(config.exp_dir / predictions_conf_file_name, "r", encoding="utf-8") as f:
                    confidences = [float(line.split("\t")[0]) for line in list(f)[3::2]]
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "Cannot use confidence as a scorer because the confidences file is missing. "
                    "Include the --save-confidences option to generate the file and enable confidence scoring."
                ) from e
        other_scores["confidence"] = gmean(confidences)

    return PairScore(book, src_iso, trg_iso, bleu_score, len(pair_sys), ref_projects, other_scores, draft_index)


def score_individual_books(
    book_dict: Dict[str, Tuple[List[str], List[List[str]]]],
    src_iso: str,
    trg_iso: str,
    predictions_detok_file_name: str,
    predictions_conf_file_name: str,
    scorers: Set[str],
    config: Config,
    ref_projects: Set[str],
):
    overall_sys: List[str] = []
    book_scores: List[PairScore] = []

    for book, book_tuple in book_dict.items():
        pair_sys = book_tuple[0]
        pair_refs = book_tuple[1]
        pair_confs = book_tuple[2]
        overall_sys.extend(pair_sys)
        book_scores.append(
            score_pair(
                pair_sys,
                pair_refs,
                book,
                src_iso,
                trg_iso,
                predictions_detok_file_name,
                predictions_conf_file_name,
                scorers,
                config,
                ref_projects,
                pair_confs=pair_confs,
            )
        )
    return book_scores


def process_individual_books(
    tokenizer: Tokenizer,
    pred_file_path: Path,
    ref_file_paths: List[Path],
    vref_file_path: Path,
    conf_file_path: Path,
    select_rand_ref_line: bool,
    books: Dict[int, List[int]],
) -> Dict[str, Tuple[List[str], List[List[str]]]]:
    # Output data structure
    book_dict: Dict[str, Tuple[List[str], List[List[str]], List[float]]] = {}
    with ExitStack() as stack:
        # Get all references
        ref_files: List[TextIO] = []
        for ref_file_path in ref_file_paths:
            ref_files.append(stack.enter_context(ref_file_path.open("r", encoding="utf-8")))

        vref_file = stack.enter_context(vref_file_path.open("r", encoding="utf-8"))
        pred_file = stack.enter_context(pred_file_path.open("r", encoding="utf-8"))
        conf_file = stack.enter_context(conf_file_path.open("r", encoding="utf-8"))
        conf_list = [float(line.strip().split("\t")[0]) for line in list(conf_file)[3::2]]

        for lines in zip(pred_file, vref_file, conf_list, *ref_files):
            # Get file lines
            pred_line = lines[0].strip()
            detok_pred = tokenizer.detokenize(pred_line)
            vref = lines[1].strip()
            confidence = lines[2]
            # Get book
            if vref == "":
                continue
            vref = VerseRef.from_string(vref, ORIGINAL_VERSIFICATION)
            # Check if book in books
            if len(books) > 0 and vref.book_num not in books:
                continue
            # If book not in dictionary add the book
            if vref.book not in book_dict:
                book_dict[vref.book] = ([], [], [])
            book_pred, book_refs, book_conf = book_dict[vref.book]

            # Add detokenized prediction and confidence to nested dictionary
            book_pred.append(detok_pred)
            book_conf.append(confidence)
            # Check if random ref line selected or not
            if select_rand_ref_line:
                ref_lines: List[str] = [line.strip() for line in lines[3:] if len(line.strip()) > 0]
                ref_index = random.randint(0, len(ref_lines) - 1)
                ref_line = ref_lines[ref_index + 3].strip()
                if len(book_refs) == 0:
                    book_refs.append([])
                book_refs[0].append(ref_line)
            else:
                # For each reference text, add to book_refs
                for ref_index in range(len(ref_files)):
                    ref_line = lines[ref_index + 3].strip()
                    if len(book_refs) == ref_index:
                        book_refs.append([])
                    book_refs[ref_index].append(ref_line)
    return book_dict


def load_test_data(
    tokenizer: Tokenizer,
    vref_file_name: str,
    pred_file_name: str,
    conf_file_name: str,
    ref_pattern: str,
    output_file_name: str,
    ref_projects: Set[str],
    config: Config,
    books: Dict[int, List[int]],
    by_book: bool,
) -> Tuple[List[str], List[List[str]], Dict[str, Tuple[List[str], List[List[str]]]]]:
    sys: List[str] = []
    refs: List[List[str]] = []
    book_dict: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    pred_file_path = config.exp_dir / pred_file_name
    conf_file_path = config.exp_dir / conf_file_name
    with ExitStack() as stack:
        pred_file = stack.enter_context(pred_file_path.open("r", encoding="utf-8"))
        out_file = stack.enter_context((config.exp_dir / output_file_name).open("w", encoding="utf-8"))

        ref_file_paths = list(config.exp_dir.glob(ref_pattern))
        select_rand_ref_line = False
        if len(ref_file_paths) > 1:
            if len(ref_projects) == 0:
                # no refs specified, so randomly select verses from all available train refs to build one ref
                select_rand_ref_line = True
                ref_file_paths = [p for p in ref_file_paths if config.is_train_project(p)]
            else:
                # use specified refs only
                ref_file_paths = [p for p in ref_file_paths if config.is_ref_project(ref_projects, p)]
        ref_files: List[IO] = []
        vref_file: Optional[IO] = None
        vref_file_path = config.exp_dir / vref_file_name
        if len(books) > 0 and vref_file_path.is_file():
            vref_file = stack.enter_context(vref_file_path.open("r", encoding="utf-8"))
        for ref_file_path in ref_file_paths:
            ref_files.append(stack.enter_context(ref_file_path.open("r", encoding="utf-8")))
        for lines in zip(pred_file, *ref_files):
            if vref_file is not None:
                vref_line = vref_file.readline().strip()
                if vref_line != "":
                    vref = VerseRef.from_string(vref_line, ORIGINAL_VERSIFICATION)
                    if vref.book_num not in books:
                        continue
            pred_line = lines[0].strip()
            detok_pred_line = tokenizer.detokenize(pred_line)
            sys.append(detok_pred_line)
            if select_rand_ref_line:
                ref_lines: List[str] = [line.strip() for line in lines[1:] if len(line.strip()) > 0]
                ref_index = random.randint(0, len(ref_lines) - 1)
                ref_line = ref_lines[ref_index]
                if len(refs) == 0:
                    refs.append([])
                refs[0].append(ref_line)
            else:
                for ref_index in range(len(ref_files)):
                    ref_line = lines[ref_index + 1].strip()
                    if len(refs) == ref_index:
                        refs.append([])
                    refs[ref_index].append(ref_line)
            out_file.write(detok_pred_line + "\n")
        if by_book:
            book_dict = process_individual_books(
                tokenizer,
                pred_file_path,
                ref_file_paths,
                vref_file_path,
                conf_file_path,
                select_rand_ref_line,
                books,
            )
    return sys, refs, book_dict


def sentence_bleu(
    hypothesis: str,
    references: List[str],
    smooth_method: str = "exp",
    smooth_value: Optional[float] = None,
    lowercase: bool = False,
    tokenize: str = "13a",
    use_effective_order: bool = True,
) -> BLEUScore:
    """
    Substitute for the sacrebleu version of sentence_bleu, which uses settings that aren't consistent with
    the values we use for corpus_bleu, and isn't fully parameterized
    """
    metric = BLEU(
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        force=False,
        lowercase=lowercase,
        tokenize=tokenize,
        effective_order=use_effective_order,
    )
    return metric.sentence_score(hypothesis, references)


def write_sentence_bleu(
    scores_path: Path,
    preds: List[str],
    refs: List[List[str]],
    lowercase: bool = False,
    tokenize: str = "13a",
):
    with scores_path.open("w", encoding="utf-8", newline="\n") as scores_file:
        scores_file.write("Verse\tBLEU\t1-gram\t2-gram\t3-gram\t4-gram\tBP\tPrediction")
        for ref in refs:
            scores_file.write("\tReference")
        scores_file.write("\n")
        verse_num = 0
        for pred in preds:
            sentences: List[str] = []
            for ref in refs:
                sentences.append(ref[verse_num])
            bleu = sentence_bleu(pred, sentences, lowercase=lowercase, tokenize=tokenize)
            scores_file.write(
                f"{verse_num + 1}\t{bleu.score:.2f}\t{bleu.precisions[0]:.2f}\t{bleu.precisions[1]:.2f}\t"
                f"{bleu.precisions[2]:.2f}\t{bleu.precisions[3]:.2f}\t{bleu.bp:.3f}\t" + pred.rstrip("\n")
            )
            for sentence in sentences:
                scores_file.write("\t" + sentence.rstrip("\n"))
            scores_file.write("\n")
            verse_num += 1


def test_checkpoint(
    config: Config,
    model: NMTModel,
    tokenizer: Tokenizer,
    force_infer: bool,
    by_book: bool,
    ref_projects: Set[str],
    checkpoint_type: CheckpointType,
    step: int,
    scorers: Set[str],
    books: Dict[int, List[int]],
    produce_multiple_translations: bool = False,
    save_confidences: bool = False,
) -> List[PairScore]:
    config.set_seed()
    vref_file_names: List[str] = []
    source_file_names: List[str] = []
    translation_file_names: List[str] = []
    refs_patterns: List[str] = []
    translation_detok_file_names: List[str] = []
    translation_conf_file_names: List[str] = []
    suffix_str = "_".join(map(lambda n: book_number_to_id(n), sorted(books.keys())))
    if len(suffix_str) > 0:
        suffix_str += "-"
    suffix_str += "avg" if step == -1 else str(step)

    features_file_name = "test.src.txt"
    if (config.exp_dir / features_file_name).is_file():
        # all test data is stored in a single file
        vref_file_names.append("test.vref.txt")
        source_file_names.append(features_file_name)
        translation_file_names.append(f"test.trg-predictions.txt.{suffix_str}")
        refs_patterns.append("test.trg.detok*.txt")
        translation_detok_file_names.append(f"test.trg-predictions.detok.txt.{suffix_str}")
        translation_conf_file_names.append(f"test.trg-predictions.txt.{suffix_str}.confidences.tsv")
    else:
        # test data is split into separate files
        for src_iso in sorted(config.test_src_isos):
            for trg_iso in sorted(config.test_trg_isos):
                if src_iso == trg_iso:
                    continue
                prefix = f"test.{src_iso}.{trg_iso}"
                features_file_name = f"{prefix}.src.txt"
                if (config.exp_dir / features_file_name).is_file():
                    vref_file_names.append(f"{prefix}.vref.txt")
                    source_file_names.append(features_file_name)
                    translation_file_names.append(f"{prefix}.trg-predictions.txt.{suffix_str}")
                    refs_patterns.append(f"{prefix}.trg.detok*.txt")
                    translation_detok_file_names.append(f"{prefix}.trg-predictions.detok.txt.{suffix_str}")
                    translation_conf_file_names.append(f"{prefix}.trg-predictions.txt.{suffix_str}.confidences.tsv")

    checkpoint_name = "averaged checkpoint" if step == -1 else f"checkpoint {step}"

    source_paths: List[Path] = []
    vref_paths: Optional[List[Path]] = [] if config.has_scripture_data else None
    translation_paths: List[Path] = []
    for i in range(len(translation_file_names)):
        predictions_path = config.exp_dir / translation_file_names[i]
        if force_infer or not predictions_path.is_file():
            source_paths.append(config.exp_dir / source_file_names[i])
            translation_paths.append(predictions_path)
            if vref_paths is not None:
                vref_paths.append(config.exp_dir / vref_file_names[i])
    if len(translation_paths) > 0:
        LOGGER.info(f"Inferencing {checkpoint_name}")
        model.translate_test_files(
            source_paths,
            translation_paths,
            produce_multiple_translations,
            save_confidences,
            vref_paths,
            step if checkpoint_type is CheckpointType.OTHER else checkpoint_type,
        )

    if produce_multiple_translations:
        num_drafts = model.get_num_drafts()
        vref_file_names = num_drafts * vref_file_names
        source_file_names = num_drafts * source_file_names
        translation_file_names = [
            str(Path(file_name).with_suffix(f".{draft_index}{Path(file_name).suffix}"))
            for draft_index in range(1, num_drafts + 1)
            for file_name in translation_file_names
        ]
        refs_patterns = num_drafts * refs_patterns
        translation_detok_file_names = [
            str(Path(file_name).with_suffix(f".{draft_index}{Path(file_name).suffix}"))
            for draft_index in range(1, num_drafts + 1)
            for file_name in translation_detok_file_names
        ]
        draft_indices = num_drafts * list(range(1, num_drafts + 1))
    else:
        draft_indices = len(source_file_names) * [1]

    LOGGER.info(f"Scoring {checkpoint_name}")
    scores: List[PairScore] = []
    overall_sys: List[str] = []
    overall_refs: List[List[str]] = []
    for (
        vref_file_name,
        features_file_name,
        predictions_file_name,
        refs_pattern,
        predictions_detok_file_name,
        predictions_conf_file_name,
        draft_index,
    ) in zip(
        vref_file_names,
        source_file_names,
        translation_file_names,
        refs_patterns,
        translation_detok_file_names,
        translation_conf_file_names,
        draft_indices,
    ):
        src_iso = config.default_test_src_iso
        trg_iso = config.default_test_trg_iso
        if features_file_name != "test.src.txt":
            parts = features_file_name.split(".")
            src_iso = parts[1]
            trg_iso = parts[2]

        pair_sys, pair_refs, book_dict = load_test_data(
            tokenizer,
            vref_file_name,
            predictions_file_name,
            predictions_conf_file_name,
            refs_pattern,
            predictions_detok_file_name,
            ref_projects,
            config,
            books,
            by_book,
        )

        start_index = len(overall_sys)
        overall_sys.extend(pair_sys)
        for i, ref in enumerate(pair_refs):
            if i == len(overall_refs):
                overall_refs.append([""] * start_index)
            overall_refs[i].extend(ref)
        # ensure that all refs are the same length as the sys
        for overall_ref in filter(lambda r: len(r) < len(overall_sys), overall_refs):
            overall_ref.extend([""] * (len(overall_sys) - len(overall_ref)))

        scores.append(
            score_pair(
                pair_sys,
                pair_refs,
                "ALL",
                src_iso,
                trg_iso,
                predictions_detok_file_name,
                predictions_conf_file_name,
                scorers,
                config,
                ref_projects,
                draft_index,
            )
        )

        if by_book:
            if len(book_dict) != 0:
                book_scores = score_individual_books(
                    book_dict,
                    src_iso,
                    trg_iso,
                    predictions_detok_file_name,
                    predictions_conf_file_name,
                    scorers,
                    config,
                    ref_projects,
                )
                scores.extend(book_scores)
            else:
                LOGGER.error("Error: book_dict did not load correctly. Not scoring individual books.")
    if len(config.test_src_isos) > 1 or len(config.test_trg_isos) > 1:
        bleu = sacrebleu.corpus_bleu(overall_sys, overall_refs, lowercase=True)
        scores.append(PairScore("ALL", "ALL", "ALL", bleu, len(overall_sys), ref_projects))

    scores_file_root = f"scores-{suffix_str}"
    if len(ref_projects) > 0:
        ref_projects_suffix = "_".join(sorted(ref_projects))
        scores_file_root += f"-{ref_projects_suffix}"
    with (config.exp_dir / f"{scores_file_root}.csv").open("w", encoding="utf-8") as scores_file:
        if scores is not None:
            scores[0].writeHeader(scores_file)
        for results in scores:
            results.write(scores_file)
    return scores


def test(
    experiment: str,
    checkpoint: Optional[str] = None,
    last: bool = False,
    avg: bool = False,
    best: bool = False,
    force_infer: bool = False,
    scorers: Set[str] = set(),
    ref_projects: Set[str] = set(),
    books: List[str] = [],
    by_book: bool = False,
    produce_multiple_translations: bool = False,
    save_confidences: bool = False,
    save_checkpoints: bool = False,
):
    exp_name = experiment
    config = load_config(exp_name, save_checkpoints)

    if not any(config.exp_dir.glob("test*.src.txt")):
        LOGGER.info("No test dataset.")
        return

    books_nums = get_chapters(books)

    if save_confidences and "confidence" not in scorers:
        scorers.add("confidence")
    if len(scorers) == 0:
        scorers.add("bleu")
    scorers.intersection_update(set(_SUPPORTED_SCORERS))

    tokenizer = config.create_tokenizer()
    model = config.create_model()
    results: Dict[int, List[PairScore]] = {}
    step: int
    if checkpoint is not None:
        step = int(checkpoint)
        results[step] = test_checkpoint(
            config,
            model,
            tokenizer,
            force_infer,
            by_book,
            ref_projects,
            CheckpointType.OTHER,
            step,
            scorers,
            books_nums,
            produce_multiple_translations,
            save_confidences,
        )

    if avg:
        try:
            step = -1
            results[step] = test_checkpoint(
                config,
                model,
                tokenizer,
                force_infer,
                by_book,
                ref_projects,
                CheckpointType.AVERAGE,
                step,
                scorers,
                books_nums,
                produce_multiple_translations,
                save_confidences,
            )
        except ValueError:
            LOGGER.warn("No average checkpoint available.")

    best_step = 0
    if best and config.has_best_checkpoint:
        _, best_step = model.get_checkpoint_path(CheckpointType.BEST)
        step = best_step
        if step not in results:
            results[step] = test_checkpoint(
                config,
                model,
                tokenizer,
                force_infer,
                by_book,
                ref_projects,
                CheckpointType.BEST,
                step,
                scorers,
                books_nums,
                produce_multiple_translations,
                save_confidences,
            )

    if last or (not best and checkpoint is None and not avg and config.model_dir.exists()):
        _, step = model.get_checkpoint_path(CheckpointType.LAST)
        if step not in results:
            results[step] = test_checkpoint(
                config,
                model,
                tokenizer,
                force_infer,
                by_book,
                ref_projects,
                CheckpointType.LAST,
                step,
                scorers,
                books_nums,
                produce_multiple_translations,
                save_confidences,
            )

    if not config.model_dir.exists():
        results[0] = test_checkpoint(
            config,
            model,
            tokenizer,
            force_infer,
            by_book,
            ref_projects,
            CheckpointType.OTHER,
            0,
            scorers,
            books_nums,
            produce_multiple_translations,
            save_confidences,
        )

    for step in sorted(results.keys()):
        num_refs = results[step][0].num_refs
        if num_refs == 0:
            num_refs = 1
        checkpoint_name: str
        if step == -1:
            checkpoint_name = "averaged checkpoint"
        elif step == best_step:
            checkpoint_name = f"best checkpoint {step}"
        else:
            checkpoint_name = f"checkpoint {step}"
        books_str = "ALL" if len(books_nums) == 0 else ", ".join(sorted(str(num) for num in books_nums.keys()))
        LOGGER.info(f"Test results for {checkpoint_name} ({num_refs} reference(s), books: {books_str})")
        header = "book,draft_index,src_iso,trg_iso,num_refs,references,sent_len"
        if len(results[step]) > 0:
            pair_score = results[step][0]
            header += (
                (
                    ",BLEU,BLEU_1gram_prec,BLEU_2gram_prec,BLEU_3gram_prec,BLEU_4gram_prec,BLEU_brevity_penalty,BLEU_total_sys_len,BLEU_total_ref_len"
                    if pair_score.bleu is not None
                    else ""
                )
                + ("," if len(pair_score.other_scores) > 0 else "")
                + ",".join(pair_score.other_scores.keys())
            )
        LOGGER.info(header)
        for score in results[step]:
            output = StringIO()
            score.write(output)
            output.seek(0)
            for line in output:
                LOGGER.info(line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--checkpoint", type=str, help="Test checkpoint")
    parser.add_argument("--last", default=False, action="store_true", help="Test last checkpoint")
    parser.add_argument("--best", default=False, action="store_true", help="Test best evaluated checkpoint")
    parser.add_argument("--avg", default=False, action="store_true", help="Test averaged checkpoint")
    parser.add_argument("--ref-projects", nargs="*", metavar="project", default=[], help="Reference projects")
    parser.add_argument("--force-infer", default=False, action="store_true", help="Force inferencing")
    parser.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        choices=_SUPPORTED_SCORERS,
        default=[],
        help=f"List of scorers - {_SUPPORTED_SCORERS}",
    )
    parser.add_argument("--books", nargs="*", metavar="book", default=[], help="Books")
    parser.add_argument("--by-book", default=False, action="store_true", help="Score individual books")
    parser.add_argument(
        "--multiple-translations",
        default=False,
        action="store_true",
        help="Produce multiple translations of each verse.",
    )
    parser.add_argument(
        "--save-confidences",
        default=False,
        action="store_true",
        help="Generate file with verse confidences.",
    )
    args = parser.parse_args()

    get_git_revision_hash()

    if len(args.books) == 1:
        books = args.books[0].split(";")
    else:
        books = args.books

    test(
        args.experiment,
        checkpoint=args.checkpoint,
        last=args.last,
        best=args.best,
        avg=args.avg,
        ref_projects=set(args.ref_projects),
        force_infer=args.force_infer,
        scorers=set(s.lower() for s in args.scorers),
        books=books,
        by_book=args.by_book,
        produce_multiple_translations=args.multiple_translations,
        save_confidences=args.save_confidences,
    )


if __name__ == "__main__":
    main()
