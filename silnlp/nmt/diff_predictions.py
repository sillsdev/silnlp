import argparse
import difflib as dl
import logging
import math
import os
import re
import string
from glob import glob
from pathlib import Path
from typing import Any, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
from sacrebleu.metrics.bleu import BLEU, BLEUScore
from tqdm import tqdm

from ..common.corpus import load_corpus
from ..common.environment import SIL_NLP_ENV
from ..common.utils import get_git_revision_hash
from .config import get_mt_exp_dir
from .sp_utils import decode_sp, decode_sp_lines

logging.basicConfig()
logging.getLogger("sacrebleu").setLevel(logging.ERROR)


first_time_diff: bool = True
wrap_format = None
text_align_format = None
normal_format = None
equal_format = None
insert_format = None
replace_format = None
delete_format = None
unknown_format = None
dictionary_format = None
score_format = None

VREF = "VREF"
SRC_SENTENCE = "Source Sentence"
TRG_SENTENCE = "Target Sentence"
SRC_TOKENS = "Source Tokens"
PREDICTION = "Prediction"
CONFIDENCE = "Confidence"
BLEU_SCORE = "BLEU"
SPBLEU_SCORE = "spBLEU"
CHRF3_SCORE = "chrF3"
CHRF3P_SCORE = "chrF3+"
CHRF3PP_SCORE = "chrF3++"
TER_SCORE = "TER"
DICT_SRC = "Source"
DICT_TRG = "Target"

_SUPPORTED_SCORERS = {BLEU_SCORE, SPBLEU_SCORE, CHRF3_SCORE, CHRF3P_SCORE, CHRF3PP_SCORE, TER_SCORE}


def sentence_bleu(
    hypothesis: str,
    references: List[str],
    smooth_method: str = "exp",
    smooth_value: Optional[float] = None,
    lowercase: bool = False,
    tokenize="13a",
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


def extract_chapter(chapter_num):
    parts = chapter_num.split()
    book_name = parts[0]
    vref_num = parts[1]
    chap_num, verse_num = vref_num.split(":")
    order_num = verse_num.split("-")[0]
    return book_name, int(chap_num), int(order_num)


histogram_offset = 0


def strip_lang_code(line: str) -> str:
    return re.sub(r" ?</s> ?[a-zA-Z]{3}_[a-zA-Z]{4}$", "", line)


def strip_lang_codes(lines: Iterable[str]) -> Iterable[str]:
    return map(strip_lang_code, lines)


def get_experiment_type(exp_dir: str) -> str:
    if os.path.exists(os.path.join(exp_dir, "scores.csv")):
        return "SMT"
    elif len(glob(os.path.join(exp_dir, "scores-*.csv"))) > 0:
        return "NMT"
    else:
        return "Unknown"


def get_best_checkpoint(exp_dir: str) -> int:
    step_num_best = 1000000000
    score_files = glob(os.path.join(exp_dir, "scores-*.csv"))
    if len(score_files) == 0:
        if os.path.exists(os.path.join(exp_dir, "scores.csv")):
            return 0
    for score_file in score_files:
        step_num = int(os.path.basename(score_file).split(".")[0].split("-")[1])
        if step_num < step_num_best:
            step_num_best = step_num
    return step_num_best


def get_last_checkpoint(exp_dir: str) -> int:
    step_num_last = 0
    score_files = glob(os.path.join(exp_dir, "scores-*.csv"))
    if len(score_files) == 0:
        if os.path.exists(os.path.join(exp_dir, "scores.csv")):
            return 0
    for score_file in score_files:
        step_num = int(os.path.basename(score_file).split(".")[0].split("-")[1])
        if step_num > step_num_last:
            step_num_last = step_num
    return step_num_last


def add_histograms(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame, scorers: List[str], bin_size: int):
    global histogram_offset
    for scorer in scorers:
        for supported_scorer in _SUPPORTED_SCORERS:
            if scorer == supported_scorer.lower():
                counts, b, bars = plt.hist(
                    df[supported_scorer], bins=range(0, math.ceil(df[supported_scorer].max()) + bin_size, bin_size)
                )
                #                                           bins=range(int(max(df[supported_scorer].min()-bin_size),0),
                #                                                      int(min(df[supported_scorer].max()+bin_size),100),
                #                                                      bin_size)
                #                                          )
                binDf = pd.DataFrame({f"{supported_scorer}": b[:-1], "Count": counts})
                binDf.to_excel(writer, index=False, sheet_name=sheet_name, startrow=histogram_offset, startcol=0)
                sheet = writer.book.get_worksheet_by_name(sheet_name)
                chart = writer.book.add_chart({"type": "column"})
                #    chart.set_title({"name": title})
                chart.add_series(
                    {
                        "categories": [sheet_name, histogram_offset + 1, 0, histogram_offset + 1 + binDf.shape[0], 0],
                        "values": [sheet_name, histogram_offset + 1, 1, histogram_offset + 1 + binDf.shape[0], 1],
                        "data_labels": {"value": True, "num_format": "0"},
                    }
                )
                chart.set_legend({"none": True})
                chart.set_x_axis({"name": supported_scorer})
                chart.set_y_axis({"name": "Number of Verses"})
                sheet.insert_chart(f"D{histogram_offset+2}", chart)
                histogram_offset += binDf.shape[0] + 2


def add_stats(df: pd.DataFrame, sheet):
    global score_format

    sheet.write_string("A1", "Score Summary")
    sheet.write_string("A2", "Mean")
    sheet.write_string("A3", "Median")
    sheet.write_string("A4", "STD")
    column_list = ["B", "C", "D", "E", "F"]
    column_idx = 0
    for column_name in [BLEU_SCORE, SPBLEU_SCORE, CHRF3_SCORE, CHRF3P_SCORE, CHRF3PP_SCORE, TER_SCORE]:
        if column_name in df:
            column_id = column_list[column_idx]
            sheet.write_string(f"{column_id}1", f"{column_name}")
            sheet.write_number(f"{column_id}2", df[column_name].mean(), score_format)
            sheet.write_number(f"{column_id}3", df[column_name].median(), score_format)
            sheet.write_number(f"{column_id}4", df[column_name].std(), score_format)
            column_idx += 1


def adjust_column_widths(df: pd.DataFrame, sheet, col_width: int):
    # Iterate through each column and set the width == the max length in that column.  If the max length is too high,
    # set the column width to the max and enable word wrapping
    for i, col in enumerate(df.columns):
        # find length of column i
        column_len = df[col].astype(str).str.len().max()
        if column_len > col_width:
            sheet.set_column(i, i, col_width, wrap_format)
        else:
            sheet.set_column(i, i, max(column_len, len(col)) - 2)


# Add the em-dash
all_punctuation = string.punctuation + "\u2014"
# Add smart quotes
all_punctuation = all_punctuation + "\u2018" + "\u2019" + "\u201c" + "\u201d"
# Add Danda (0964) and Double Danda (0965) for Devanagari scripts
all_punctuation = all_punctuation + "\u0964" + "\u0965"


def strip_punct(s: str):
    return s.strip(all_punctuation)


def split_punct(s: str):
    return re.findall(all_punctuation, s)


def get_diff_segments(ref: str, pred: str) -> List:
    segments: List[Optional[str]] = []
    #    ref_split = ref.split()
    #    pred_split = pred.split()
    ref_split = re.findall(rf"[\w']+|[{all_punctuation}]", ref)
    pred_split = re.findall(rf"[\w']+|[{all_punctuation}]", pred)
    seq_matcher = dl.SequenceMatcher(None, ref_split, pred_split)
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == "equal":
            segments.append(equal_format)
            segments.append(" ".join(pred_split[j1:j2]) + " ")
        elif tag == "insert":
            segments.append(insert_format)
            segments.append(" ".join(pred_split[j1:j2]) + " ")
        elif tag == "replace":
            segments.append(replace_format)
            segments.append(" ".join(pred_split[j1:j2]) + " ")
        elif tag == "delete":
            segments.append(delete_format)
            segments.append(" ".join(ref_split[i1:i2]) + " ")
    if len(segments) <= 2:
        segments.append(insert_format)
        segments.append(delete_format)
    return segments


def load_words(lines: Iterable[str]) -> List[str]:
    unique_words: List[str] = []
    for train_str in lines:
        for line_word in train_str.split(" "):
            stripped_word = strip_punct(line_word.lower())
            if stripped_word != "" and stripped_word not in unique_words:
                unique_words.append(stripped_word)
    unique_words.sort()
    return unique_words


def split_words(s: str) -> List[str]:
    return s.split(" ")


def apply_unknown_formatting(df: pd.DataFrame, sheet, stats_offset: int, corpus: str, corpus_words: List[str]):
    if corpus == "src":
        column = "B"
        column_name = SRC_SENTENCE
    else:
        column = "C"
        column_name = TRG_SENTENCE
    for index, row in df.iterrows():
        text = row[column_name]
        segments: List[Optional[str]] = []
        last_state = ""
        s = ""
        for word in split_words(text):
            if strip_punct(word.lower()) in corpus_words:
                if last_state != "known":
                    last_state = "known"
                    if s != "":
                        segments.append(s)
                    segments.append(normal_format)
                    s = word + " "
                else:
                    s = s + word + " "
            else:
                if last_state != "unknown":
                    last_state = "unknown"
                    if s != "":
                        segments.append(s)
                    segments.append(unknown_format)
                    s = word + " "
                else:
                    s = s + word + " "
        segments.append(s)
        if len(segments) > 2:
            segments.append(wrap_format)
            sheet.write_rich_string(f"{column}{stats_offset+index+2}", *segments)


def apply_pred_diff_formatting(
    df: pd.DataFrame, sheet, ref_column: str, pred_column: str, sheet_col: str, stats_offset: int
):
    for index, row in df.iterrows():
        ref = row[ref_column]
        pred = row[pred_column]
        if ref != "" and pred != "" and ref != pred:
            segments = get_diff_segments(ref, pred)
            rc = sheet.write_rich_string(f"{sheet_col}{stats_offset+index+2}", *segments)
            if rc != 0:
                print(
                    f"Unable to apply diff formatting to prediction (error: {rc}) for verse {row[VREF]}, >>>{pred}<<<"
                )
                print(f"{segments}")
            sheet.write_comment(f"{sheet_col}{stats_offset+index+2}", pred)


def apply_dict_formatting(df: pd.DataFrame, sheet, stats_offset: int, dictDf: pd.DataFrame):
    column = "B"
    for index, row in df.iterrows():
        text = row[SRC_SENTENCE]
        segments: List[Optional[str]] = []
        last_state = "notdict"
        s = ""
        for word in split_words(text):
            if len(dictDf[dictDf[DICT_SRC] == strip_punct(word.lower())]):
                if last_state != "dict":
                    last_state = "dict"
                    if s != "":
                        segments.append(s)
                        s = ""
                    segments.append(dictionary_format)
            else:
                if last_state == "dict":
                    last_state = "notdict"
                    if s != "":
                        segments.append(s)
                        s = ""
                    segments.append(normal_format)
            s = s + word + " "
        segments.append(s)
        if len(segments) > 2:
            segments.append(wrap_format)
            sheet.write_rich_string(f"{column}{stats_offset+index+2}", *segments)


def add_training_corpora(writer, exp1_dir: Path, col_width: int):
    sheet_name = "Training Data"

    df = pd.DataFrame(columns=[VREF, SRC_SENTENCE, TRG_SENTENCE])

    if os.path.exists(os.path.join(exp1_dir, "train.vref.txt")):
        df[VREF] = list(load_corpus(Path(os.path.join(exp1_dir, "train.vref.txt"))))

    df[SRC_SENTENCE] = list(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "train.src.txt")))))
    df[TRG_SENTENCE] = list(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "train.trg.txt")))))

    df.to_excel(writer, index=False, sheet_name=sheet_name)

    sheet = writer.sheets[sheet_name]
    sheet.set_column(1, 2, col_width, wrap_format)
    sheet.autofilter(0, 0, df.shape[0], df.shape[1] - 1)


def add_legend(writer: pd.ExcelWriter, sheet_name: str):
    sheet = writer.book.add_worksheet(sheet_name)
    sheet.write_string("A1", "Legend")
    sheet.write_rich_string("A2", equal_format, "Green", normal_format, ": matching text")
    sheet.write_rich_string("A3", insert_format, "Blue (italics)", normal_format, ": text inserted in prediction")
    sheet.write_rich_string("A4", delete_format, "Red", normal_format, ": text missing from prediction")
    sheet.write_rich_string("A5", replace_format, "Purple (bold)", normal_format, ": text replaced in prediction")
    sheet.write_rich_string("A6", unknown_format, "Orange (underline)", normal_format, ": Unknown source word")
    sheet.write_rich_string("A7", dictionary_format, "Green (underline)", normal_format, ": Source dictionary word")


def load_dictionary(exp_dir: Path):
    dictDf = pd.DataFrame(columns=[DICT_SRC, DICT_TRG])

    src_file_name = os.path.join(exp_dir, "dict.src.txt")
    trg_file_name = os.path.join(exp_dir, "dict.trg.txt")
    if not os.path.exists(src_file_name) or not os.path.exists(trg_file_name):
        print("Warning: dictionary files available")
        return None

    dictDf[DICT_SRC] = [decode_sp(line.split("\t")[0]).lower() for line in load_corpus(Path(src_file_name))]
    dictDf[DICT_TRG] = [decode_sp(line.split("\t")[0]).lower() for line in load_corpus(Path(trg_file_name))]
    return dictDf


def add_dictionary(writer, dictDf: pd.DataFrame, col_width: int):
    sheet_name = "Dictionary"
    dictDf.to_excel(writer, index=False, sheet_name=sheet_name)
    sheet = writer.sheets[sheet_name]
    sheet.set_column(1, 2, col_width, wrap_format)
    sheet.autofilter(0, 0, dictDf.shape[0], dictDf.shape[1] - 1)


def get_digit_list(s: str) -> List[str]:
    return re.findall(r"\d+(?:,*\d+)*", s)


def add_digits_analysis(writer, df: pd.DataFrame, col_width: int):
    sheet_name = "Digits"

    digit_rows: List[List[Any]] = []
    for index, row in df.iterrows():
        vref = row[VREF]
        trg = row[TRG_SENTENCE]
        if trg != "":
            trg_digits = get_digit_list(trg)
            if len(trg_digits) > 0:
                src_digits = get_digit_list(row[SRC_SENTENCE])
                exp_digits = get_digit_list(row[PREDICTION])
                all_digits = list(set(src_digits) | set(trg_digits) | set(exp_digits))
                for digit_str in all_digits:
                    digit_rows.append(
                        [
                            vref,
                            digit_str,
                            digit_str in src_digits,
                            digit_str in trg_digits,
                            digit_str in exp_digits,
                        ]
                    )

    digits_df = pd.DataFrame(digit_rows, columns=[VREF, "Digits", "Source", "Reference", "Experiment1", "Experiment2"])

    digits_df.to_excel(writer, index=False, sheet_name=sheet_name)
    sheet = writer.sheets[sheet_name]
    sheet.autofilter(0, 0, digits_df.shape[0], digits_df.shape[1] - 1)


def add_score_to_chap(df: pd.DataFrame):
    chapters_pred = {}
    chapters_trg = {}
    for _, row in tqdm(df.iterrows()):
        cur_vref = row[VREF]
        book_name, chap_num, _ = extract_chapter(cur_vref)
        key = book_name + " " + str(chap_num)
        if key not in chapters_pred:
            chapters_pred[key] = []
            chapters_trg[key] = []
            chapters_pred[key].append(row[PREDICTION])
            chapters_trg[key].append([row[TRG_SENTENCE]])

    return chapters_pred, chapters_trg


def add_chap_scores(df: pd.DataFrame, df_chap: pd.DataFrame, scorers: List[str], preserve_case: bool):
    chapters_pred, chapters_trg = add_score_to_chap(df)
    for scorer in scorers:
        scorer = scorer.lower()
        if scorer == BLEU_SCORE.lower():
            print("Sentence BLEU is not provided for chapter scoring!")
            continue
        if scorer == SPBLEU_SCORE.lower():
            df_chap[SPBLEU_SCORE] = None
            print("Calculating spBLEU scores ...")
            for chap, pred in chapters_pred.items():
                spbleu_score = sacrebleu.corpus_bleu(
                    pred, chapters_trg[chap], lowercase=not preserve_case, tokenize="flores200"
                )
                df_chap.loc[df_chap[VREF] == chap, SPBLEU_SCORE] = spbleu_score.score
        elif scorer == CHRF3_SCORE.lower():
            df_chap[CHRF3_SCORE] = None
            print("Calculating chrF3 scores ...")
            for chap, pred in chapters_pred.items():
                chrf3_score = sacrebleu.corpus_chrf(
                    pred, chapters_trg[chap], char_order=6, beta=3, remove_whitespace=True
                )
                df_chap.loc[df_chap[VREF] == chap, CHRF3_SCORE] = chrf3_score.score
        elif scorer == CHRF3P_SCORE.lower():
            print("Calculating chrF3+ scores ...")
            for chap, pred in chapters_pred.items():
                chrf3p_score = sacrebleu.corpus_chrf(
                    pred,
                    chapters_trg[chap],
                    char_order=6,
                    beta=3,
                    word_order=1,
                    remove_whitespace=True,
                    eps_smoothing=True,
                )
                df_chap.loc[df_chap[VREF] == chap, CHRF3P_SCORE] = chrf3p_score.score
        elif scorer == CHRF3PP_SCORE.lower():
            print("Calculating chrF3++ scores ...")
            for chap, pred in chapters_pred.items():
                chrf3pp_score = sacrebleu.corpus_chrf(
                    pred,
                    chapters_trg[chap],
                    char_order=6,
                    beta=3,
                    word_order=2,
                    remove_whitespace=True,
                    eps_smoothing=True,
                )
                df_chap.loc[df_chap[VREF] == chap, CHRF3PP_SCORE] = chrf3pp_score.score
        elif scorer == TER_SCORE.lower():
            print("Calculating TER scores ...")
            for chap, pred in chapters_pred.items():
                ter_score = sacrebleu.corpus_ter(pred, chapters_trg[chap])
                df_chap.loc[df_chap[VREF] == chap, TER_SCORE] = ter_score.score if ter_score.score >= 0 else 0


def add_scores(df: pd.DataFrame, scorers: List[str], preserve_case: bool, tokenize: bool):
    for scorer in scorers:
        scores: List[float] = []
        scorer = scorer.lower()
        if scorer == BLEU_SCORE.lower():
            for index, row in tqdm(df.iterrows(), desc="Calculating BLEU scores ..."):
                bleu = sentence_bleu(
                    row[PREDICTION], [row[TRG_SENTENCE]], lowercase=not preserve_case, tokenize=tokenize
                )
                scores.append(bleu.score)
            df[BLEU_SCORE] = scores
        elif scorer == SPBLEU_SCORE.lower():
            for index, row in tqdm(df.iterrows(), desc="Calculating spBLEU scores ..."):
                spbleu_score = sacrebleu.corpus_bleu(
                    [row[PREDICTION]], [[row[TRG_SENTENCE]]], lowercase=not preserve_case, tokenize="flores200"
                )
                scores.append(spbleu_score.score)
            df[SPBLEU_SCORE] = scores
        elif scorer == CHRF3_SCORE.lower():
            for index, row in tqdm(df.iterrows(), desc="Calculating chrF3 scores ..."):
                chrf3_score = sacrebleu.corpus_chrf(
                    [row[PREDICTION]], [[row[TRG_SENTENCE]]], char_order=6, beta=3, remove_whitespace=True
                )
                scores.append(chrf3_score.score)
            df[CHRF3_SCORE] = scores
        elif scorer == CHRF3P_SCORE.lower():
            for index, row in tqdm(df.iterrows(), desc="Calculating chrF3+ scores ..."):
                chrf3p_score = sacrebleu.corpus_chrf(
                    [row[PREDICTION]],
                    [[row[TRG_SENTENCE]]],
                    char_order=6,
                    beta=3,
                    word_order=1,
                    remove_whitespace=True,
                    eps_smoothing=True,
                )
                scores.append(chrf3p_score.score)
            df[CHRF3P_SCORE] = scores
        elif scorer == CHRF3PP_SCORE.lower():
            for index, row in tqdm(df.iterrows(), desc="Calculating chrF3++ scores ..."):
                chrf3pp_score = sacrebleu.corpus_chrf(
                    [row[PREDICTION]],
                    [[row[TRG_SENTENCE]]],
                    char_order=6,
                    beta=3,
                    word_order=2,
                    remove_whitespace=True,
                    eps_smoothing=True,
                )
                scores.append(chrf3pp_score.score)
            df[CHRF3PP_SCORE] = scores
        elif scorer == TER_SCORE.lower():
            for index, row in tqdm(df.iterrows(), desc="Calculating TER scores ..."):
                ter_score = sacrebleu.corpus_ter([row[PREDICTION]], [[row[TRG_SENTENCE]]])
                scores.append(ter_score.score if ter_score.score >= 0 else 0)
            df[TER_SCORE] = scores


def get_sequence_confidences(confidences_file: Path) -> List[float]:
    confidences = []
    with open(confidences_file, "r") as f:
        lines = f.readlines()
        for i in range(3, len(lines), 2):
            confidences.append(float(lines[i].split("\t")[0]))
    return confidences


def main() -> None:
    global wrap_format
    global text_align_format
    global normal_format
    global equal_format
    global replace_format
    global insert_format
    global delete_format
    global unknown_format
    global dictionary_format
    global score_format
    text_wrap_column_width = 35

    parser = argparse.ArgumentParser(description="Compare the predictions across 2 experiments")
    parser.add_argument("exp1", type=str, help="Experiment 1 folder")
    parser.add_argument("--last", default=False, action="store_true", help="Use the last result (instead of best)")
    parser.add_argument(
        "--show-diffs", default=False, action="store_true", help="Show difference (prediction vs reference)"
    )
    parser.add_argument("--show-unknown", default=False, action="store_true", help="Show unknown words in source verse")
    parser.add_argument("--show-dict", default=False, action="store_true", help="Show dictionary words in source verse")
    parser.add_argument(
        "--include-train",
        default=False,
        action="store_true",
        help="Include the src/trg training corpora in the spreadsheet",
    )
    parser.add_argument(
        "--include-dict", default=False, action="store_true", help="Include the src/trg dictionary in the spreadsheet"
    )
    parser.add_argument("--analyze-digits", default=False, action="store_true", help="Perform digits analysis")
    parser.add_argument(
        "--preserve-case", default=False, action="store_true", help="Score predictions with case preserved"
    )
    parser.add_argument(
        "--tokenize", type=str, default="13a", help="Sacrebleu tokenizer (none,13a,intl,zh,ja-mecab,char)"
    )
    parser.add_argument(
        "--chapter-score", default=False, action="store_true", help="Show scores in chapters rather than in verses"
    )
    parser.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        choices=[scorer.lower() for scorer in _SUPPORTED_SCORERS],
        default=[BLEU_SCORE.lower()],
        type=str.lower,
        help=f"List of scorers - {_SUPPORTED_SCORERS}",
    )
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    stats_offset = 5

    exp1_name = args.exp1
    SIL_NLP_ENV.copy_experiment_from_bucket(exp1_name)
    exp1_dir = get_mt_exp_dir(exp1_name)

    # SIL_NLP_ENV.copy_experiment_from_bucket(exp1_name, patterns="*.txt*")
    # SIL_NLP_ENV.copy_experiment_from_bucket(exp1_name, patterns="*.csv")
    # SIL_NLP_ENV.copy_experiment_from_bucket(exp1_name, patterns="*.yaml")

    # exp1_type = "NMT"

    exp1_type = get_experiment_type(str(exp1_dir))
    if exp1_type != "SMT" and exp1_type != "NMT":
        print("Can't determine experiment type!")
        return

    exp1_step = (
        0
        if exp1_type == "SMT"
        else get_last_checkpoint(str(exp1_dir)) if args.last else get_best_checkpoint(str(exp1_dir))
    )
    output_path = os.path.join(exp1_dir, f"diff_predictions.{exp1_step}.xlsx")
    if args.chapter_score:
        output_path = os.path.join(exp1_dir, f"diff_predictions.chapters.{exp1_step}.xlsx")

    # Set up to generate the Excel output
    writer: pd.ExcelWriter = pd.ExcelWriter(output_path, engine="xlsxwriter")
    workbook = writer.book
    wrap_format = workbook.add_format({"text_wrap": True})
    text_align_format = workbook.add_format({"align": "left", "valign": "top"})
    normal_format = workbook.add_format({"color": "black"})
    equal_format = workbook.add_format({"color": "green"})
    insert_format = workbook.add_format({"color": "blue", "italic": True})
    delete_format = workbook.add_format({"color": "red", "font_strikeout": 1})
    replace_format = workbook.add_format({"color": "purple", "bold": True})
    unknown_format = workbook.add_format({"color": "orange", "underline": True})
    dictionary_format = workbook.add_format({"color": "green", "underline": True})
    score_format = workbook.add_format({"num_format": "0.00"})

    vref_file = None  # Is VREF file available?
    if os.path.exists(os.path.join(exp1_dir, "test.vref.txt")):
        vref_file = os.path.join(exp1_dir, "test.vref.txt")

    if args.show_unknown:
        src_words = load_words(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "train.src.txt")))))
        trg_words = load_words(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "train.trg.txt")))))

    if exp1_type == "NMT":
        exp1_test_trg_filename = os.path.join(exp1_dir, "test.trg.detok.txt")
        exp1_file_name = os.path.join(exp1_dir, f"test.trg-predictions.detok.txt.{exp1_step}")
    else:
        exp1_test_trg_filename = os.path.join(exp1_dir, "test.trg.txt")
        exp1_file_name = os.path.join(exp1_dir, f"test.trg-predictions.txt")
    if not os.path.exists(exp1_file_name):
        print(f"Predictions file not found: {exp1_file_name}")
        return

    print(f"Analyzing predictions file: {exp1_file_name}")
    sheet_name = f"{exp1_step}"

    dictDf = load_dictionary(exp1_dir) if args.include_dict or args.show_dict else None

    # Create the initial data frame
    df = pd.DataFrame(columns=[VREF, SRC_SENTENCE, TRG_SENTENCE, PREDICTION, CONFIDENCE])

    # Load the datasets
    df[VREF] = list(load_corpus(Path(vref_file))) if vref_file is not None else ""
    df[SRC_SENTENCE] = list(
        strip_lang_codes(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "test.src.txt")))))
    )
    df[TRG_SENTENCE] = list(load_corpus(Path(exp1_test_trg_filename)))
    df[PREDICTION] = list(load_corpus(Path(exp1_file_name)))
    prediction_col = "D"

    if exp1_type == "NMT":
        df.insert(2, SRC_TOKENS, list(strip_lang_codes(load_corpus(Path(os.path.join(exp1_dir, "test.src.txt"))))))
        prediction_col = "E"
        df[CONFIDENCE] = get_sequence_confidences(
            os.path.join(exp1_dir, f"test.trg-predictions.txt.{exp1_step}.confidences.tsv")
        )

    if args.chapter_score:
        df[["ChapName", "ChapNum", "SortNum"]] = df[VREF].apply(lambda x: pd.Series(extract_chapter(x)))
        df = df.sort_values(by=["ChapName", "ChapNum", "SortNum"])

        df_chap = (
            df.groupby(["ChapName", "ChapNum"])
            .agg({SRC_SENTENCE: " ".join, SRC_TOKENS: " ".join, TRG_SENTENCE: " ".join, PREDICTION: " ".join})
            .reset_index()
        )
        df_chap[VREF] = df_chap["ChapName"] + " " + df_chap["ChapNum"].astype(str)
        df_chap = df_chap[[VREF, SRC_SENTENCE, SRC_TOKENS, TRG_SENTENCE, PREDICTION]]
        add_chap_scores(df, df_chap, args.scorers, args.preserve_case)
        df = df_chap
    else:
        add_scores(df, args.scorers, args.preserve_case, args.tokenize)

    df.sort_values(VREF, inplace=True)
    df = df.reset_index(drop=True)
    df.to_excel(writer, index=False, float_format="%.2f", sheet_name=sheet_name, startrow=stats_offset)
    sheet = writer.sheets[sheet_name]
    sheet.autofilter(stats_offset, 0, stats_offset + df.shape[0], df.shape[1] - 1)

    if args.show_unknown:
        apply_unknown_formatting(df, sheet, stats_offset, "src", src_words)
        apply_unknown_formatting(df, sheet, stats_offset, "trg", trg_words)
    if args.show_diffs:
        apply_pred_diff_formatting(df, sheet, TRG_SENTENCE, PREDICTION, prediction_col, stats_offset)
        add_legend(writer, "Legend")
    if args.show_dict and dictDf is not None:
        apply_dict_formatting(df, sheet, stats_offset, dictDf)

    adjust_column_widths(df, sheet, text_wrap_column_width)
    add_stats(df, sheet)

    data_offset = stats_offset + 2

    add_histograms(writer, "Charts", df, args.scorers, 5)

    if args.analyze_digits:
        add_digits_analysis(writer, df, text_wrap_column_width + 20)

    if args.include_train:
        add_training_corpora(writer, exp1_dir, text_wrap_column_width + 20)

    if args.include_dict and dictDf is not None:
        add_dictionary(writer, dictDf, text_wrap_column_width + 20)

    writer.close()
    #    os.remove(exp1_graph)
    SIL_NLP_ENV.copy_experiment_to_bucket(exp1_name)
    print(f"Output is in {output_path}")

    # SIL_NLP_ENV.copy_experiment_to_bucket(exp1_name, patterns=("diff_predictions.*"), overwrite=True)


if __name__ == "__main__":
    main()
