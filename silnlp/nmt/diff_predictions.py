import argparse
import logging
import os
import glob
import filecmp
from typing import List, Iterable
from pathlib import Path
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import difflib as dl
from ..common.corpus import load_corpus
from .utils import decode_sp, decode_sp_lines, get_best_model_dir, get_last_checkpoint
from .config import get_git_revision_hash, get_mt_exp_dir
import sacrebleu
from sacrebleu.metrics import BLEU, BLEUScore
import xlsxwriter

logging.basicConfig()

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
bleu_format = None

VREF = "VREF"
SRC_SENTENCE = "Source Sentence"
TRG_SENTENCE = "Target Sentence"
EXP1_SRC_TOKENS = "Exp1 Source Tokens"
EXP2_SRC_TOKENS = "Exp2 Source Tokens"
EXP1_PREDICTION = "Exp1 Prediction"
EXP2_PREDICTION = "Exp2 Prediction"
EXP1_SCORE = "Exp1 Score"
EXP2_SCORE = "Exp2 Score"
SCORE_DELTA = "Score Delta"
DICT_SRC = "Source"
DICT_TRG = "Target"


def sentence_bleu(
    hypothesis: str,
    references: List[str],
    smooth_method: str = "exp",
    smooth_value: float = None,
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


histogram_offset = 0


def add_histogram(writer: pd.ExcelWriter, sheet_name: str, data: pd.Series, bin_size: int, title: str, x_axis: str):
    global histogram_offset

    counts, b, bars = plt.hist(
        data, bins=range(int(min(data.values) - bin_size), int(max(data.values) + bin_size), bin_size)
    )
    df = pd.DataFrame({"bin": b[:-1], "count": counts})
    df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=histogram_offset, startcol=0)
    sheet = writer.book.get_worksheet_by_name(sheet_name)
    chart = writer.book.add_chart({"type": "column"})
    chart.set_title({"name": title})
    chart.add_series(
        {
            "categories": [sheet_name, histogram_offset + 1, 0, histogram_offset + 1 + df.shape[0], 0],
            "values": [sheet_name, histogram_offset + 1, 1, histogram_offset + 1 + df.shape[0], 1],
            "data_labels": {"value": True, "num_format": "0"},
        }
    )
    chart.set_legend({"none": True})
    chart.set_x_axis({"name": x_axis})
    chart.set_y_axis({"name": "Number of Verses"})
    sheet.insert_chart(f"D{histogram_offset+2}", chart)
    histogram_offset += df.shape[0] + 2


def add_stats(df: pd.DataFrame, sheet, exp1: str, exp2: str):
    global bleu_format

    sheet.write_string("A1", "BLEU Scores")
    sheet.write_string("B1", exp1)
    sheet.write_string("C1", exp2)
    sheet.write_string("D1", "Delta Scores")
    sheet.write_string("A2", "Mean")
    sheet.write_string("A3", "Median")
    sheet.write_string("A4", "STD")
    sheet.write_number("B2", df[EXP1_SCORE].mean(), bleu_format)
    sheet.write_number("B3", df[EXP1_SCORE].median(), bleu_format)
    sheet.write_number("B4", df[EXP1_SCORE].std(), bleu_format)
    sheet.write_number("C2", df[EXP2_SCORE].mean(), bleu_format)
    sheet.write_number("C3", df[EXP2_SCORE].median(), bleu_format)
    sheet.write_number("C4", df[EXP2_SCORE].std(), bleu_format)
    sheet.write_number("D2", df[SCORE_DELTA].mean(), bleu_format)
    sheet.write_number("D3", df[SCORE_DELTA].median(), bleu_format)
    sheet.write_number("D4", df[SCORE_DELTA].std(), bleu_format)


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


def get_diff_segments(ref: str, pred: str) -> List:
    segments = []
    seq_matcher = dl.SequenceMatcher(None, ref, pred)
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == "equal":
            segments.append(equal_format)
            segments.append(pred[j1:j2])
        elif tag == "insert":
            segments.append(insert_format)
            segments.append(pred[j1:j2])
        elif tag == "replace":
            segments.append(replace_format)
            segments.append(pred[j1:j2])
        elif tag == "delete":
            segments.append(delete_format)
            segments.append(ref[i1:i2])
    return segments


# Add the em-dash
all_punctuation = string.punctuation + "\u2014"
# Add smart quotes
all_punctuation = all_punctuation + "\u2018" + "\u2019" + "\u201C" + "\u201D"
# Add Danda (0964) and Double Danda (0965) for Devanagari scripts
all_punctuation = all_punctuation + "\u0964" + "\u0965"


def strip_punct(s: str):
    return s.strip(all_punctuation)


def split_punct(s: str):
    return re.findall(all_punctuation, s)


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
        segments = []
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


def apply_diff_formatting(df: pd.DataFrame, sheet, stats_offset: int):
    for index, row in df.iterrows():
        ref = row[TRG_SENTENCE]
        p1 = row[EXP1_PREDICTION]
        p2 = row[EXP2_PREDICTION]
        if ref != "":
            if p1 != "" and p1 != ref:
                segments = get_diff_segments(ref, p1)
                sheet.write_rich_string(f"E{stats_offset+index+2}", *segments)
                sheet.write_comment(f"E{stats_offset+index+2}", p1)
            if p2 != "" and p2 != ref:
                segments = get_diff_segments(ref, p2)
                sheet.write_rich_string(f"G{stats_offset+index+2}", *segments)
                sheet.write_comment(f"G{stats_offset+index+2}", p2)


def apply_dict_formatting(df: pd.DataFrame, sheet, stats_offset: int, dictDf: pd.DataFrame):
    column = "B"
    for index, row in df.iterrows():
        text = row[SRC_SENTENCE]
        segments = []
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


def add_training_corpora(writer, exp1_dir: Path, exp2_dir: Path, col_width: int):
    sheet_name = "Training Data"

    df = pd.DataFrame(columns=[VREF, SRC_SENTENCE, TRG_SENTENCE])

    if os.path.exists(os.path.join(exp1_dir, "train.vref.txt")):
        df[VREF] = list(load_corpus(Path(os.path.join(exp1_dir, "train.vref.txt"))))
    elif exp2_dir is not None and os.path.exists(os.path.join(exp2_dir, "train.vref.txt")):
        df[VREF] = list(load_corpus(Path(os.path.join(exp2_dir, "train.vref.txt"))))

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


def load_dictionary(exp1_dir: Path, exp2_dir: Path):
    dictDf = pd.DataFrame(columns=[DICT_SRC, DICT_TRG])

    src_file_name = os.path.join(exp1_dir, "dict.src.txt")
    trg_file_name = os.path.join(exp1_dir, "dict.trg.txt")
    if not os.path.exists(src_file_name) or not os.path.exists(trg_file_name):
        src_file_name = os.path.join(exp2_dir, "dict.src.txt")
        trg_file_name = os.path.join(exp2_dir, "dict.trg.txt")
        if not os.path.exists(src_file_name) or not os.path.exists(trg_file_name):
            print("Warning: no dictionary files available")
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

    digit_rows: List[List[str], List[str], List[str], List[str]] = []
    for index, row in df.iterrows():
        vref = row[VREF]
        trg = row[TRG_SENTENCE]
        if trg != "":
            trg_digits = get_digit_list(trg)
            if len(trg_digits) > 0:
                src_digits = get_digit_list(row[SRC_SENTENCE])
                p1_digits = get_digit_list(row[EXP1_PREDICTION])
                p2_digits = get_digit_list(row[EXP2_PREDICTION])
                all_digits = list(set(src_digits) | set(trg_digits) | set(p1_digits) | set(p2_digits))
                for digit_str in all_digits:
                    digit_rows.append(
                        [
                            vref,
                            digit_str,
                            digit_str in src_digits,
                            digit_str in trg_digits,
                            digit_str in p1_digits,
                            digit_str in p2_digits,
                        ]
                    )

    digits_df = pd.DataFrame(digit_rows, columns=[VREF, "Digits", "Source", "Reference", "Experiment1", "Experiment2"])

    digits_df.to_excel(writer, index=False, sheet_name=sheet_name)
    sheet = writer.sheets[sheet_name]
    sheet.autofilter(0, 0, digits_df.shape[0], digits_df.shape[1] - 1)


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
    global bleu_format
    text_wrap_column_width = 35

    parser = argparse.ArgumentParser(description="Compare the predictions across 2 experiments")
    parser.add_argument("exp1", type=str, help="Experiment 1 folder")
    parser.add_argument("step1", type=int, help="Experiment 1 step")
    parser.add_argument("exp2", type=str, help="Experiment 2 folder")
    parser.add_argument("step2", type=int, help="Experiment 2 step")
    parser.add_argument(
        "--show-diffs", default=False, action="store_true", help="Show difference (prediction vs reference"
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
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    stats_offset = 5

    exp1_name = args.exp1
    exp1_dir = get_mt_exp_dir(exp1_name)
    exp2_name = args.exp2
    exp2_dir = get_mt_exp_dir(exp2_name)
    output_path = os.path.join(exp1_dir, "diff_predictions.xlsx")

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
    bleu_format = workbook.add_format({"num_format": "0.00"})

    exp1_test_trg_filename = os.path.join(exp1_dir, "test.trg.detok.txt")
    exp2_test_trg_filename = os.path.join(exp2_dir, "test.trg.detok.txt")

    vref_file = None  # Are VREF's available, and do they match?
    if os.path.exists(os.path.join(exp1_dir, "test.vref.txt")):
        vref_file = os.path.join(exp1_dir, "test.vref.txt")
        if os.path.exists(os.path.join(exp2_dir, "test.vref.txt")) and not filecmp.cmp(
            os.path.join(exp1_dir, "test.vref.txt"), os.path.join(exp2_dir, "test.vref.txt")
        ):
            print(f"{exp1_name} and {exp2_name} have different test set verse references; exiting")
            return
    elif os.path.exists(os.path.join(exp2_dir, "test.vref.txt")):
        vref_file = os.path.join(exp2_dir, "test.vref.text")

    if args.show_unknown:
        src_words = load_words(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "train.src.txt")))))
        trg_words = load_words(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "train.trg.txt")))))

    exp1_file_name = os.path.join(exp1_dir, f"test.trg-predictions.detok.txt.{args.step1}")
    if not os.path.exists(exp1_file_name):
        print(f"Predictions file not found: {exp1_file_name}")
        return
    print(f"Experiment 1: {exp1_file_name}")
    sheet_name = f"{args.step1}"

    exp2_file_name = os.path.join(exp2_dir, f"test.trg-predictions.detok.txt.{args.step2}")
    if not os.path.exists(exp2_file_name):
        print(f"Predictions file (best) not found: {exp2_file_name}")
        return
    print(f"-- vs Experiment 2: {exp2_file_name}")
    sheet_name += f" vs {args.step2}"

    dictDf = load_dictionary(exp1_dir, exp2_dir) if args.include_dict or args.show_dict else None

    # Create the initial data frame
    df = pd.DataFrame(
        columns=[
            VREF,
            SRC_SENTENCE,
            TRG_SENTENCE,
            EXP1_SRC_TOKENS,
            EXP1_PREDICTION,
            EXP2_SRC_TOKENS,
            EXP2_PREDICTION,
            EXP1_SCORE,
            EXP2_SCORE,
            SCORE_DELTA,
        ]
    )
    # Load the datasets
    df[VREF] = list(load_corpus(Path(vref_file))) if vref_file is not None else ""
    df[SRC_SENTENCE] = list(decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "test.src.txt")))))
    df[TRG_SENTENCE] = list(load_corpus(Path(exp1_test_trg_filename)))
    df[EXP1_SRC_TOKENS] = list(load_corpus(Path(os.path.join(exp1_dir, "test.src.txt"))))
    df[EXP1_PREDICTION] = list(load_corpus(Path(exp1_file_name)))
    df[EXP2_SRC_TOKENS] = list(load_corpus(Path(os.path.join(exp2_dir, "test.src.txt"))))
    df[EXP2_PREDICTION] = list(load_corpus(Path(exp2_file_name)))

    # Calculate the sentence BLEU scores for each prediction
    exp1_scores: List[float] = []
    exp2_scores: List[float] = []
    for index, row in df.iterrows():
        bleu = sentence_bleu(
            row[EXP1_PREDICTION], [row[TRG_SENTENCE]], lowercase=not args.preserve_case, tokenize=args.tokenize
        )
        exp1_scores.append(bleu.score)
        bleu = sentence_bleu(
            row[EXP2_PREDICTION], [row[TRG_SENTENCE]], lowercase=not args.preserve_case, tokenize=args.tokenize
        )
        exp2_scores.append(bleu.score)

    # Update the DF with the scores and score differences
    df[EXP1_SCORE] = exp1_scores
    df[EXP2_SCORE] = exp2_scores
    df[SCORE_DELTA] = df[EXP2_SCORE] - df[EXP1_SCORE]

    add_legend(writer, "Legend")

    df.to_excel(writer, index=False, float_format="%.2f", sheet_name=sheet_name, startrow=stats_offset)
    sheet = writer.sheets[sheet_name]
    sheet.autofilter(stats_offset, 0, stats_offset + df.shape[0], df.shape[1] - 1)

    if args.show_unknown:
        apply_unknown_formatting(df, sheet, stats_offset, "src", src_words)
        apply_unknown_formatting(df, sheet, stats_offset, "trg", trg_words)
    if args.show_diffs:
        apply_diff_formatting(df, sheet, stats_offset)
    if args.show_dict and dictDf is not None:
        apply_dict_formatting(df, sheet, stats_offset, dictDf)

    adjust_column_widths(df, sheet, text_wrap_column_width)
    add_stats(df, sheet, f"{args.exp1}({args.step1})", f"{args.exp2}({args.step2})")

    data_offset = stats_offset + 2

    add_histogram(writer, "Charts", df[SCORE_DELTA], 2, "Delta BLEU Scores", "Delta BLEU")
    add_histogram(writer, "Charts", df[EXP1_SCORE], 5, "Experiment 1", "BLEU")
    add_histogram(writer, "Charts", df[EXP2_SCORE], 5, "Experiment 2", "BLEU")

    if args.analyze_digits:
        add_digits_analysis(writer, df, text_wrap_column_width + 20)

    if args.include_train:
        add_training_corpora(writer, exp1_dir, exp2_dir, text_wrap_column_width + 20)

    if args.include_dict and dictDf is not None:
        add_dictionary(writer, dictDf, text_wrap_column_width + 20)

    writer.save()
    #    os.remove(exp1_graph)
    print(f"Output is in {output_path}")


if __name__ == "__main__":
    main()
