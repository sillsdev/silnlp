import argparse
import logging
import os
import glob
import filecmp
from typing import List
from pathlib import Path
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import difflib as dl
from ..common.corpus import load_corpus
from .utils import decode_sp, decode_sp_lines
from .config import get_git_revision_hash, get_mt_exp_dir
import sacrebleu
from sacrebleu.metrics import BLEU, BLEUScore

logging.basicConfig()

first_time_diff: bool = True
wrap_format = None
normal_format = None
equal_format = None
insert_format = None
replace_format = None
delete_format = None
unknown_format = None


def sentence_bleu(
    hypothesis: str,
    references: List[str],
    smooth_method: str = "exp",
    smooth_value: float = None,
    lowercase: bool = False,
    tokenize=sacrebleu.DEFAULT_TOKENIZER,
    use_effective_order: bool = False,
) -> BLEUScore:
    """
    Substitute for the sacrebleu version of sentence_bleu, which uses settings that aren't consistent with
    the values we use for corpus_bleu, and isn't fully parameterized
    """
    args = argparse.Namespace(
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        force=False,
        short=False,
        lc=lowercase,
        tokenize=tokenize,
    )

    metric = BLEU(args)
    return metric.sentence_score(hypothesis, references, use_effective_order=use_effective_order)


def add_histogram(df: pd.DataFrame, sheet, exp1: str, exp2: str):
    graph = f"{sheet.name}.jpg"
    # Create the histogram
    ax = df['Score_Delta'].plot.hist(bins=20,
                                     figsize=(8, 3),
                                     grid=True,
                                     title=f'{exp1} vs {exp2}',
                                     xlabel='BLEU Score Delta',
                                     ylabel='Number of Verses'
                                     )
    # Calculate the mean, median, and STD
    score_delta_mean = df['Score_Delta'].mean()
    score_delta_median = df['Score_Delta'].median()
    score_delta_std = df['Score_Delta'].std()
    # Display the mean, median, and STD
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_str = f'Mean: {score_delta_mean:.2f}\nMedian: {score_delta_median:.2f}\nSTD: {score_delta_std:.2f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    # Add the histogram to the sheet
    plt.plot()
    plt.savefig(graph, dpi=80)
    plt.clf()
    sheet.insert_image('B1', graph)


def adjust_column_widths(df: pd.DataFrame, sheet, col_width: int):
    # Iterate through each column and set the width == the max length in that column.  If the max length is > 45,
    # set the column width to 45 and enable word wrapping
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
        if tag == 'equal':
            segments.append(equal_format)
            segments.append(pred[j1:j2])
        elif tag == 'insert':
            segments.append(insert_format)
            segments.append(pred[j1:j2])
        elif tag == 'replace':
            segments.append(replace_format)
            segments.append(pred[j1:j2])
        elif tag == 'delete':
            segments.append(delete_format)
            segments.append(ref[i1:i2])
    return segments


# Add the em-dash
all_punctuation = string.punctuation + u"\u2014"
# Add smart quotes
all_punctuation = all_punctuation + u"\u2018" + u"\u2019" + u"\u201C" + u"\u201D"
# Add Danda (0964) and Double Danda (0965) for Devanagari scripts
all_punctuation = all_punctuation + u"\u0964" + u"\u0965"


def strip_punct(s: str):
    return s.strip(all_punctuation)


def split_punct(s: str):
    return re.findall(all_punctuation, s)


def load_words(word_file: Path, decode: bool = True) -> List[str]:
    unique_words: List[str] = []
    for train_str in load_corpus(word_file):
        if decode:
            line_words = decode_sp(train_str).split(' ')
        else:
            line_words = train_str.split(' ')
        for line_word in line_words:
            stripped_word = strip_punct(line_word.lower())
            if stripped_word is not "" and stripped_word not in unique_words:
                unique_words.append(stripped_word)
    unique_words.sort()
    return unique_words


def split_words(s: str) -> List[str]:
    return s.split(' ')


def apply_unknown_formatting(df: pd.DataFrame, workbook, sheet, histogram_offset: int, src_words: List[str]):
    sheet.write_rich_string('E6', unknown_format, 'Orange (underline)', normal_format, ': Unknown source word')
    for index, row in df.iterrows():
        src = row['Source_Sentence']
        segments = []
        last_state = ""
        s = ""
        for src_word in split_words(src):
            if strip_punct(src_word.lower()) in src_words:
                if last_state is not 'known':
                    last_state = 'known'
                    if s is not '':
                        segments.append(s)
                    segments.append(normal_format)
                    s = src_word + ' '
                else:
                    s = s + src_word + ' '
            else:
                if last_state is not 'unknown':
                    last_state = 'unknown'
                    if s is not '':
                        segments.append(s)
                    segments.append(unknown_format)
                    s = src_word + ' '
                else:
                    s = s + src_word + ' '
        segments.append(s)
        if len(segments) > 2:
            segments.append(wrap_format)
            sheet.write_rich_string(f'B{histogram_offset+index+2}', *segments)


def apply_diff_formatting(df: pd.DataFrame, workbook, sheet, histogram_offset: int):
    sheet.write_rich_string('E2', equal_format, 'Green', normal_format, ': matching text')
    sheet.write_rich_string('E3', insert_format, 'Blue (italics)', normal_format, ': text inserted in prediction')
    sheet.write_rich_string('E4', delete_format, 'Red', normal_format, ': text missing from prediction')
    sheet.write_rich_string('E5', replace_format, 'Purple (bold)', normal_format, ': text replaced in prediction')

    for index, row in df.iterrows():
        ref = row['Target_Sentence']
        p1 = row['Exp1_Prediction']
        p2 = row['Exp2_Prediction']
        if ref != "":
            if p1 != "" and p1 != ref:
                segments = get_diff_segments(ref, p1)
                sheet.write_rich_string(f'D{histogram_offset+index+2}', *segments)
                sheet.write_comment(f'D{histogram_offset+index+2}', p1)
            if p2 != "" and p2 != ref:
                segments = get_diff_segments(ref, p2)
                sheet.write_rich_string(f'E{histogram_offset+index+2}', *segments)
                sheet.write_comment(f'E{histogram_offset+index+2}', p2)


def add_training_corpora(writer, exp1_dir: Path, col_width: int):
    sheet_name = 'Training Data'

    vref_file = os.path.join(exp1_dir, 'train.vref.txt')
    if not os.path.exists(vref_file):
        print(f'No verse reference file {vref_file}; training corpora is not included in spreadsheet')
        return

    vrefs = load_corpus(Path(vref_file))
    srcs = decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, 'train.src.txt'))))
    trgs = decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, 'train.trg.txt'))))

    df = pd.DataFrame(list(zip(vrefs, srcs, trgs)), columns=['Verse_Ref', 'Source', 'Target'])
    df.to_excel(writer, index=False, sheet_name=sheet_name)

    sheet = writer.sheets[sheet_name]
    sheet.set_column(1, 2, col_width, wrap_format)
    sheet.autofilter(0, 0, df.shape[0] - 1, df.shape[1] - 1)


def main() -> None:
    global wrap_format
    global normal_format
    global equal_format
    global replace_format
    global insert_format
    global delete_format
    global unknown_format
    histogram_offset = 15
    text_wrap_column_width = 45

    parser = argparse.ArgumentParser(description="Compare the predictions across 2 experiments")
    parser.add_argument("exp1", type=str, help="Experiment 1 folder")
    parser.add_argument("exp2", type=str, help="Experiment 2 folder")
    parser.add_argument("--show-diffs", default=False, action="store_true",
                        help="Show difference (prediction vs reference")
    parser.add_argument("--show-unknown", default=False, action="store_true",
                        help="Show unknown words in source verse")
    parser.add_argument("--include-train", default=False, action="store_true",
                        help="Include the src/trg training corpora in the spreadsheet")
    parser.add_argument("--preserve-case", default=False, action="store_true",
                        help="Score predictions with case preserved")
    parser.add_argument("--tokenize", type=str, default="13a",
                        help="Sacrebleu tokenizer (none,13a,intl,zh,ja-mecab,char)")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp1_name = args.exp1
    exp1_dir = get_mt_exp_dir(exp1_name)
    exp1_xls = os.path.join(exp1_dir, 'diff_predictions.xlsx')
    exp2_name = args.exp2
    exp2_dir = get_mt_exp_dir(exp2_name)

    if not filecmp.cmp(os.path.join(exp1_dir, "test.vref.txt"), os.path.join(exp2_dir, "test.vref.txt")):
        print(f'{exp1_name} and {exp2_name} have different test set verse references; exiting')
        return

    if not filecmp.cmp(os.path.join(exp1_dir, "test.src.txt"), os.path.join(exp2_dir, "test.src.txt")):
        print(f'{exp1_name} and {exp2_name} have different test set source verse text; exiting')
        return

    if not filecmp.cmp(os.path.join(exp1_dir, "test.trg.detok.txt"), os.path.join(exp2_dir, "test.trg.detok.txt")):
        print(f'{exp1_name} and {exp2_name} have different test set target verse text; exiting')
        return

    # Set up to generate the Excel output
    writer: pd.ExcelWriter = pd.ExcelWriter(exp1_xls, engine='xlsxwriter')
    workbook = writer.book
    wrap_format = workbook.add_format({'text_wrap': True})
    normal_format = workbook.add_format({'color': 'black'})
    equal_format = workbook.add_format({'color': 'green'})
    insert_format = workbook.add_format({'color': 'blue', 'italic': True})
    delete_format = workbook.add_format({'color': 'red', 'font_strikeout': 1})
    replace_format = workbook.add_format({'color': 'purple', 'bold': True})
    unknown_format = workbook.add_format({'color': 'orange', 'underline': True})

    if args.show_unknown:
        src_words = load_words(os.path.join(exp1_dir, "train.src.txt"), True)

    for exp1_file_name in sorted(glob.glob(os.path.join(exp1_dir, f'test.trg-predictions.detok.txt.*'))):
        for exp2_file_name in sorted(glob.glob(os.path.join(exp2_dir, f'test.trg-predictions.detok.txt.*'))):
            exp1_scores: List[float] = []
            exp2_scores: List[float] = []
            exp1_checkpoint = os.path.splitext(exp1_file_name)[1]
            exp2_checkpoint = os.path.splitext(exp2_file_name)[1]
            print(f'Comparing results for {args.exp1}({exp1_checkpoint}) and {args.exp2}({exp2_checkpoint})')
            sheet_name = f'{exp1_checkpoint} vs {exp2_checkpoint}'
            exp1_vrefs = load_corpus(Path(os.path.join(exp1_dir, "test.vref.txt")))
            exp1_srcs = decode_sp_lines(load_corpus(Path(os.path.join(exp1_dir, "test.src.txt"))))
            exp1_refs = load_corpus(Path(os.path.join(exp1_dir, "test.trg.detok.txt")))
            exp1_preds = load_corpus(Path(exp1_file_name))
            exp2_preds = load_corpus(Path(exp2_file_name))

            # Create the initial data frame
            df = pd.DataFrame(list(zip(exp1_vrefs, exp1_srcs, exp1_refs, exp1_preds, exp2_preds)),
                              columns=['Verse_Ref', 'Source_Sentence', 'Target_Sentence',
                                       'Exp1_Prediction', 'Exp2_Prediction'])
            # Calculate the sentence BLEU scores for each prediction
            for index, row in df.iterrows():
                bleu = sentence_bleu(row['Exp1_Prediction'], [row['Target_Sentence']],
                                     lowercase=not args.preserve_case, tokenize=args.tokenize)
                exp1_scores.append(bleu.score)
                bleu = sentence_bleu(row['Exp2_Prediction'], [row['Target_Sentence']],
                                     lowercase=not args.preserve_case, tokenize=args.tokenize)
                exp2_scores.append(bleu.score)
            # Update the DF with the scores and score differences
            df['Exp1_Score'] = exp1_scores
            df['Exp2_Score'] = exp2_scores
            df['Score_Delta'] = df['Exp2_Score'] - df['Exp1_Score']

            df.to_excel(writer, index=False, float_format="%.2f", sheet_name=sheet_name, startrow=histogram_offset)
            sheet = writer.sheets[sheet_name]
            sheet.autofilter(histogram_offset, 0, histogram_offset+df.shape[0]-1, df.shape[1]-1)

            if args.show_unknown:
                apply_unknown_formatting(df, workbook, sheet, histogram_offset, src_words)
            if args.show_diffs:
                apply_diff_formatting(df, workbook, sheet, histogram_offset)

            adjust_column_widths(df, sheet, text_wrap_column_width)
            add_histogram(df, sheet, f'{args.exp1}{exp1_checkpoint}', f'{args.exp2}{exp2_checkpoint}')

    if args.include_train:
        add_training_corpora(writer, exp1_dir, text_wrap_column_width+20)

    writer.save()
#    os.remove(exp1_graph)
    print(f'Output is in {exp1_xls}')


if __name__ == "__main__":
    main()
