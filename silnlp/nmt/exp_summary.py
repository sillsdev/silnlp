import argparse
import glob
import os
import re

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import get_mt_exp_dir

chap_num = 0
trained_books = []
target_book = ""
all_books = []
metrics = []
key_word = ""


def read_data(file_path, data, chapters):
    global chap_num
    global all_books
    global key_word

    for lang_pair in os.listdir(file_path):
        lang_pattern = re.compile(r"([\w-]+)\-([\w-]+)")
        if not lang_pattern.match(lang_pair):
            continue

        data[lang_pair] = {}
        prefix = "+".join(all_books)
        pattern = re.compile(rf"^{re.escape(prefix)}_{key_word}_order_(\d+)_ch$")

        for groups in os.listdir(os.path.join(file_path, lang_pair)):
            m = pattern.match(os.path.basename(groups))
            if m:
                folder_path = os.path.join(file_path, lang_pair, os.path.basename(groups))
                diff_pred_file = glob.glob(os.path.join(folder_path, "diff_predictions*"))
                if diff_pred_file:
                    r = extract_data(diff_pred_file[0])
                    data[lang_pair][int(m.group(1))] = r
                    chapters.append(int(m.group(1)))
                    if int(m.group(1)) > chap_num:
                        chap_num = int(m.group(1))
                else:
                    print(folder_path + " has no diff_predictions file.")


def extract_data(filename, header_row=5) -> dict:
    global chap_num
    global metrics
    global target_book

    metrics = [m.lower() for m in metrics]
    df = pd.read_excel(filename, header=header_row)
    df.columns = [col.strip().lower() for col in df.columns]

    result = {}
    metric_warning = False
    for _, row in df.iterrows():
        vref = row["vref"]
        m = re.match(r"(\d?[A-Z]{2,3}) (\d+)", str(vref))

        book_name, chap = m.groups()
        if book_name != target_book:
            continue

        if int(chap) > chap_num:
            chap_num = int(chap)

        values = []
        for metric in metrics:
            if metric in row:
                values.append(row[metric])
            else:
                metric = True
                values.append(None)

        result[int(chap)] = values

    if metric_warning:
        print("Warning: {metric} is not calculated in {filename}")

    return result


def flatten_dict(data, chapters, baseline={}) -> list:
    global chap_num
    global metrics

    rows = []
    for lang_pair in data:
        for chap in range(1, chap_num + 1):
            row = [lang_pair, chap]
            row.extend([None, None, None] * len(metrics) * len(data[lang_pair]))
            row.extend([None] * len(chapters))
            row.extend([None] * (1 + len(metrics)))

            for res_chap in data[lang_pair]:
                if chap in data[lang_pair][res_chap]:
                    for m in range(len(metrics)):
                        index_m = 3 + 1 + len(metrics) + chapters.index(res_chap) * (len(metrics) * 3 + 1) + m * 3
                        row[index_m] = data[lang_pair][res_chap][chap][m]
            if len(baseline) > 0:
                for m in range(len(metrics)):
                    row[3 + m] = baseline[lang_pair][chap][m]
            rows.append(row)
    return rows


def create_xlsx(rows, chapters, output_path):
    global chap_num
    global metrics

    wb = Workbook()
    ws = wb.active

    num_col = len(metrics) * 3 + 1
    groups = [("language pair", 1), ("Chapter", 1), ("Baseline", (1 + len(metrics)))]
    for chap in chapters:
        groups.append((chap, num_col))

    col = 1
    for header, span in groups:
        start_col = get_column_letter(col)
        end_col = get_column_letter(col + span - 1)
        ws.merge_cells(f"{start_col}1:{end_col}1")
        ws.cell(row=1, column=col, value=header)
        col += span

    sub_headers = []
    baseline_headers = []

    for i, metric in enumerate(metrics):
        if i == 0:
            baseline_headers.append("rank")
            sub_headers.append("rank")
        baseline_headers.append(metric)
        sub_headers.append(metric)
        sub_headers.append("diff (prev)")
        sub_headers.append("diff (start)")

    for i, baseline_header in enumerate(baseline_headers):
        ws.cell(row=2, column=3 + i, value=baseline_header)

    col = 3 + len(metrics) + 1
    for _ in range(len(groups) - 2):
        for i, sub_header in enumerate(sub_headers):
            ws.cell(row=2, column=col + i, value=sub_header)

        col += len(sub_headers)

    for row in rows:
        ws.append(row)

    for row_idx in [1, 2]:
        for col in range(1, ws.max_column + 1):
            ws.cell(row=row_idx, column=col).font = Font(bold=True)
            ws.cell(row=row_idx, column=col).alignment = Alignment(horizontal="center", vertical="center")

    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    ws.merge_cells(start_row=1, start_column=2, end_row=2, end_column=2)
    ws.cell(row=1, column=1).alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")

    cur_lang_pair = 3
    for row_idx in range(3, ws.max_row + 1):
        if ws.cell(row=row_idx, column=4).value is not None:
            ws.cell(row=row_idx, column=3).value = (
                f"=RANK.EQ(D{row_idx}, INDEX(D:D, INT((ROW(D{row_idx})-3)/{chap_num})*{chap_num}+3):INDEX(D:D, \
                        INT((ROW(D{row_idx})-3)/{chap_num})*{chap_num}+{chap_num}+2), 0)"
            )

        start_col = 3 + len(metrics) + 1
        end_col = ws.max_column

        while start_col < end_col:
            start_col += 1
            if ws.cell(row=row_idx, column=start_col).value is None:
                for col in range(start_col - 1, ws.max_column + 1):
                    ws.cell(row=row_idx, column=col).fill = PatternFill(
                        fill_type="solid", start_color="CCCCCC", end_color="CCCCCC"
                    )
                break

            col_letter = get_column_letter(start_col)
            ws.cell(row=row_idx, column=start_col - 1).value = (
                f"=RANK.EQ({col_letter}{row_idx}, INDEX({col_letter}:{col_letter}, \
                    INT((ROW({col_letter}{row_idx})-3)/{chap_num})*{chap_num}+3):INDEX({col_letter}:{col_letter}, \
                        INT((ROW({col_letter}{row_idx})-3)/{chap_num})*{chap_num}+{chap_num}+2), 0)"
            )

            for i in range(1, len(metrics) + 1):
                start_letter = get_column_letter(3 + i)

                diff_prev_col = start_col + 1
                diff_start_col = start_col + 2

                prev_letter = (
                    start_letter
                    if diff_prev_col <= 3 + len(metrics) + 1 + 3 * len(metrics)
                    else get_column_letter(diff_prev_col - 1 - 1 - 3 * len(metrics))
                )
                cur_letter = get_column_letter(diff_prev_col - 1)

                ws.cell(row=row_idx, column=diff_prev_col).value = f"={cur_letter}{row_idx}-{prev_letter}{row_idx}"
                ws.cell(row=row_idx, column=diff_start_col).value = f"={cur_letter}{row_idx}-{start_letter}{row_idx}"

                start_col += 3

        if ws.cell(row=row_idx, column=1).value != ws.cell(row=cur_lang_pair, column=1).value:
            ws.merge_cells(start_row=cur_lang_pair, start_column=1, end_row=row_idx - 1, end_column=1)
            cur_lang_pair = row_idx
        elif row_idx == ws.max_row:
            ws.merge_cells(start_row=cur_lang_pair, start_column=1, end_row=row_idx, end_column=1)

    wb.save(output_path)


# Sample command:
# python -m silnlp.nmt.exp_summary Catapult_Reloaded_Confidences
# --trained-books MRK --target-book MAT --metrics chrf3 confidence --key-word conf --baseline Catapult_Reloaded/2nd_book/MRK
def main() -> None:
    global chap_num
    global trained_books
    global target_book
    global all_books
    global metrics
    global key_word

    parser = argparse.ArgumentParser(description="Pull results")
    parser.add_argument("exp1", type=str, help="Experiment folder")
    parser.add_argument(
        "--trained-books", nargs="*", required=True, type=str.upper, help="Books that are trained in the exp"
    )
    parser.add_argument("--target-book", required=True, type=str.upper, help="Book that is going to be analyzed")
    parser.add_argument(
        "--metrics",
        nargs="*",
        metavar="metrics",
        default=["chrf3", "confidence"],
        type=str.lower,
        help="Metrics that will be analyzed with",
    )
    parser.add_argument("--key-word", type=str, default="conf", help="Key word in the filename for the exp group")
    parser.add_argument("--baseline", type=str, help="Baseline for the exp group")
    args = parser.parse_args()

    trained_books = args.trained_books
    target_book = args.target_book
    all_books = trained_books + [target_book]
    metrics = args.metrics
    key_word = args.key_word

    exp1_name = args.exp1
    exp1_dir = get_mt_exp_dir(exp1_name)

    exp2_name = args.baseline
    exp2_dir = get_mt_exp_dir(exp2_name) if exp2_name else None

    folder_name = "+".join(all_books)
    os.makedirs(os.path.join(exp1_dir, "a_result_folder"), exist_ok=True)
    output_path = os.path.join(exp1_dir, "a_result_folder", f"{folder_name}.xlsx")

    data = {}
    chapters = []
    read_data(exp1_dir, data, chapters)
    chapters = sorted(set(chapters))

    baseline_data = {}
    if exp2_dir:
        for lang_pair in os.listdir(exp2_dir):
            lang_pattern = re.compile(r"([\w-]+)\-([\w-]+)")
            if not lang_pattern.match(lang_pair):
                continue

            baseline_path = os.path.join(exp2_dir, lang_pair)
            baseline_diff_pred = glob.glob(os.path.join(baseline_path, "diff_predictions*"))
            if baseline_diff_pred:
                baseline_data[lang_pair] = extract_data(baseline_diff_pred[0])
            else:
                print(f"Baseline experiment has no diff_predictions file in {baseline_path}")

    print("Writing data...")
    rows = flatten_dict(data, chapters, baseline=baseline_data)
    create_xlsx(rows, chapters, output_path)
    print(f"Result is in {output_path}")


if __name__ == "__main__":
    main()
