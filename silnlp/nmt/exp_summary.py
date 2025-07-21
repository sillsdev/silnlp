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


def extract_data(filename, metrics, target_book, header_row=5) -> dict:
    global chap_num

    metrics = [m.lower() for m in metrics]
    df = pd.read_excel(filename, header=header_row)
    df.columns = [col.strip().lower() for col in df.columns]

    result = {}
    for _, row in df.iterrows():
        vref = row["vref"]
        m = re.match(r"([A-Za-z]+)\s+(\d+)", str(vref))

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
                print("Warning: {metric} is not calculated in {filename}")
                values.append(None)

        result[int(chap)] = values
    return result


def flatten_dict(data, metrics, chapters) -> list:
    global chap_num

    res = []
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
            res.append(row)
    return res


def create_xlsx(rows, metrics, chapters, output_path):
    global chap_num

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


def main() -> None:
    global chap_num

    # TODO: Add args for books, metrics, key word, baseline
    parser = argparse.ArgumentParser(description="Pull results")
    parser.add_argument("exp1", type=str, help="Experiment folder")
    args = parser.parse_args()

    trained_books = ["MRK"]
    target_book = ["MAT"]
    all_books = trained_books + target_book

    metrics = ["chrf3", "confidence"]

    key_word = "conf"

    exp1_name = args.exp1
    exp1_dir = get_mt_exp_dir(exp1_name)

    folder_name = "+".join(all_books)
    os.makedirs(os.path.join(exp1_dir, "a_result_folder"), exist_ok=True)
    output_path = os.path.join(exp1_dir, "a_result_folder", f"{folder_name}.xlsx")

    data = {}
    chapters = []

    for lang_pair in os.listdir(exp1_dir):
        lang_pattern = re.compile(r"([\w-]+)\-([\w-]+)")
        if not lang_pattern.match(lang_pair):
            continue

        data[lang_pair] = {}
        prefix = "+".join(all_books)
        pattern = re.compile(rf"^{re.escape(prefix)}_{key_word}_order_(\d+)_ch$")

        for groups in os.listdir(os.path.join(exp1_dir, lang_pair)):
            m = pattern.match(os.path.basename(groups))
            if m:
                base_name = "diff_predictions"
                folder_path = os.path.join(exp1_dir, lang_pair, os.path.basename(groups))
                diff_pred_file = glob.glob(os.path.join(folder_path, f"{base_name}*"))
                if diff_pred_file:
                    r = extract_data(diff_pred_file[0], metrics, target_book[0])
                    data[lang_pair][int(m.group(1))] = r
                    chapters.append(int(m.group(1)))
                    if int(m.group(1)) > chap_num:
                        chap_num = int(m.group(1))
                else:
                    print(os.path.basename(groups) + " has no diff_predictions file.")

    chapters = sorted(set(chapters))
    print("Writing data...")
    rows = flatten_dict(data, metrics, chapters)
    create_xlsx(rows, metrics, chapters, output_path)
    print(f"Result is in {output_path}")


if __name__ == "__main__":
    main()
