import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from machine.scripture import VerseRef
from openpyxl import load_workbook
from scipy.stats import linregress

from silnlp.nmt.config import get_mt_exp_dir


@dataclass
class VerseScore:
    vref: str
    confidence: float
    projected_chrf3: Optional[float] = None


def estimate_quality(
    experiment: str, diff_predictions_file_name: str, confidences_relative_path: str
) -> List[VerseScore]:
    exp_dir = Path(get_mt_exp_dir(experiment))
    chrf3_scores, confidence_scores = extract_diff_predictions(exp_dir / diff_predictions_file_name)
    if len(chrf3_scores) != len(confidence_scores):
        raise ValueError("The number of chrF3 scores and confidence scores do not match.")
    slope, intercept = linregress(confidence_scores, chrf3_scores)[:2]
    vref_scores = extract_confidences(exp_dir / confidences_relative_path)
    with open(exp_dir / "projected_chrf3.tsv", "w", encoding="utf-8") as output_file:
        output_file.write("VRef\tConfidence\tProjected chrF3\n")
        for vref_score in vref_scores:
            projected_chrf3 = slope * vref_score.confidence + intercept
            vref_score.projected_chrf3 = projected_chrf3
            output_file.write(f"{vref_score.vref}\t{vref_score.confidence}\t{projected_chrf3:.2f}\n")

    compute_usable_proportions(vref_scores, exp_dir)


def extract_diff_predictions(diff_predictions_file_path) -> Tuple[List[str], List[str]]:
    chrf3_scores = extract_diff_predictions_column(diff_predictions_file_path, "chrf3")
    confidence_scores = extract_diff_predictions_column(diff_predictions_file_path, "confidence")
    return chrf3_scores, confidence_scores


def extract_diff_predictions_column(file_path: Path, target_header: str) -> List[str]:
    wb = load_workbook(file_path)
    ws = wb.active

    header_row_idx = None
    col_idx = None

    for row in ws.iter_rows():
        for cell in row:
            if cell.value == "Score Summary":
                break
            if str(cell.value).lower() == target_header.lower():
                header_row_idx = cell.row
                col_idx = cell.column
                break
        if header_row_idx:
            break

    if not header_row_idx:
        raise ValueError(f"Header '{target_header}' not found.")

    data = []
    for row in ws.iter_rows(min_row=header_row_idx + 1, min_col=col_idx, max_col=col_idx):
        cell_value = row[0].value
        if cell_value is not None:
            data.append(cell_value)

    return data


def extract_confidences(input_file_path: Path) -> List[VerseScore]:
    current_book = ""
    current_chapter = 0
    current_verse = 0
    is_at_verse_reference = False

    vref_confidences: List[VerseScore] = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.lower().startswith("vref") or line.lower().startswith("sequence score"):
                continue

            match = re.match(r"^([0-9A-Z][A-Z]{2}) (\d+):(\d+)(/.*)?", line)
            if match:
                current_book = match.group(1)
                current_chapter = int(match.group(2))
                current_verse = int(match.group(3))
                extra = match.group(4)

                is_at_verse_reference = current_verse != 0 and not extra
            elif is_at_verse_reference:
                cols = line.split("\t")
                if cols:
                    vref_confidences += [
                        VerseScore(f"{current_book} {current_chapter}:{current_verse}", float(cols[0]))
                    ]
    return vref_confidences


@dataclass
class UsabilityParameters:
    count: float
    mean: float
    variance: float


def compute_usable_proportions(
    verse_scores: List[VerseScore], output_dir: Path
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]]]:
    usable_params, unusable_params = parse_parameters(output_dir / "usability_parameters.tsv")

    book_totals = defaultdict(float)
    book_counts = defaultdict(int)
    chapter_totals = defaultdict(lambda: defaultdict(float))
    chapter_counts = defaultdict(lambda: defaultdict(int))

    for verse_score in verse_scores:
        vref = VerseRef.from_string(verse_score.vref)
        if vref.verse_num == 0:
            continue

        prob = calculate_usable_prob(verse_score.projected_chrf3, usable_params, unusable_params)
        book_totals[vref.book] += prob
        book_counts[vref.book] += 1
        chapter_totals[vref.book][vref.chapter_num] += prob
        chapter_counts[vref.book][vref.chapter_num] += 1

    with open(output_dir / "usability_books.tsv", "w", encoding="utf-8", newline="\n") as f:
        for book in sorted(book_totals):
            avg_prob = book_totals[book] / book_counts[book]
            f.write(f"{book}\t{avg_prob:.6f}\n")

    with open(output_dir / "usability_chapters.tsv", "w", encoding="utf-8", newline="\n") as f:
        for book in sorted(chapter_totals):
            for chapter in sorted(chapter_totals[book]):
                avg_prob = chapter_totals[book][chapter] / chapter_counts[book][chapter]
                f.write(f"{book}\t{chapter}\t{avg_prob:.6f}\n")


def parse_parameters(parameter_file: Path) -> Tuple[UsabilityParameters, UsabilityParameters]:
    params = {
        "usable": UsabilityParameters(263, 51.4, 95.19),
        "unusable": UsabilityParameters(97, 45.85, 99.91),
    }
    if parameter_file.exists():
        with open(parameter_file, "r", encoding="utf-8") as f:
            for line in f:
                label, count, mean, variance = line.strip().split("\t")
                params[label] = UsabilityParameters(float(count), float(mean), float(variance))
    else:
        print(f"Warning: {parameter_file} does not exist. Using default parameters.")

    return params["usable"], params["unusable"]


def calculate_usable_prob(
    chrf3: float,
    usable: UsabilityParameters,
    unusable: UsabilityParameters,
) -> float:
    usable_weight = exp(-((chrf3 - usable.mean) ** 2) / (2 * usable.variance)) * usable.count
    unusable_weight = exp(-((chrf3 - unusable.mean) ** 2) / (2 * unusable.variance)) * unusable.count

    return usable_weight / (usable_weight + unusable_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the quality of drafts created by an NMT model.")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "diff_predictions_file_name", help="The diff predictions filename to determine line of best fit."
    )
    parser.add_argument(
        "confidences_relative_path",
        help="The file path to the confidences file relative to the current experiment directory.",
    )
    args = parser.parse_args()

    estimate_quality(args.experiment, args.diff_predictions_file_name, args.confidences_relative_path)


if __name__ == "__main__":
    main()
