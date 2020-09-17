import argparse
from typing import Dict


def get_best_scores(input_path: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with open(input_path, "r", encoding="utf-8") as input_file:
        for row in input_file:
            _, _, hash, score_str = row.strip().split("\t")
            score = float(score_str)
            best_score = scores.get(hash)
            if best_score is None or score > best_score:
                scores[hash] = score
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Removes duplicate sentence pairs from a parallel corpus")
    parser.add_argument("--input", type=str, help="The tab-delimited input corpus file")
    parser.add_argument("--src-output", type=str, help="The output source corpus file")
    parser.add_argument("--trg-output", type=str, help="The output target corpus file")
    args = parser.parse_args()

    best_scores = get_best_scores(args.input)

    dup_count = 0
    with open(args.input, "r", encoding="utf-8") as input_file, open(
        args.src_output, "w", encoding="utf-8"
    ) as src_output_file, open(args.trg_output, "w", encoding="utf-8") as trg_output_file:
        for row in input_file:
            src_sentence, trg_sentence, hash, score_str = row.strip().split("\t")
            score = float(score_str)
            best_score = best_scores.get(hash)
            if best_score is not None and score == best_score:
                src_output_file.write(src_sentence + "\n")
                trg_output_file.write(trg_sentence + "\n")
                del best_scores[hash]
            else:
                dup_count += 1
    print(f"{dup_count} duplicate(s) were removed.")


if __name__ == "__main__":
    main()
