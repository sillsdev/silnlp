import argparse
from nlp.common.corpus import write_corpus
from typing import List, Tuple

from nltk.translate import Alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts GIZA format to Pharaoh format")
    parser.add_argument("input", help="The GIZA file")
    parser.add_argument("output", help="The output Pharaoh file")
    args = parser.parse_args()

    alignments: List[Tuple[int, Alignment]] = []
    with open(args.input, "r", encoding="utf-8") as in_file:
        line_index = 0
        segment_index = 0
        for line in in_file:
            line = line.strip()
            if line.startswith("#"):
                start = line.index("(")
                end = line.index(")")
                segment_index = int(line[start + 1 : end])
                line_index = 0
            elif line_index == 2:
                start = line.find("({")
                end = line.find("})")
                src_index = -1
                pairs: List[Tuple[int, int]] = []
                while start != -1 and end != -1:
                    if src_index > -1:
                        trg_indices_str = line[start + 3 : end - 1].strip()
                        trg_indices = trg_indices_str.split()
                        for trg_index in trg_indices:
                            pairs.append((src_index, int(trg_index) - 1))
                    start = line.find("({", start + 2)
                    end = line.find("})", end + 2)
                    src_index += 1
                alignments.append((segment_index, Alignment(pairs)))
            line_index += 1

    write_corpus(args.output, map(lambda a: str(a[1]), sorted(alignments, key=lambda a: a[0])))


if __name__ == "__main__":
    main()
