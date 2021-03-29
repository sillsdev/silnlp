import argparse
from pathlib import Path
from typing import Set

from .config import get_aligner
from .lexicon import Lexicon
from .utils import get_experiment_dirs, get_experiment_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generates translation model for Clear from an alignment model")
    parser.add_argument("experiments", type=str, help="Experiment pattern")
    parser.add_argument("--aligner", type=str, help="Aligner")
    parser.add_argument("--output", type=str, help="The output directory")
    args = parser.parse_args()

    for exp_dir in get_experiment_dirs(args.experiments):
        exp_name = get_experiment_name(exp_dir)
        print(f"=== Generating models ({exp_name}) ===")
        aligner = get_aligner(args.aligner, exp_dir)
        output_dir = Path(args.output, exp_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        aligner.align(output_dir / "alignments.txt", sym_heuristic="intersection")

        print("Extracting translation model...", end="", flush=True)
        direct_lexicon = aligner.get_direct_lexicon()
        lexicon = Lexicon()
        if aligner.has_inverse_model:
            inverse_lexicon = aligner.get_inverse_lexicon()

            src_words: Set[str] = set(direct_lexicon.source_words)
            src_words.update(inverse_lexicon.target_words)

            trg_words: Set[str] = set(inverse_lexicon.source_words)
            trg_words.update(direct_lexicon.target_words)

            for src_word in src_words:
                for trg_word in trg_words:
                    direct_prob = direct_lexicon[src_word, trg_word]
                    inverse_prob = inverse_lexicon[trg_word, src_word]
                    prob = max(direct_prob, inverse_prob)
                    if prob > 0.1:
                        lexicon[src_word, trg_word] = prob
        else:
            for src_word in direct_lexicon.source_words:
                for trg_word, prob in direct_lexicon.get_target_word_probs(src_word):
                    if prob > 0.1:
                        lexicon[src_word, trg_word] = prob
        lexicon.write(output_dir / "transModel.tsv")
        print(" done.")


if __name__ == "__main__":
    main()
