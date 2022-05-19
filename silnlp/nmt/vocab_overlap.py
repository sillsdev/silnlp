import argparse
import logging
import os
from ..common.corpus import load_corpus, write_corpus
from .config import get_git_revision_hash, get_mt_exp_dir

logging.basicConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate the vocab overlap between two experiments")
    parser.add_argument("exp1", type=str, help="Experiment 1 folder")
    parser.add_argument("exp2", type=str, help="Experiment 2 folder")
    args = parser.parse_args()

    exp1_dir = get_mt_exp_dir(args.exp1)
    exp2_dir = get_mt_exp_dir(args.exp2)

    exp1_src_vocab = set(line.strip() for line in open(os.path.join(exp1_dir, 'src-onmt.vocab'), 'r', encoding='utf-8'))
    exp2_src_vocab = set(line.strip() for line in open(os.path.join(exp2_dir, 'src-onmt.vocab'), 'r', encoding='utf-8'))
    exp1_trg_vocab = set(line.strip() for line in open(os.path.join(exp1_dir, 'trg-onmt.vocab'), 'r', encoding='utf-8'))
    exp2_trg_vocab = set(line.strip() for line in open(os.path.join(exp2_dir, 'trg-onmt.vocab'), 'r', encoding='utf-8'))

    all_src_vocab = exp1_src_vocab.union(exp2_src_vocab)
    all_trg_vocab = exp1_trg_vocab.union(exp2_trg_vocab)
    overlap_src_vocab = exp1_src_vocab.intersection(exp2_src_vocab)
    overlap_trg_vocab = exp1_trg_vocab.intersection(exp2_trg_vocab)

    print('Source vocabulary metrics:')
    print(f'\tSize:\tExperiment 1: {len(exp1_src_vocab)}\tExperiment 2: {len(exp2_src_vocab)}\tTotal: {len(all_src_vocab)}')
    print(f'\tOverlap:\tExperiment 2 vs Experiment 1: {len(overlap_src_vocab)}')
    print('Target vocabulary metrics:')
    print(f'\tSize:\tExperiment 1: {len(exp1_trg_vocab)}\tExperiment 2: {len(exp2_trg_vocab)}\tTotal: {len(all_trg_vocab)}')
    print(f'\tOverlap:\tExperiment 2 vs Experiment 1: {len(overlap_trg_vocab)}')


if __name__ == "__main__":
    main()
