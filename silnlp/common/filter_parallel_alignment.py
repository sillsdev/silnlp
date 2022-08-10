import os
from pathlib import Path
import argparse
import pandas as pd

from .utils import get_git_revision_hash
from .corpus import load_corpus, write_corpus
from ..alignment.utils import add_alignment_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtering parallel corpora based on alignment score threshold")
    parser.add_argument('src', type=str, help='source file')
    parser.add_argument('trg', type=str, help='target file')
    parser.add_argument('score', type=str, help='score file')
    parser.add_argument('--quantile', type=float, default=0.20, help='quantile threshold for exclusion')
    parser.add_argument('--aligner', type=str, default='fast_align', help='aligner (fast_align, ibm-1, ibm-2, ibm-4, hmm')
    parser.add_argument('--errors', default=False, action="store_true", help="log errors")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    aligner = args.aligner

    src_out_file_name = f'{os.path.splitext(args.src)[0]}.filtered{os.path.splitext(args.src)[1]}'
    trg_out_file_name = f'{os.path.splitext(args.trg)[0]}.filtered{os.path.splitext(args.trg)[1]}'
    score_file_name = args.score

    print('Loading corpus ...')
    corpus = pd.DataFrame(columns=['source', 'target'])
    corpus['source'] = list(load_corpus(Path(args.src)))
    corpus['target'] = list(load_corpus(Path(args.trg)))

    if not os.path.exists(score_file_name):
        print('Aligning corpus ...')
        add_alignment_scores(corpus, aligner)
        write_corpus(Path(score_file_name), corpus['score'].astype(str))
    else:
        print('Loading alignment scores ...')
        corpus['score'] = list(load_corpus(Path(args.score)))

    print(f'Filtering corpus (lowest {args.quantile*100}% of alignment scores)')
    score_threshold = corpus['score'].quantile(args.quantile)
    filtered_corpus = corpus[corpus['score'] > score_threshold]

    alignment_score = corpus['score'].mean()
    filtered_alignment_score = filtered_corpus['score'].mean()
    print(f'Alignment scores: Original: {alignment_score}, Filtered: {filtered_alignment_score}')

    print('Saving filtered corpus ...')
    write_corpus(Path(src_out_file_name), filtered_corpus['source'])
    write_corpus(Path(trg_out_file_name), filtered_corpus['target'])

    print(f'Filtered {len(corpus) - len(filtered_corpus)}')
    if args.errors:
        src_err_file_name = f'{os.path.splitext(args.src)[0]}.filter.errors{os.path.splitext(args.src)[1]}'
        print(f'Writing filtered sentence pairs to {src_err_file_name}')
        excluded_corpus = corpus[corpus['score'] <= score_threshold]
        excluded_corpus.to_csv(src_err_file_name, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
