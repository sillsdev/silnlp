import os
from pathlib import Path

from silnlp.alignment.machine_aligner import FastAlignMachineAligner
from silnlp.common.corpus import tokenize_corpus
from silnlp.alignment.utils import compute_alignment_score


def align_set(src_input_path: Path, trg_input_path: Path, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    src_tok_output_path = output_dir / "tokenize-src-output.txt"
    trg_tok_output_path = output_dir / "tokenize-trg-output.txt"

    tokenize_corpus(src_input_path, src_tok_output_path)
    tokenize_corpus(trg_input_path, trg_tok_output_path)

    fast_align = FastAlignMachineAligner(output_dir)

    sym_align_path = output_dir / "sym-align.txt"
    fast_align.train(src_tok_output_path, trg_tok_output_path)
    fast_align.align(sym_align_path)

    direct_lexicon = fast_align.get_direct_lexicon(include_special_tokens=True)
    inverse_lexicon = fast_align.get_inverse_lexicon(include_special_tokens=True)

    scores = []
    with src_tok_output_path.open("r", encoding="utf-8") as src_tok_output_file, trg_tok_output_path.open(
        "r", encoding="utf-8"
    ) as trg_tok_output_file, sym_align_path.open("r", encoding="utf-8") as sym_align_file:
        for src_sentence, trg_sentence, alignment in zip(src_tok_output_file, trg_tok_output_file, sym_align_file):
            scores.append(
                compute_alignment_score(direct_lexicon, inverse_lexicon, src_sentence, trg_sentence, alignment)
            )
    with (output_dir / "alignment.scores.txt").open("w+", encoding="utf-8") as as_file:
        as_file.writelines(scores)


def full_bibles(scripture_dir: Path, threshold_present=0.95):
    reference_len = len((scripture_dir / "vref.txt").open(encoding="utf-8").readlines())
    complete_files = []

    for f in scripture_dir.iterdir():
        if str(f).endswith("vref.txt"):
            continue
        populated_len = sum([len(l) > 1 for l in f.open(encoding="utf-8").readlines()])
        if populated_len >= reference_len * threshold_present:
            complete_files.append(f)
    return complete_files


scripture_dir = Path("C:\\Users\\johnm\\Documents\\repos\\bible-parallel-corpus-internal\\corpus\\scripture")
alignment_dir = scripture_dir / "..\\..\\alignments"
# complete_files = full_bibles(scripture_dir)
# (alignment_dir / "alignment_sources.txt").open('w+').writelines([f'{p.name}\n' for p in complete_files])
complete_files = (alignment_dir / "alignment_sources.txt").open().readlines()
trg_path = alignment_dir / "hbo-HEB.txt"
for f in complete_files[:3]:
    filename = os.path.split(f)[1]
    name = os.path.splitext(filename)[0]
    f_dir = alignment_dir / name
    f_dir.mkdir(exist_ok=True)
    align_set(scripture_dir / filename, trg_path, f_dir)
