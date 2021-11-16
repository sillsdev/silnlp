import os
import tempfile
import logging
import multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path

from silnlp.alignment.machine_aligner import FastAlignMachineAligner
from silnlp.common.corpus import tokenize_corpus
from silnlp.alignment.utils import compute_alignment_score

LOGGER = logging.getLogger("silnlp")


def align_set(src_input_path: Path, trg_input_path: Path, output_dir: Path):
    if not src_input_path.exists():
        raise FileExistsError(f"The source file does not exist:{src_input_path}")
    if not trg_input_path.exists():
        raise FileExistsError(f"The target file does not exist:{trg_input_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        src_ranged_path = output_dir / src_input_path.name
        trg_ranged_path = output_dir / trg_input_path.name
        account_for_ranges(src_input_path, trg_input_path, src_ranged_path, trg_ranged_path)

        src_tok_output_path = temp_dir / "tokenize-src-output.txt"
        trg_tok_output_path = temp_dir / "tokenize-trg-output.txt"

        tokenize_corpus(src_ranged_path, src_tok_output_path)
        tokenize_corpus(trg_ranged_path, trg_tok_output_path)

        fast_align = FastAlignMachineAligner(temp_dir)

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
                if src_sentence.strip() == "" or trg_sentence.strip() == "":
                    scores.append(-1)
                else:
                    scores.append(
                        compute_alignment_score(direct_lexicon, inverse_lexicon, src_sentence, trg_sentence, alignment)
                    )
        with (output_dir / "alignment.scores.txt").open("w+", encoding="utf-8") as as_file:
            as_file.writelines(["%0.2f\n" % s for s in scores])
        plt.plot(scores, "k.", markersize=2)
        plt.xlabel("Verses")
        plt.ylabel("Alignment Score")
        plt.savefig(output_dir / "alignment.png")


def align_worker(kwargs):
    return align_set(**kwargs)


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


def account_for_ranges(src_input_path: Path, trg_input_path: Path, src_output_path: Path, trg_output_path: Path):
    src_lines = src_input_path.open(encoding="utf-8").readlines()
    trg_lines = trg_input_path.open(encoding="utf-8").readlines()
    src_concat = ""
    trg_concat = ""
    src_ranged = []
    trg_ranged = []
    for i in range(min(len(src_lines), len(trg_lines)) - 1, -1, -1):
        if src_lines[i] == "<range>\n":
            if trg_lines[i] != "<range>\n":
                trg_concat = trg_lines[i].strip() + " " + trg_concat
                src_concat = ""
            src_lines[i] = "\n"
            trg_lines[i] = "\n"
        else:
            if trg_lines[i] == "<range>\n":
                src_concat = src_lines[i].strip() + " " + src_concat
                trg_concat = ""
                src_lines[i] = "\n"
                trg_lines[i] = "\n"
            else:
                if src_concat != "":
                    src_ranged.insert(0, src_lines[i].strip() + " " + src_concat.strip() + "\n")
                    src_concat = ""
                if trg_concat != "":
                    src_ranged.insert(0, trg_lines[i].strip() + " " + trg_concat.strip() + "\n")
                    trg_concat = ""
    src_output_path.open("w+", encoding="utf-8").writelines(src_lines)
    trg_output_path.open("w+", encoding="utf-8").writelines(trg_lines)


def process_alignments(
    scripture_dir: Path, alignment_dir: Path, src_path: Path, complete_files: list, suffix: str = ""
):
    cpu_num = multiprocessing.cpu_count() // 2
    all_kwargs = []
    for f in complete_files:
        filename = os.path.split(f.strip())[1]
        name = os.path.splitext(filename)[0]
        f_dir = alignment_dir / (name + suffix)
        f_dir.mkdir(exist_ok=True)
        if (f_dir / "alignment.scores.txt").exists():
            LOGGER.info("Already aligned: " + (name + suffix))
        else:
            all_kwargs.append(
                {"src_input_path": src_path, "trg_input_path": scripture_dir / filename, "output_dir": f_dir}
            )
    pool = multiprocessing.Pool(cpu_num)
    result = pool.map_async(align_worker, all_kwargs)
    result.get()
    pool.close()
    pool.join()
