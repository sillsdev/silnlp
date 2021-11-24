import argparse
import logging
import os
import multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from .utils import compute_alignment_scores
from .config import ALIGNERS
from ..common.corpus import get_scripture_parallel_corpus, tokenize_corpus

LOGGER = logging.getLogger(__name__)


def align_worker(kwargs):
    return align_set(**kwargs)


def align_set(src_input_path: Path, trg_input_path: Path, output_dir: Path, aligner: str = "fast_align"):
    if not src_input_path.exists():
        raise FileExistsError(f"The source file does not exist:{src_input_path}")
    if not trg_input_path.exists():
        raise FileExistsError(f"The target file does not exist:{trg_input_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    src_synced_path = output_dir / src_input_path.name
    trg_synced_path = output_dir / trg_input_path.name
    pcorp_df = get_scripture_parallel_corpus(src_input_path, trg_input_path, remove_empty_sentences=False)
    with src_synced_path.open("w+", encoding="utf-8") as source_file:
        source_file.write("\n".join(sentence for sentence in pcorp_df["source"]))
    with trg_synced_path.open("w+", encoding="utf-8") as target_file:
        target_file.write("\n".join(sentence for sentence in pcorp_df["target"]))

    scores = compute_alignment_scores(
        src_input_path=src_synced_path,
        trg_input_path=trg_synced_path,
        aligner_id=aligner,
        sym_align_path=output_dir / "sym-align.txt",
    )
    with (output_dir / "alignment.scores.txt").open("w+", encoding="utf-8") as as_file:
        as_file.writelines(["%0.2f\n" % s for s in scores])
    plt.plot(scores, "k.", markersize=2)
    plt.xlabel("Verses")
    plt.ylabel("Alignment Score")
    plt.savefig(output_dir / "alignment.png")


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


def process_alignments(src_path: Path, trg_paths: List[Path], output_dir: Path, aligner: str = "fast_align"):
    output_dir.mkdir(exist_ok=True)
    cpu_num = multiprocessing.cpu_count() // 2
    all_kwargs = []
    for trg_path in trg_paths:
        filename = trg_path.name
        name = os.path.splitext(filename)[0]
        f_dir = output_dir / name
        f_dir.mkdir(exist_ok=True)
        if (f_dir / "alignment.scores.txt").exists():
            LOGGER.info("Already aligned: " + name)
        else:
            all_kwargs.append(
                {"src_input_path": src_path, "trg_input_path": trg_path, "output_dir": f_dir, "aligner": aligner}
            )
    pool = multiprocessing.Pool(cpu_num)
    result = pool.map_async(align_worker, all_kwargs)
    result.get()
    pool.close()
    pool.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligns source bible to defined set of bibles")
    parser.add_argument("src_path", type=str, help="Path to source bible text")
    parser.add_argument("trg_dir", type=str, help="folder of bibles to align to")
    parser.add_argument("output_dir", type=str, help="folder to contain bible alignments")
    parser.add_argument("--aligner", type=str, default="fast_align", help="Aligner to use for extraction")
    args = parser.parse_args()

    if args.aligner not in ALIGNERS.keys():
        raise Exception("Need to use one of the following aligners:\n  " + "\n  ".join(ALIGNERS.keys()))
    if not os.path.exists(args.src_path):
        raise Exception("Source path does not exist:" + args.src_path)
    if not os.path.isdir(args.trg_dir):
        raise Exception("Target dir is not a real directory:" + args.output_dir)
    if not os.path.isdir(args.output_dir):
        raise Exception("Output dir is not a real directory:" + args.output_dir)

    src_basename = os.path.splitext(os.path.basename(args.src_path))[0]

    process_alignments(
        src_path=Path(args.src_path),
        trg_paths=list(Path(args.trg_dir).iterdir()),
        output_dir=Path(args.output_dir) / (args.aligner + "_" + src_basename),
        aligner=args.aligner,
    )


if __name__ == "__main__":
    main()