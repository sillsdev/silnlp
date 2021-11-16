import logging
import multiprocessing
from pathlib import Path

from align_helper import process_alignments

LOGGER = logging.getLogger("silnlp")


if __name__ == "__main__":
    scripture_dir = Path("C:\\Users\\johnm\\Documents\\repos\\bible-parallel-corpus-internal\\corpus\\scripture")
    alignment_dir = scripture_dir / "..\\..\\alignments"
    # complete_files = full_bibles(scripture_dir)
    # (alignment_dir / "alignment_sources.txt").open('w+').writelines([f'{p.name}\n' for p in complete_files])
    complete_files = (alignment_dir / "alignment_sources.txt").open().readlines()

    src_path = alignment_dir / "hbo-HEB.txt"
    process_alignments(
        scripture_dir=scripture_dir,
        alignment_dir=alignment_dir,
        src_path=src_path,
        complete_files=complete_files,
        suffix="_HEB",
    )

    src_path = alignment_dir / "grc-GRK.txt"
    process_alignments(
        scripture_dir=scripture_dir,
        alignment_dir=alignment_dir,
        src_path=src_path,
        complete_files=complete_files,
        suffix="_GRK",
    )
