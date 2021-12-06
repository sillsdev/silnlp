import os
import logging
from pathlib import Path

from silnlp.alignment.bulk_align import process_alignments

LOGGER = logging.getLogger("silnlp")

# python -m silnlp.alignment.bulk_align "C:\Users\johnm\Documents\repos\bible-parallel-corpus-internal\corpus\scripture\hbo-HEB.txt" "C:\Users\johnm\Documents\repos\bible-parallel-corpus-internal\alignments\targets" "C:\Users\johnm\Documents\repos\bible-parallel-corpus-internal\alignments"

if __name__ == "__main__":
    alignment_dir = Path("C:\\Users\\johnm\\Documents\\repos\\bible-parallel-corpus-internal\\alignments")
    target_dir = alignment_dir / "targets"
    output_dir = alignment_dir
    aligner = "hmm"
    src_filename = "hbo-HEB.txt"
    src_path = alignment_dir / src_filename
    src_basename = os.path.splitext("hbo-HEB.txt")[0]

    process_alignments(
        src_path=src_path,
        trg_paths=list(target_dir.glob("*.txt")),
        output_dir=output_dir / (aligner + "_" + src_basename),
    )
