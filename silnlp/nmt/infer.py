import argparse
import logging
import os
import time
import tempfile
from typing import IO, Dict, List, Optional, Set, Tuple
import sentencepiece as sp

logging.basicConfig()

from ..common.utils import get_git_revision_hash, set_seed
from ..common.corpus import load_corpus, write_corpus
from .config import create_runner, get_mt_root_dir, load_config
from .utils import decode_sp_lines, encode_sp_lines, get_best_model_dir, get_last_checkpoint


def infer_checkpoint(
    root_dir: str,
    config: dict,
    srcFiles: List[str],
    trgFiles: List[str],
    memory_growth: bool,
    checkpoint_path: str,
    step: int,
):

    src_spp = sp.SentencePieceProcessor()
    src_spp.Load(os.path.join(root_dir, "src-sp.model"))

    checkpoint_name = "averaged checkpoint" if step == -1 else f"checkpoint {step}"
    print(f"Inferencing {checkpoint_name}...")

    runner = create_runner(config, memory_growth=memory_growth)

    with tempfile.TemporaryDirectory() as td:
        for srcFile, trgFile in zip(srcFiles, trgFiles):
            start = time.time()

            f = os.path.splitext(os.path.basename(srcFile))
            srcTokFile = os.path.join(td, f[0] + ".tok" + f[1])
            f = os.path.splitext(os.path.basename(trgFile))
            trgTokFile = os.path.join(td, f[0] + ".tok" + f[1])
            sentences = load_corpus(srcFile)
            write_corpus(srcTokFile, encode_sp_lines(src_spp, sentences))

            if os.path.basename(checkpoint_path) == "saved_model.pb":
                runner.saved_model_infer_multiple([srcTokFile], [trgTokFile])
            else:
                runner.infer_multiple([srcTokFile], [trgTokFile], checkpoint_path=checkpoint_path)

            sentences = load_corpus(trgTokFile)
            write_corpus(trgFile, decode_sp_lines(sentences))

            end = time.time()
            print(f"Inferenced {srcTokFile} to {trgTokFile} in {((end-start)/60):.2f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a NMT model using OpenNMT-tf")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use")
    parser.add_argument("--last", default=False, action="store_true", help="Test last checkpoint")
    parser.add_argument("--best", default=False, action="store_true", help="Test best evaluated checkpoint")
    parser.add_argument("--avg", default=False, action="store_true", help="Test averaged checkpoint")
    parser.add_argument("--src-prefix", default=None, type=str, help="Source file prefix (e.g., de-news2019-)")
    parser.add_argument("--trg-prefix", default=None, type=str, help="Target file prefix (e.g., en-news2019-)")
    parser.add_argument("--start-seq", default=None, type=int, help="Starting file sequence #")
    parser.add_argument("--end-seq", default=None, type=int, help="Ending file sequence #")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_mt_root_dir(exp_name)
    config = load_config(exp_name)
    model_dir: str = config["model_dir"]
    data_config: dict = config["data"]

    if args.src_prefix is None or args.trg_prefix is None:
        print("--src-prefix and --trg-prefix are required")
    if args.start_seq is None or args.end_seq is None:
        print("--start-seq and --end-seq are required")

    srcFiles: List[str] = []
    trgFiles: List[str] = []
    cwd = os.getcwd()
    for i in range(args.start_seq, args.end_seq + 1):
        fileNum = f"{i:04d}"
        srcFileName = os.path.join(cwd, f"{args.src_prefix}{fileNum}.txt")
        trgFileName = os.path.join(cwd, f"{args.trg_prefix}{fileNum}.txt")
        if os.path.isfile(srcFileName) and not os.path.isfile(trgFileName):
            srcFiles.append(srcFileName)
            trgFiles.append(trgFileName)

    best_model_path, best_step = get_best_model_dir(model_dir)
    step: int
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(model_dir, f"ckpt-{args.checkpoint}")
        step = int(args.checkpoint)
    elif args.avg:
        checkpoint_path, _ = get_last_checkpoint(os.path.join(model_dir, "avg"))
        step = -1
    elif args.best:
        step = best_step
        checkpoint_path = os.path.join(best_model_path, "ckpt")
        if not os.path.isfile(checkpoint_path + ".index"):
            checkpoint_path = os.path.join(model_dir, "saved_model.pb")
    elif args.last or (not args.best and args.checkpoint is None and not args.avg):
        checkpoint_path, step = get_last_checkpoint(model_dir)
    else:
        print("Must specify --checkpoint <step> or --last or --best or --avg")
        return

    set_seed(data_config["seed"])
    start = time.time()
    infer_checkpoint(root_dir, config, srcFiles, trgFiles, args.memory_growth, checkpoint_path, step)


if __name__ == "__main__":
    main()
