import argparse
import shutil

from silnlp.nmt.config_utils import load_config

_COMMON_TERMS = {"god", "lord", "yhwh", "yahweh", "jehovah", "jehovah god"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filters a dictionary")
    parser.add_argument("experiment", help="Experiment name")

    args = parser.parse_args()

    config = load_config(args.experiment)
    tokenizer = config.create_tokenizer()
    src_file_path = config.exp_dir / "dict.src.txt"
    orig_src_file_path = config.exp_dir / "dict.src.txt.orig"
    trg_file_path = config.exp_dir / "dict.trg.txt"
    orig_trg_file_path = config.exp_dir / "dict.trg.txt.orig"
    vref_file_path = config.exp_dir / "dict.vref.txt"
    orig_vref_file_path = config.exp_dir / "dict.vref.txt.orig"

    if not orig_src_file_path.is_file():
        shutil.copy(src_file_path, orig_src_file_path)
    if not orig_trg_file_path.is_file():
        shutil.copy(trg_file_path, orig_trg_file_path)
    if not orig_vref_file_path.is_file():
        shutil.copy(vref_file_path, orig_vref_file_path)

    with orig_src_file_path.open("r", encoding="utf-8-sig") as orig_src_file, orig_trg_file_path.open(
        "r", encoding="utf-8-sig"
    ) as orig_trg_file, orig_vref_file_path.open("r", encoding="utf-8-sig") as orig_vref_file, src_file_path.open(
        "w", encoding="utf-8"
    ) as src_file, trg_file_path.open(
        "w", encoding="utf-8"
    ) as trg_file, vref_file_path.open(
        "w", encoding="utf-8"
    ) as vref_file:
        for src, trg, vref in zip(orig_src_file, orig_trg_file, orig_vref_file):
            src = src.strip()
            trg = trg.strip()
            vref = vref.strip()

            trg_words = [tokenizer.detokenize(t).lower() for t in trg.split("\t")]
            if all(t not in _COMMON_TERMS for t in trg_words):
                src_file.write(src + "\n")
                trg_file.write(trg + "\n")
                vref_file.write(vref + "\n")


if __name__ == "__main__":
    main()
