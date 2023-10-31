import argparse
import logging

import yaml
from machine.tokenization import (
    WHITESPACE_DETOKENIZER,
    WHITESPACE_TOKENIZER,
    Detokenizer,
    LatinWordDetokenizer,
    LatinWordTokenizer,
    RangeTokenizer,
    ZwspWordDetokenizer,
    ZwspWordTokenizer,
)
from machine.translation.thot import ThotWordAlignmentModelType

from ..common.utils import get_git_revision_hash, get_mt_exp_dir, merge_dict

LOGGER = logging.getLogger(__package__ + ".config")

_BASE_CONFIG: dict = {
    "model": "hmm",
    "seed": 111,
    "test_size": 250,
    "src_tokenizer": "latin",
    "trg_tokenizer": "latin",
}

_SUPPORTED_TOKENIZERS = {"whitespace", "latin", "zwsp"}


def load_config(exp_name: str) -> dict:
    exp_dir = get_mt_exp_dir(exp_name)
    config_path = exp_dir / "config.yml"

    config = _BASE_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)


def get_thot_word_alignment_type(word_alignment_type: str) -> ThotWordAlignmentModelType:
    if word_alignment_type == "fast_align" or word_alignment_type == "fastAlign":
        return ThotWordAlignmentModelType.FAST_ALIGN
    if word_alignment_type == "ibm1":
        return ThotWordAlignmentModelType.IBM1
    if word_alignment_type == "ibm2":
        return ThotWordAlignmentModelType.IBM2
    if word_alignment_type == "hmm":
        return ThotWordAlignmentModelType.HMM
    if word_alignment_type == "ibm3":
        return ThotWordAlignmentModelType.IBM3
    if word_alignment_type == "ibm4":
        return ThotWordAlignmentModelType.IBM4
    return ThotWordAlignmentModelType.HMM


def create_word_tokenizer(tokenizer_type: str) -> RangeTokenizer[str, int, str]:
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "latin":
        return LatinWordTokenizer()
    if tokenizer_type == "zwsp":
        return ZwspWordTokenizer()
    if tokenizer_type == "whitespace":
        return WHITESPACE_TOKENIZER
    return WHITESPACE_TOKENIZER


def create_word_detokenizer(tokenizer_type: str) -> Detokenizer[str, str]:
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "latin":
        return LatinWordDetokenizer()
    if tokenizer_type == "zwsp":
        return ZwspWordDetokenizer()
    if tokenizer_type == "whitespace":
        return WHITESPACE_DETOKENIZER
    return WHITESPACE_DETOKENIZER


def main() -> None:
    parser = argparse.ArgumentParser(description="Creates a NMT experiment config file")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--src-lang", type=str, required=True, help="Source language")
    parser.add_argument("--trg-lang", type=str, required=True, help="Target language")
    parser.add_argument("--src-tokenizer", type=str, required=False, help="Source language tokenizer")
    parser.add_argument("--trg-tokenizer", type=str, required=False, help="Target language tokenizer")
    parser.add_argument("--force", default=False, action="store_true", help="Overwrite existing config file")
    parser.add_argument("--seed", type=int, help="Randomization seed")
    parser.add_argument("--model", type=str, help="The word alignment model")
    args = parser.parse_args()

    get_git_revision_hash()

    exp_dir = get_mt_exp_dir(args.experiment)
    config_path = exp_dir / "config.yml"
    if config_path.is_file() and not args.force:
        LOGGER.warn(
            f'The experiment config file {config_path} already exists. Use "--force" if you want to overwrite the '
            + "existing config."
        )
        return

    exp_dir.mkdir(exist_ok=True, parents=True)

    config = _BASE_CONFIG.copy()
    if args.model is not None:
        config["model"] = args.model
    config["src_lang"] = args.src_lang
    config["trg_lang"] = args.trg_lang
    if args.src_tokenizer is not None and args.src_tokenizer in _SUPPORTED_TOKENIZERS:
        config["src_tokenizer"] = args.src_tokenizer
    if args.trg_tokenizer is not None and args.trg_tokenizer in _SUPPORTED_TOKENIZERS:
        config["trg_tokenizer"] = args.trg_tokenizer
    if args.seed is not None:
        config["seed"] = args.seed
    with config_path.open("w", encoding="utf-8") as file:
        yaml.dump(config, file)
    print(f"Config file created: {config_path}")


if __name__ == "__main__":
    main()
