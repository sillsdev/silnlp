#!/usr/bin/env python3

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Set

from transformers import AutoTokenizer

ARTIFACT_FILENAME = "partial_word_first_step_allowed_token_ids.json.gz"


def normalize_partial_word(partial_word: str) -> str:
    # With SentencePiece tokenizers, leading spaces are represented with U+2581.
    if partial_word.startswith(" "):
        stripped = partial_word.lstrip(" ")
        return "▁" + stripped
    return partial_word


def get_default_tokenizer_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "silnlp" / "assets" / "tokenizers" / "facebook" / "nllb-200"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute first-step partial-word token constraints for an NLLB tokenizer."
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=get_default_tokenizer_dir(),
        help="Path to tokenizer directory containing tokenizer.json/tokenizer_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output artifact path. Defaults to <tokenizer-dir>/partial_word_first_step_allowed_token_ids.json.gz",
    )
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir.resolve()
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_dir}")

    output_path = args.output.resolve() if args.output is not None else tokenizer_dir / ARTIFACT_FILENAME

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    token_ids = sorted(set(tokenizer.get_vocab().values()))
    all_special_ids = set(tokenizer.all_special_ids)

    text_to_token_ids: DefaultDict[str, List[int]] = defaultdict(list)
    prefix_to_token_ids: DefaultDict[str, List[int]] = defaultdict(list)
    partial_word_candidates: Set[str] = set()

    for token_id in token_ids:
        if token_id in all_special_ids:
            continue

        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is None or len(token) == 0:
            continue

        text = token
        if len(text) == 0:
            continue
        text_to_token_ids[text].append(token_id)
        for i in range(1, len(text) + 1):
            prefix = text[:i]
            prefix_to_token_ids[prefix].append(token_id)
            partial_word_candidates.add(prefix)

    allowed_token_ids_by_partial_word: Dict[str, List[int]] = {}
    for partial_word in sorted(partial_word_candidates):
        normalized_partial_word = normalize_partial_word(partial_word)
        allowed_ids: Set[int] = set(prefix_to_token_ids.get(normalized_partial_word, []))
        for i in range(1, len(normalized_partial_word) + 1):
            allowed_ids.update(text_to_token_ids.get(normalized_partial_word[:i], []))

        if len(allowed_ids) > 0:
            allowed_token_ids_by_partial_word[normalized_partial_word] = sorted(allowed_ids)

    artifact = {
        "schema_version": 2,
        "tokenizer_dir": str(tokenizer_dir),
        "token_count": len(token_ids),
        "partial_word_count": len(allowed_token_ids_by_partial_word),
        "text_to_token_ids": text_to_token_ids,
        "prefix_to_token_ids": prefix_to_token_ids,
        "allowed_token_ids_by_partial_word": allowed_token_ids_by_partial_word,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=False, indent=2)

    print(f"Wrote: {output_path}")
    print(f"Token count: {len(token_ids)}")
    print(f"Partial words indexed: {len(allowed_token_ids_by_partial_word)}")


if __name__ == "__main__":
    main()
