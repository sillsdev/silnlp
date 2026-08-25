"""Downloads general instruction-following datasets and reformats each into a pair of
line-aligned parallel text files (<name>.input.txt / <name>.output.txt) under the MT
"instructions" directory, so they can be mixed into LLM fine-tuning batches alongside
translation data without touching the existing translation corpora.
"""

import argparse
from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

from silnlp.common.environment import SilNlpEnv


def _normalize(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def _write_pairs(pairs: Iterator[Tuple[str, str]], output_dir: Path, name: str) -> None:
    input_path = output_dir / f"{name}.input.txt"
    output_path = output_dir / f"{name}.output.txt"
    count = 0
    skipped = 0
    with (
        input_path.open("w", encoding="utf-8", newline="\n") as input_file,
        output_path.open("w", encoding="utf-8", newline="\n") as output_file,
    ):
        for raw_input, raw_output in pairs:
            input_text = _normalize(raw_input)
            output_text = _normalize(raw_output)
            if not input_text or not output_text:
                skipped += 1
                continue
            input_file.write(input_text + "\n")
            output_file.write(output_text + "\n")
            count += 1
    print(f"{name}: wrote {count} pairs to {input_path.name} / {output_path.name} ({skipped} skipped as empty)")


def _iter_dolly() -> Iterator[Tuple[str, str]]:
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    df = pd.read_json(path, lines=True)
    for row in df.itertuples():
        instruction = row.instruction
        if row.context:
            instruction = f"{instruction}\n\n{row.context}"
        yield instruction, row.response


def _iter_no_robots(exclude_multi_turn: bool = False) -> Iterator[Tuple[str, str]]:
    # train_sft: the split HuggingFaceH4 recommends for supervised fine-tuning.
    path = hf_hub_download("HuggingFaceH4/no_robots", "data/train_sft-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    for messages in df["messages"]:
        messages = list(messages)
        if not messages or messages[-1]["role"] != "assistant":
            continue
        output_text = messages[-1]["content"]
        history = messages[:-1]
        system_lines = [m["content"] for m in history if m["role"] == "system"]
        turns = [m for m in history if m["role"] != "system"]
        is_multi_turn = len(turns) != 1 or bool(system_lines)
        if is_multi_turn and exclude_multi_turn:
            continue
        if is_multi_turn:
            # Multi-turn conversation: flatten history into role-prefixed lines rather than
            # dropping the example, so the input still carries the prior turns as context.
            lines = list(system_lines) + [f"{m['role'].capitalize()}: {m['content']}" for m in turns]
            input_text = "\n".join(lines)
        else:
            # Single-turn (the common case): input is just the bare user message, matching
            # the shape of the other datasets and the translation prompts it will be mixed with.
            input_text = turns[0]["content"]
        yield input_text, output_text


def _iter_aya_dataset() -> Iterator[Tuple[str, str]]:
    path = hf_hub_download("CohereForAI/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    for row in df.itertuples():
        yield row.inputs, row.targets


DATASETS = {
    "dolly": lambda exclude_multi_turn: _iter_dolly(),
    "no_robots": lambda exclude_multi_turn: _iter_no_robots(exclude_multi_turn),
    "aya_dataset": lambda exclude_multi_turn: _iter_aya_dataset(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS), help="Datasets to prepare"
    )
    parser.add_argument(
        "--exclude-multi-turn",
        action="store_true",
        help="Drop multi-turn conversations, keeping only single-turn instruction/response pairs "
        "(only affects no_robots; the other datasets have no multi-turn examples)",
    )
    args = parser.parse_args()

    environment = SilNlpEnv.create_standard_environment()
    output_dir = environment.mt_dir / "instructions"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        _write_pairs(DATASETS[name](args.exclude_multi_turn), output_dir, name)


if __name__ == "__main__":
    main()
