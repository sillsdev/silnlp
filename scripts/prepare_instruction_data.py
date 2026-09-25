"""Downloads general instruction-following datasets and reformats each into a JSON Lines file
(<name>.jsonl) under the MT "instructions" directory, so they can be mixed into LLM fine-tuning
batches alongside translation data without touching the existing translation corpora.

Each line is a {"turns": [{"role": ..., "content": ...}, ...], "output": ...} object: "turns" is
the real, role-tagged conversation history (using the standard system/user/assistant vocabulary),
and "output" is the assistant response to train on. Keeping turns structured -- rather than
flattening history into a single string -- lets training render it through the model's actual
chat template, so it matches what a real multi-turn conversation looks like at inference time.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

from silnlp.common.environment import SilNlpEnv


def _normalize(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def _iter_turn_examples(
    turns: List[dict],
    role_key: str,
    content_key: str,
    assistant_role: str,
    role_labels: Dict[str, str],
    prefix_turns: List[Dict[str, str]] = (),
) -> Iterator[Tuple[List[Dict[str, str]], str]]:
    """Yield one (turns, output) pair per assistant turn in a conversation: ``turns`` is the real
    prior history up to that point (role names normalized to system/user/assistant) and
    ``output`` is that turn's assistant response -- rather than collapsing the whole conversation
    into a single example that only ever trains on the final turn."""
    for i, turn in enumerate(turns):
        if turn[role_key] != assistant_role:
            continue
        history = turns[:i]
        if not history:
            continue
        context = list(prefix_turns) + [
            {"role": role_labels.get(m[role_key], m[role_key]), "content": m[content_key]} for m in history
        ]
        yield context, turn[content_key]


def _write_jsonl(examples: Iterator[Tuple[List[Dict[str, str]], str]], output_dir: Path, name: str) -> None:
    output_path = output_dir / f"{name}.jsonl"
    count = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out_file:
        for turns, output_text in examples:
            normalized_turns = [{"role": t["role"], "content": _normalize(t["content"])} for t in turns]
            normalized_output = _normalize(output_text)
            if not normalized_turns or not normalized_output or any(not t["content"] for t in normalized_turns):
                skipped += 1
                continue
            out_file.write(
                json.dumps({"turns": normalized_turns, "output": normalized_output}, ensure_ascii=False) + "\n"
            )
            count += 1
    print(f"{name}: wrote {count} examples to {output_path.name} ({skipped} skipped as empty)")


def _iter_dolly() -> Iterator[Tuple[List[Dict[str, str]], str]]:
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    df = pd.read_json(path, lines=True)
    for row in df.itertuples():
        instruction = row.instruction
        if row.context:
            instruction = f"{instruction}\n\n{row.context}"
        yield [{"role": "user", "content": instruction}], row.response


def _iter_no_robots(exclude_multi_turn: bool = False) -> Iterator[Tuple[List[Dict[str, str]], str]]:
    # train_sft: the split HuggingFaceH4 recommends for supervised fine-tuning. Roles are already
    # the standard system/user/assistant vocabulary, so no role_labels remapping is needed.
    path = hf_hub_download("HuggingFaceH4/no_robots", "data/train_sft-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    for messages in df["messages"]:
        messages = list(messages)
        system_turns = [{"role": "system", "content": m["content"]} for m in messages if m["role"] == "system"]
        turns = [m for m in messages if m["role"] != "system"]
        is_multi_turn = len(turns) > 2 or bool(system_turns)
        if is_multi_turn and exclude_multi_turn:
            continue
        yield from _iter_turn_examples(
            turns,
            role_key="role",
            content_key="content",
            assistant_role="assistant",
            role_labels={},
            prefix_turns=system_turns,
        )


def _iter_aya_dataset() -> Iterator[Tuple[List[Dict[str, str]], str]]:
    path = hf_hub_download("CohereForAI/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    for row in df.itertuples():
        yield [{"role": "user", "content": row.inputs}], row.targets


_TOWERBLOCKS_ROLE_LABELS = {"human": "user", "gpt": "assistant"}


def _iter_towerblocks(exclude_multi_turn: bool = False) -> Iterator[Tuple[List[Dict[str, str]], str]]:
    # Unbabel's own mix of general chat (ultrachat, code assistance) with translation-adjacent
    # tasks (MT, post-editing, MT evaluation, NER, terminology, ...), used to fine-tune TowerLLM
    # into a strong translator without losing general instruction-following ability -- the exact
    # scenario this "instructions" directory exists for.
    for shard in range(4):
        path = hf_hub_download(
            "Unbabel/TowerBlocks-v0.2", f"data/train-0000{shard}-of-00004.parquet", repo_type="dataset"
        )
        df = pd.read_parquet(path)
        for conversation in df["conversations"]:
            turns = list(conversation)
            is_multi_turn = len(turns) > 2
            if is_multi_turn and exclude_multi_turn:
                continue
            yield from _iter_turn_examples(
                turns, role_key="from", content_key="value", assistant_role="gpt", role_labels=_TOWERBLOCKS_ROLE_LABELS
            )


DATASETS = {
    "dolly": lambda exclude_multi_turn: _iter_dolly(),
    "no_robots": lambda exclude_multi_turn: _iter_no_robots(exclude_multi_turn),
    "aya_dataset": lambda exclude_multi_turn: _iter_aya_dataset(),
    "towerblocks": lambda exclude_multi_turn: _iter_towerblocks(exclude_multi_turn),
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
        "(only affects no_robots and towerblocks; the other datasets have no multi-turn examples)",
    )
    args = parser.parse_args()

    environment = SilNlpEnv.create_standard_environment()
    output_dir = environment.mt_dir / "instructions"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        _write_jsonl(DATASETS[name](args.exclude_multi_turn), output_dir, name)


if __name__ == "__main__":
    main()
