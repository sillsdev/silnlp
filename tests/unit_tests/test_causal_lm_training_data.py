import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.causal_lm_trainer import DataCollatorForCausalLM
from silnlp.nmt.causal_lm_training_data import (
    CausalLMEncoder,
    CausalLMTrainingDataSets,
    InstructionMix,
    InterleavedTrainDataset,
    OverlengthRowFilter,
)
from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.example_retrieval import ExampleFormatterFactory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.experiment_languages import ExperimentLanguages
from silnlp.nmt.instruction_data import InstructionDataSettings
from silnlp.nmt.llm_config import PromptBuilder, PromptTemplate, PromptTemplateCollection
from silnlp.nmt.local_llm_prompts import LocalLLMPromptMessagesFactory, LocalLLMPrompts
from silnlp.nmt.model_name import ModelName

EOS = 99


class _CharacterTokenizer:
    """One id per character, so an encoded row's length shows which text it came from."""

    chat_template = "a template"
    eos_token_id = EOS
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
        return [ord(c) for message in messages for c in message["content"]]

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


def _encoder() -> CausalLMEncoder:
    tokenizer = _CharacterTokenizer()
    return CausalLMEncoder(tokenizer, DataCollatorForCausalLM(tokenizer))


def _data_sets(exp_dir: Path, max_seq_length: int = 100, mix_ratio: float = 0.5) -> CausalLMTrainingDataSets:
    environment = SilNlpEnv.create_environment_with_mt_dir(exp_dir)
    inventory = CorpusInventory([], include_glosses=False, environment=environment)
    template = PromptTemplate("", "{source}", ExampleFormatterFactory.create("text"))
    builder = PromptBuilder(PromptTemplateCollection([template]), 0, None, LocalLLMPromptMessagesFactory())
    instructions = InstructionMix(
        exp_dir / "instruction.jsonl",
        InstructionDataSettings({"datasets": [], "size": 0, "mix_ratio": mix_ratio}, exp_dir),
        num_devices=1,
        seed=0,
    )
    return CausalLMTrainingDataSets(
        ExperimentFiles(exp_dir, inventory, multi_ref_eval=False),
        LocalLLMPrompts(ModelName("google/gemma-2-2b-it"), builder),
        ExperimentLanguages({}, inventory),
        OverlengthRowFilter(max_seq_length),
        instructions,
    )


def _write(path: Path, lines) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _training_args(max_steps: int = 0) -> SimpleNamespace:
    return SimpleNamespace(max_steps=max_steps, per_device_train_batch_size=2, gradient_accumulation_steps=1)


def test_each_sentence_pair_becomes_its_prompt_followed_by_its_completion(tmp_path):
    _write(tmp_path / "train.src.txt", ["ab"])
    _write(tmp_path / "train.trg.txt", ["xyz"])

    row = _data_sets(tmp_path).translations(_encoder())[0]

    assert row["input_ids"] == [ord("a"), ord("b"), ord("x"), ord("y"), ord("z"), EOS]
    # The prompt is masked, so only the completion and its end of sequence are scored.
    assert row["labels"] == [-100, -100, ord("x"), ord("y"), ord("z"), EOS]
    assert row["attention_mask"] == [1] * 6


def test_there_is_no_training_set_without_training_files(tmp_path):
    assert _data_sets(tmp_path).translations(_encoder()) is None


def test_the_validation_set_is_read_from_the_validation_files(tmp_path):
    _write(tmp_path / "val.src.txt", ["a", "b"])
    _write(tmp_path / "val.trg.txt", ["c", "d"])

    assert len(_data_sets(tmp_path).validation(_encoder())) == 2


def test_rows_too_long_to_train_on_are_dropped(tmp_path):
    _write(tmp_path / "train.src.txt", ["a", "abcdefgh"])
    _write(tmp_path / "train.trg.txt", ["b", "ijklmnop"])

    rows = _data_sets(tmp_path, max_seq_length=5).translations(_encoder())

    assert [row["input_ids"] for row in rows] == [[ord("a"), ord("b"), EOS]]


def test_without_instruction_data_the_training_set_is_left_as_it_is(tmp_path):
    _write(tmp_path / "train.src.txt", ["a"])
    _write(tmp_path / "train.trg.txt", ["b"])
    data_sets = _data_sets(tmp_path)
    translations = data_sets.translations(_encoder())

    assert data_sets.mixed_with_instructions(translations, _encoder(), _training_args()) is translations


def test_instruction_data_is_interleaved_with_the_translations(tmp_path):
    _write(tmp_path / "train.src.txt", ["a", "b"])
    _write(tmp_path / "train.trg.txt", ["c", "d"])
    rows = [{"turns": [{"role": "user", "content": "say hi"}], "output": "hi"}]
    (tmp_path / "instruction.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    data_sets = _data_sets(tmp_path)

    mixed = data_sets.mixed_with_instructions(data_sets.translations(_encoder()), _encoder(), _training_args())

    assert isinstance(mixed, InterleavedTrainDataset)
    # Two translations plus half as many again from the instruction data.
    assert len(mixed) == 3


def test_a_zero_mix_ratio_leaves_the_instruction_data_out(tmp_path):
    _write(tmp_path / "train.src.txt", ["a"])
    _write(tmp_path / "train.trg.txt", ["b"])
    (tmp_path / "instruction.jsonl").write_text(json.dumps({"turns": [], "output": "x"}) + "\n", encoding="utf-8")
    data_sets = _data_sets(tmp_path, mix_ratio=0.0)
    translations = data_sets.translations(_encoder())

    assert data_sets.mixed_with_instructions(translations, _encoder(), _training_args()) is translations


def test_an_empty_instruction_file_counts_as_none(tmp_path):
    (tmp_path / "instruction.jsonl").write_text("", encoding="utf-8")
    mix = InstructionMix(
        tmp_path / "instruction.jsonl",
        InstructionDataSettings({"datasets": [], "size": 0, "mix_ratio": 0.5}, tmp_path),
        num_devices=1,
        seed=0,
    )

    assert mix.load() is None


def test_an_invalid_mix_ratio_is_reported_when_training_reads_it(tmp_path):
    data_sets = _data_sets(tmp_path, mix_ratio=2.0)

    with pytest.raises(ValueError, match="mix_ratio"):
        data_sets.mixed_with_instructions(None, _encoder(), _training_args())
