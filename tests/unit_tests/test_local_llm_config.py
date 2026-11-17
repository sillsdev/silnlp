import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from jinja2.exceptions import UndefinedError

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.causal_lm_generator import CausalLMGenerator
from silnlp.nmt.causal_lm_provider import CausalLMModelSettings, CausalLMProvider
from silnlp.nmt.causal_lm_tokenizer import CausalLMTokenizer
from silnlp.nmt.causal_lm_trainer import DataCollatorForCausalLM
from silnlp.nmt.causal_lm_training_data import (
    CausalLMEncoder,
    InstructionMix,
    InterleavedEpochCallback,
    InterleavedTrainDataset,
    OverlengthRowFilter,
)
from silnlp.nmt.checkpoints import CheckpointDirectory, CheckpointType
from silnlp.nmt.config_utils import ConfiguredModelType
from silnlp.nmt.example_retrieval import Example, ExampleFormatterFactory, ExamplePools
from silnlp.nmt.experiment_languages import Language
from silnlp.nmt.experiment_settings import TrainerSettings
from silnlp.nmt.finetune_method import FinetuneMethod
from silnlp.nmt.finetuning import AdapterSettings
from silnlp.nmt.generation_settings import GenerationSettings
from silnlp.nmt.instruction_data import InstructionDataSettings
from silnlp.nmt.llm_config import PromptBuilder, PromptMessages, PromptTemplate, PromptTemplateCollection
from silnlp.nmt.local_llm_config import LocalLLMConfig, LocalLLMModel
from silnlp.nmt.local_llm_prompts import (
    LocalLLMPromptMessages,
    LocalLLMPromptMessagesFactory,
    LocalLLMPrompts,
    TranslateGemmaPromptMessages,
)
from silnlp.nmt.model_name import ModelName


def is_local_llm(config: dict) -> bool:
    return ConfiguredModelType(config).is_local_llm()


def test_is_local_llm_config_explicit_model_type():
    assert is_local_llm({"model_type": "local_llm", "model": "anything"})
    assert is_local_llm({"model_type": "llm", "model": "anything"})
    assert not is_local_llm({"model_type": "nmt", "model": "google/gemma-2-2b-it"})
    assert not is_local_llm({"model_type": "remote_llm", "model": "anything"})


def test_is_local_llm_config_prefix_fallback():
    assert is_local_llm({"model": "google/gemma-2-2b-it"})
    assert is_local_llm({"model": "tencent/Hunyuan-MT-7B"})
    assert not is_local_llm({"model": "facebook/nllb-200-distilled-1.3B"})
    assert not is_local_llm({"model": "google/madlad400-3b-mt"})


def test_prompt_messages_to_chat_messages():
    prompt = LocalLLMPromptMessages(
        system_message="You are a translator.", instruction="Translate: hello", target="bonjour"
    )
    assert prompt.to_chat_messages() == [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate: hello"},
        {"role": "assistant", "content": "bonjour"},
    ]


def test_prompt_messages_folds_system_message_into_user_turn():
    prompt = LocalLLMPromptMessages(system_message="You are a translator.", instruction="Translate: hello")
    assert prompt.to_folded_chat_messages() == [
        {"role": "user", "content": "You are a translator.\n\nTranslate: hello"}
    ]


def test_prompt_messages_without_system_message():
    prompt = LocalLLMPromptMessages(system_message="", instruction="Translate: hello")
    assert prompt.to_chat_messages() == [{"role": "user", "content": "Translate: hello"}]
    assert prompt.to_folded_chat_messages() == [{"role": "user", "content": "Translate: hello"}]


def test_translate_gemma_prompt_messages_is_a_prompt_messages():
    prompt = TranslateGemmaPromptMessages(
        source_language=Language("en", "English"), target_language=Language("fr", "French"), text="hello"
    )
    assert isinstance(prompt, PromptMessages)


def test_translate_gemma_prompt_messages_has_no_folding_or_plain_text_fallback():
    prompt = TranslateGemmaPromptMessages(
        source_language=Language("en", "English"), target_language=Language("fr", "French"), text="hello"
    )
    with pytest.raises(NotImplementedError):
        prompt.to_folded_chat_messages()
    with pytest.raises(NotImplementedError):
        prompt.to_plain_text()


@dataclass
class _StubTokenizer:
    pad_token_id: int = 0


def test_data_collator_right_pads_inputs_and_masks_label_padding():
    collator = DataCollatorForCausalLM(_StubTokenizer(pad_token_id=0))
    features = [
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "attention_mask": [1, 1, 1]},
        {"input_ids": [8, 9], "labels": [-100, 9], "attention_mask": [1, 1]},
    ]
    batch = collator(features)

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    # Padding positions in labels are masked with -100 so they are ignored by the loss.
    assert batch["labels"].tolist() == [[-100, 6, 7], [-100, 9, -100]]


def test_generation_by_beam_search():
    settings = GenerationSettings({"max_new_tokens": 256, "num_beams": 4, "do_sample": False, "temperature": 0.7})
    gen_kwargs = settings.as_keyword_arguments(num_return_sequences=2, pad_token_id=0)
    assert gen_kwargs["num_beams"] == 4
    assert gen_kwargs["num_return_sequences"] == 2
    assert "do_sample" not in gen_kwargs
    assert "temperature" not in gen_kwargs


def test_generation_by_sampling_does_not_set_num_beams():
    settings = GenerationSettings({"max_new_tokens": 256, "num_beams": 4, "do_sample": True, "temperature": 0.7})
    gen_kwargs = settings.as_keyword_arguments(num_return_sequences=3, pad_token_id=0)
    assert gen_kwargs["do_sample"] is True
    assert gen_kwargs["temperature"] == 0.7
    assert gen_kwargs["num_return_sequences"] == 3
    assert "num_beams" not in gen_kwargs


def test_generation_by_beam_search_rejects_more_drafts_than_beams():
    settings = GenerationSettings({"max_new_tokens": 256, "num_beams": 1, "do_sample": False, "temperature": 0.7})
    with pytest.raises(RuntimeError, match="num_beams"):
        settings.as_keyword_arguments(num_return_sequences=2, pad_token_id=0)


def _encoded_rows(*lengths: int) -> Dataset:
    return Dataset.from_dict({"input_ids": [list(range(length)) for length in lengths]})


def test_overlength_row_filter_drops_only_rows_longer_than_max_seq_length():
    kept = OverlengthRowFilter(4).apply(_encoded_rows(3, 5, 4, 9), "train")
    assert [len(ids) for ids in kept["input_ids"]] == [3, 4]


def test_overlength_row_filter_warns_about_each_dropped_row(caplog):
    with caplog.at_level(logging.WARNING):
        OverlengthRowFilter(4).apply(_encoded_rows(3, 5, 4, 9), "val")
    warnings = [record.message for record in caplog.records if "Dropping" in record.message]
    assert warnings == [
        "Dropping val row 1: its 5 tokens exceed params.max_seq_length (4).",
        "Dropping val row 3: its 9 tokens exceed params.max_seq_length (4).",
    ]


def test_overlength_row_filter_rejects_a_dataset_with_no_row_short_enough():
    with pytest.raises(RuntimeError, match="All 2 instruction rows"):
        OverlengthRowFilter(4).apply(_encoded_rows(5, 9), "instruction")


def test_data_collator_pad_to_multiple_of():
    collator = DataCollatorForCausalLM(_StubTokenizer(pad_token_id=0), pad_to_multiple_of=4)
    features = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "attention_mask": [1, 1, 1]}]
    batch = collator(features)
    assert batch["input_ids"].shape[1] == 4
    assert batch["labels"].tolist() == [[-100, 6, 7, -100]]


def _prompts(model="google/gemma-2-2b-it", **prompt_overrides) -> LocalLLMPrompts:
    return LocalLLMPrompts(ModelName(model), _prompt_builder(**prompt_overrides))


ENGLISH = Language("en", "English")
FRENCH = Language("fr", "French")


def _prompt_builder(
    instruction_template="Translate from {src_lang} to {trg_lang}.\n\n{source}",
    system_message="",
    num_examples=0,
    pool=None,
    example_format="text",
):
    template = PromptTemplate(system_message, instruction_template, ExampleFormatterFactory.create(example_format))
    return PromptBuilder(PromptTemplateCollection([template]), num_examples, pool, LocalLLMPromptMessagesFactory())


class _FakePool:
    """Verifies delegation, not retrieval (see test_example_retrieval.py for that)."""

    def __init__(self, examples=()):
        self._examples = list(examples)
        self.calls = []

    def __len__(self):
        return len(self._examples)

    def covers_whole_pool(self, k):
        return False

    def select(self, query, k, pool_index=None):
        self.calls.append((query, k, pool_index))
        return self._examples


def test_build_prompt_messages_translate_gemma_uses_structured_content():
    prompts = _prompts(model="google/translategemma-4b-it")
    prompt = prompts.build("hello", Language("en", "en"), Language("fr", "fr"), target="bonjour")
    assert prompt == TranslateGemmaPromptMessages(
        source_language=Language("en", "en"), target_language=Language("fr", "fr"), text="hello", target="bonjour"
    )
    assert isinstance(prompt, TranslateGemmaPromptMessages)
    assert prompt.to_chat_messages() == [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "fr", "text": "hello"}],
        },
        {"role": "assistant", "content": "bonjour"},
    ]


def test_build_prompt_messages_generic_model_uses_instruction_template():
    prompt = _prompts().build("hello", ENGLISH, FRENCH)
    assert prompt == LocalLLMPromptMessages(
        system_message="", instruction="Translate from English to French.\n\nhello", target=None
    )


def test_build_prompt_messages_zero_examples_matches_legacy_output_exactly():
    # Pins the exact zero-shot text, since existing fine-tuned checkpoints depend on it.
    prompts = _prompts(instruction_template=LocalLLMConfig.DEFAULT_INSTRUCTION_TEMPLATE)
    prompt = prompts.build("hello", ENGLISH, FRENCH)
    assert prompt.instruction == "Translate the following text from English to French.\n\nhello"


def test_build_prompt_messages_splices_rendered_examples_before_source():
    pool = _FakePool([Example("cat", "chat"), Example("dog", "chien")])
    prompts = _prompts(
        instruction_template="Translate from {src_lang} to {trg_lang}.\n\n{examples}{source}",
        num_examples=2,
        pool=pool,
        example_format={"type": "text", "template": "Source: {source}\nTarget: {target}\n\n"},
    )
    prompt = prompts.build("hello", ENGLISH, FRENCH)
    assert prompt.instruction == (
        "Translate from English to French.\n\n" "Source: cat\nTarget: chat\n\n" "Source: dog\nTarget: chien\n\n" "hello"
    )
    assert pool.calls == [("hello", 2, None)]


def test_build_prompt_messages_passes_pool_index_through_for_training():
    pool = _FakePool()
    prompts = _prompts(instruction_template="{examples}{source}", num_examples=2, pool=pool)
    prompts.build("hello", ENGLISH, FRENCH, example_pool_index=3)
    assert pool.calls == [("hello", 2, 3)]


class _StubTranslateGemmaTokenizer:
    chat_template = "{# a real chat template would render this #}"
    bos_token = "<bos>"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
        # Mimics the real template's behavior for a language code outside its fixed lookup table.
        raise UndefinedError("'dict object' has no attribute 'tst'")

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


def test_apply_prompt_template_translate_gemma_falls_back_for_unrecognized_language_code():
    tokenizer = _StubTranslateGemmaTokenizer()
    prompt = _prompts(model="google/translategemma-4b-it").build("hello", ENGLISH, Language("tst", "Test Language"))

    text = prompt.apply_prompt_template(tokenizer, add_generation_prompt=True, tokenize=False)
    assert text == (
        "<bos><start_of_turn>user\n"
        "You are a professional English (en) to Test Language (tst) translator. Your goal is to accurately convey "
        "the meaning and nuances of the original English text while adhering to Test Language grammar, vocabulary, "
        "and cultural sensitivities.\n"
        "Produce only the Test Language translation, without any additional explanations or commentary. Please "
        "translate the following English text into Test Language:\n\n\nhello<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    token_ids = prompt.apply_prompt_template(tokenizer, add_generation_prompt=True, tokenize=True)
    assert token_ids == [ord(c) for c in text]


def _construct_llm_config(
    tmp_path: Path,
    prompt_overrides: dict,
    model: str = "google/gemma-2-2b-it",
    section: str = "train",
) -> LocalLLMConfig:
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path)
    return LocalLLMConfig(
        tmp_path,
        {"data": {"corpus_pairs": []}, "model": model, section: {"prompt": prompt_overrides}},
        environment,
    )


def test_llm_config_rejects_translate_gemma_with_num_examples(tmp_path):
    with pytest.raises(RuntimeError, match="TranslateGemma"):
        _construct_llm_config(tmp_path, {"num_examples": 2}, model="google/translategemma-4b-it")


def test_llm_config_warns_when_num_examples_set_without_examples_placeholder(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _construct_llm_config(
            tmp_path, {"num_examples": 2, "instruction_template": "Translate {src_lang} to {trg_lang}: {source}"}
        )
    assert any("{examples}" in record.message for record in caplog.records)


def _construct_llm_config_with_params(tmp_path: Path, params: dict) -> LocalLLMConfig:
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path)
    return LocalLLMConfig(tmp_path, {"data": {"corpus_pairs": []}, "params": params}, environment)


def test_llm_config_warns_about_params_prompt(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _construct_llm_config_with_params(tmp_path, {"prompt": {"num_examples": 2}})
    assert any("params.prompt" in record.message for record in caplog.records)


def test_llm_config_does_not_warn_about_params_prompt_when_absent(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _construct_llm_config_with_params(tmp_path, {"finetune_method": "lora"})
    assert not any("params.prompt" in record.message for record in caplog.records)


def test_llm_config_rejects_unknown_example_selection_method(tmp_path):
    with pytest.raises(ValueError, match="Unknown example_selection.method"):
        _construct_llm_config(
            tmp_path,
            {
                "num_examples": 2,
                "instruction_template": "Translate {src_lang} to {trg_lang}.\n\n{examples}{source}",
                "example_selection": {"method": "bogus"},
            },
        )


def test_build_adapter_config_plain_lora():
    peft_config = AdapterSettings(
        {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"}
    ).lora_config(use_dora=False)
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.modules_to_save is None
    assert peft_config.use_dora is False


def test_build_adapter_config_passes_through_modules_to_save():
    peft_config = AdapterSettings(
        {
            "rank": 64,
            "alpha": 256,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "modules_to_save": ["embed_tokens", "lm_head"],
        }
    ).lora_config(use_dora=False)
    assert peft_config.r == 64
    assert peft_config.lora_alpha == 256
    assert peft_config.modules_to_save == ["embed_tokens", "lm_head"]


def test_build_adapter_config_dora():
    adapter = AdapterSettings({"rank": 64, "alpha": 256, "dropout": 0.05, "target_modules": "all-linear"})
    assert adapter.lora_config(use_dora=True).use_dora is True


def test_a_configured_lora_key_becomes_the_adapter_the_model_is_built_with(tmp_path):
    config = _construct_llm_config_with_params(tmp_path, {"lora": {"rank": 8}})
    assert config.params["adapter"]["rank"] == 8
    assert "lora" not in config.params


def _instruction_data(section: dict, mt_dir: Path = Path("mt")) -> InstructionDataSettings:
    return InstructionDataSettings({"datasets": [], "size": 100000, "mix_ratio": 0.1, **section}, mt_dir)


def test_instruction_datasets_defaults_to_empty():
    settings = _instruction_data({})
    assert settings.size() == 100000
    assert settings.paths() == []


def test_instruction_data_paths_resolved_under_mt_dir_instructions(tmp_path):
    settings = _instruction_data({"datasets": ["dolly", "no_robots"]}, tmp_path)
    assert settings.paths() == [
        tmp_path / "instructions" / "dolly.jsonl",
        tmp_path / "instructions" / "no_robots.jsonl",
    ]


def test_instruction_data_size_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _instruction_data({"size": -1}).size()


def test_instruction_mix_ratio_default():
    assert _instruction_data({}).mix_ratio() == 0.1


def test_instruction_mix_ratio_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _instruction_data({"mix_ratio": -0.1}).mix_ratio()


def test_instruction_mix_ratio_rejects_more_than_the_whole_set():
    with pytest.raises(ValueError, match="no greater than 1"):
        _instruction_data({"mix_ratio": 1.5}).mix_ratio()


class _StubRejectingSystemTokenizer:
    chat_template = "some template"
    eos_token_id = 99

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [len(text)]}

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
        assert tokenize is True
        assert return_dict is False
        if any(m["role"] == "system" for m in messages):
            raise ValueError("this template does not support a separate system role")
        return [len(m["content"]) for m in messages]


def _encoded_turns(tokenizer, turns, completion="ok") -> dict:
    return CausalLMEncoder(tokenizer, DataCollatorForCausalLM(tokenizer)).turns_with_completion(turns, completion)


def test_render_turns_happy_path():
    encoded = _encoded_turns(_StubRejectingSystemTokenizer(), [{"role": "user", "content": "hi"}])
    assert encoded["input_ids"] == [2, len("ok"), 99]
    # Only the completion is scored.
    assert encoded["labels"] == [-100, len("ok"), 99]


def test_render_turns_folds_leading_system_turn_into_first_user_turn_on_failure():
    turns = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    encoded = _encoded_turns(_StubRejectingSystemTokenizer(), turns)
    # folded: [{"role": "user", "content": "sys\n\nhi"}, {"role": "assistant", "content": "yo"}]
    assert encoded["input_ids"][:2] == [len("sys\n\nhi"), len("yo")]


class _AlwaysFailingTokenizer:
    chat_template = "some template"
    eos_token_id = 99

    def apply_chat_template(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_render_turns_reraises_when_there_is_no_system_turn_to_fold():
    with pytest.raises(RuntimeError, match="boom"):
        _encoded_turns(_AlwaysFailingTokenizer(), [{"role": "user", "content": "hi"}])


def test_render_turns_reraises_when_system_turn_has_nothing_to_fold_into():
    with pytest.raises(RuntimeError, match="boom"):
        _encoded_turns(_AlwaysFailingTokenizer(), [{"role": "system", "content": "sys"}])


def _make_tagged_dataset(prefix: str, size: int) -> Dataset:
    return Dataset.from_dict({"input_ids": [[i] for i in range(size)], "tag": [f"{prefix}{i}" for i in range(size)]})


def test_interleaved_train_dataset_length_is_sum_of_both_counts():
    dataset = InterleavedTrainDataset(
        _make_tagged_dataset("t", 5), _make_tagged_dataset("i", 100), translation_count=12, instruction_count=7, seed=0
    )
    assert len(dataset) == 19


def test_interleaved_train_dataset_routes_indices_to_the_right_pool():
    dataset = InterleavedTrainDataset(
        _make_tagged_dataset("t", 5), _make_tagged_dataset("i", 100), translation_count=12, instruction_count=7, seed=0
    )
    tags = [dataset[i]["tag"] for i in range(len(dataset))]
    assert all(tag.startswith("t") for tag in tags[:12])
    assert all(tag.startswith("i") for tag in tags[12:])


def test_interleaved_train_dataset_does_not_repeat_within_a_lap():
    # 12 translation slots from a pool of 5 is 2 full laps (0-4, 5-9) plus a partial lap (10-11);
    # each full lap must be a permutation of all 5 rows -- no repeats until the pool is exhausted.
    dataset = InterleavedTrainDataset(
        _make_tagged_dataset("t", 5), _make_tagged_dataset("i", 100), translation_count=12, instruction_count=0, seed=0
    )
    tags = [dataset[i]["tag"] for i in range(12)]
    assert sorted(tags[0:5]) == sorted(f"t{i}" for i in range(5))
    assert sorted(tags[5:10]) == sorted(f"t{i}" for i in range(5))
    assert set(tags[10:12]).issubset({f"t{i}" for i in range(5)})


def test_interleaved_train_dataset_instruction_pool_does_not_repeat_when_it_fits():
    dataset = InterleavedTrainDataset(
        _make_tagged_dataset("t", 5), _make_tagged_dataset("i", 20), translation_count=0, instruction_count=20, seed=0
    )
    tags = [dataset[i]["tag"] for i in range(20)]
    assert sorted(tags) == sorted(f"i{i}" for i in range(20))


def test_interleaved_train_dataset_is_deterministic_across_repeated_access():
    dataset = InterleavedTrainDataset(
        _make_tagged_dataset("t", 5), _make_tagged_dataset("i", 20), translation_count=12, instruction_count=20, seed=0
    )
    first_pass = [dataset[i]["tag"] for i in range(len(dataset))]
    second_pass = [dataset[i]["tag"] for i in range(len(dataset))]
    assert first_pass == second_pass


def test_interleaved_train_dataset_raises_index_error_out_of_range():
    dataset = InterleavedTrainDataset(
        _make_tagged_dataset("t", 5), _make_tagged_dataset("i", 5), translation_count=5, instruction_count=5, seed=0
    )
    with pytest.raises(IndexError):
        dataset[10]


def test_interleaved_train_dataset_supports_getitem_based_fallback_iteration():
    # No __iter__ is defined; this exercises Python's __getitem__-based fallback iteration
    # protocol, the same path transformers' LengthGroupedSampler relies on to measure lengths.
    dataset = InterleavedTrainDataset(
        _make_tagged_dataset("t", 3), _make_tagged_dataset("i", 3), translation_count=3, instruction_count=3, seed=0
    )
    tags = [row["tag"] for row in dataset]
    assert len(tags) == 6


def _tags_for_epoch(dataset, epoch):
    dataset.start_epoch(epoch)
    # The sampler permutes the order but covers every index exactly once per epoch.
    return [dataset[i]["tag"] for i in range(len(dataset))]


def _interleaved(translation_count, instruction_count, pool=100):
    return InterleavedTrainDataset(
        _make_tagged_dataset("t", 10), _make_tagged_dataset("i", pool), translation_count, instruction_count, seed=0
    )


def test_each_epoch_is_a_full_pass_over_the_translation_pool():
    dataset = _interleaved(translation_count=10, instruction_count=5)
    for epoch in range(3):
        tags = _tags_for_epoch(dataset, epoch)
        assert sorted(tag for tag in tags if tag.startswith("t")) == sorted(f"t{i}" for i in range(10))


def test_later_epochs_draw_instruction_rows_the_earlier_ones_did_not():
    dataset = _interleaved(translation_count=10, instruction_count=5)
    seen = [{tag for tag in _tags_for_epoch(dataset, epoch) if tag.startswith("i")} for epoch in range(4)]

    assert all(len(epoch_tags) == 5 for epoch_tags in seen)
    # 4 epochs x 5 slots drawn from a pool of 100, so no row should come round twice.
    assert len(set().union(*seen)) == 20


def test_instruction_rows_only_repeat_once_the_pool_is_exhausted():
    dataset = _interleaved(translation_count=10, instruction_count=3, pool=6)
    first_two = set(_tags_for_epoch(dataset, 0)) | set(_tags_for_epoch(dataset, 1))
    assert {tag for tag in first_two if tag.startswith("i")} == {f"i{i}" for i in range(6)}


def test_epoch_callback_passes_the_trainer_epoch_to_the_dataset():
    # Verified against a real Trainer run: state.epoch is exactly the loop epoch here, and it is
    # restored from the checkpoint, so a resumed run carries on rather than replaying.
    recorded = []
    callback = InterleavedEpochCallback(SimpleNamespace(start_epoch=recorded.append))

    for epoch in (0.0, 1.0, 2.6):
        callback.on_epoch_begin(None, SimpleNamespace(epoch=epoch), None)

    assert recorded == [0, 1, 2]


def _mixed(num_devices, max_steps, translation_size=1000, mix_ratio=0.1, num_train_epochs=3.0):
    training_args = SimpleNamespace(
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=num_train_epochs,
    )
    mix = InstructionMix(Path("instruction.jsonl"), _instruction_data({"mix_ratio": mix_ratio}), num_devices, seed=0)
    return mix.interleave(_make_tagged_dataset("t", translation_size), _make_tagged_dataset("i", 10), training_args)


def _length(num_devices, max_steps, translation_size=1000, mix_ratio=0.1, num_train_epochs=3.0):
    return len(_mixed(num_devices, max_steps, translation_size, mix_ratio, num_train_epochs))


def test_a_step_budget_sizes_the_dataset_for_the_whole_run():
    # The Trainer stops at max_steps, so the dataset covers exactly what the run consumes.
    assert _length(num_devices=2, max_steps=100) == 100 * 4 * 8 * 2


def test_an_epoch_budget_sizes_the_dataset_for_one_pass_plus_the_instruction_share():
    # The Trainer derives its step count from this length and then runs num_train_epochs of them,
    # so a whole-run length here would square the epoch count.
    mixed = _mixed(num_devices=1, max_steps=-1)
    assert len(mixed) == 1100
    # And the split hands a full pass back, so instructions never displace translations.
    assert sum(1 for row in mixed if row["tag"].startswith("t")) == 1000


def test_an_unset_step_budget_is_treated_as_an_epoch_budget():
    # Neither the epoch count nor the device count belongs here; the Trainer applies both itself.
    assert _length(num_devices=4, max_steps=0, num_train_epochs=10.0) == 1100


class _StubInferenceModel:
    def eval(self):
        return self

    def to(self, device):
        return self


class _RecordingProvider(CausalLMProvider):
    def __init__(self):
        super().__init__(
            "google/gemma-2-2b-it",
            CausalLMModelSettings({}),
            FinetuneMethod("lora"),
            TrainerSettings({}),
            mixed_precision=False,
        )
        self.last_checkpoint_path = "unset"

    def create_model_for_inference(self, checkpoint_path):
        self.last_checkpoint_path = checkpoint_path
        return _StubInferenceModel()


class _UnloadedTokenizer(CausalLMTokenizer):
    def __init__(self):
        super().__init__("google/gemma-2-2b-it", trust_remote_code=False)

    def load(self):
        return None


def _model_over(provider: _RecordingProvider, model_dir: Path) -> LocalLLMModel:
    generator = CausalLMGenerator(provider, _UnloadedTokenizer(), _prompts(), GenerationSettings({}), ExamplePools([]))
    return LocalLLMModel(CheckpointDirectory(model_dir), 1, None, generator, None, seed=111)


def test_create_inference_model_resolves_checkpoint_when_model_dir_exists(tmp_path):
    model_dir = tmp_path / "run"
    (model_dir / "checkpoint-999").mkdir(parents=True)
    provider = _RecordingProvider()

    _model_over(provider, model_dir).load_for_inference(CheckpointType.LAST)

    assert provider.last_checkpoint_path == model_dir / "checkpoint-999"


def test_create_inference_model_falls_back_to_base_model_with_no_checkpoints(tmp_path):
    # This is what lets the chat script load a bare model name (with no experiment directory,
    # so model_dir can never exist) as the pristine pretrained model.
    provider = _RecordingProvider()

    _model_over(provider, tmp_path / "run").load_for_inference(CheckpointType.LAST)

    assert provider.last_checkpoint_path is None


# --- train.prompt ------------------------------------------------------------------------


def _train_prompt(config, source, example_pool_index=None, rotation_index=None):
    languages = config.create_languages()
    return config.create_training_prompts().build(
        source,
        languages.of("en"),
        languages.of("fr"),
        example_pool_index=example_pool_index,
        rotation_index=rotation_index,
    )


def _infer_prompt(config, source):
    languages = config.create_languages()
    return config.create_inference_prompts().build(source, languages.of("en"), languages.of("fr"))


def _train_instruction(config, source, rotation_index=None):
    return _train_prompt(config, source, rotation_index=rotation_index).instruction


def _infer_instruction(config, source):
    return _infer_prompt(config, source).instruction


def _write_templates(path: Path, entries) -> Path:
    # A template file line has to carry every field, so fill in the ones a case does not vary.
    complete = [{"system_message": "", "example_format": "text", **entry} for entry in entries]
    path.write_text("".join(json.dumps(entry) + "\n" for entry in complete), encoding="utf-8")
    return path


def test_train_prompt_defaults_to_a_fixed_zero_shot_prompt(tmp_path):
    # No corpus is written, so drawing examples would raise; the default asks for none.
    config = _construct_llm_config(tmp_path, {})
    instruction = _train_instruction(config, "hello")
    assert instruction == LocalLLMConfig.DEFAULT_INSTRUCTION_TEMPLATE.format(
        src_lang="en", trg_lang="fr", source="hello"
    )


def test_train_and_infer_prompts_are_configured_independently(tmp_path):
    (tmp_path / "train.src.txt").write_text("".join(f"source {i}\n" for i in range(6)), encoding="utf-8")
    (tmp_path / "train.trg.txt").write_text("".join(f"target {i}\n" for i in range(6)), encoding="utf-8")
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path)
    config = LocalLLMConfig(
        tmp_path,
        {
            "data": {"corpus_pairs": []},
            "model": "google/gemma-2-2b-it",
            "train": {"prompt": {"num_examples": 4}},
            "infer": {"prompt": {"num_examples": 1}},
        },
        environment,
    )
    assert _train_instruction(config, "one").count("Source (") == 4
    assert _infer_instruction(config, "one").count("Source (") == 1


def test_train_prompt_rejects_an_unknown_type(tmp_path):
    with pytest.raises(ValueError, match="Unknown train.prompt.type"):
        _construct_llm_config(tmp_path, {"type": "wobbling"})


def test_train_prompt_rejects_a_template_file_with_a_fixed_prompt(tmp_path):
    with pytest.raises(ValueError, match="only valid with train.prompt.type"):
        _construct_llm_config(tmp_path, {"type": "fixed", "template_file": "templates.jsonl"})


def test_rotating_train_prompt_requires_a_template_file(tmp_path):
    with pytest.raises(ValueError, match="requires train.prompt.template_file"):
        _construct_llm_config(tmp_path, {"type": "rotating"})


def test_rotating_train_prompt_rejects_a_system_message(tmp_path):
    _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {source}"}])
    with pytest.raises(ValueError, match='only valid with train.prompt.type: "fixed"'):
        _construct_llm_config(
            tmp_path, {"type": "rotating", "template_file": "templates.jsonl", "system_message": "anything"}
        )


def test_rotating_train_prompt_rejects_an_instruction_template(tmp_path):
    _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {source}"}])
    with pytest.raises(ValueError, match='only valid with train.prompt.type: "fixed"'):
        _construct_llm_config(
            tmp_path, {"type": "rotating", "template_file": "templates.jsonl", "instruction_template": "anything"}
        )


def test_rotating_train_prompt_rejects_an_example_format(tmp_path):
    _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {source}"}])
    with pytest.raises(ValueError, match='only valid with train.prompt.type: "fixed"'):
        _construct_llm_config(
            tmp_path, {"type": "rotating", "template_file": "templates.jsonl", "example_format": "anything"}
        )


def test_rotating_train_prompt_reads_its_templates_from_the_file(tmp_path):
    _write_templates(
        tmp_path / "templates.jsonl",
        [{"instruction_template": "A: {source}"}, {"instruction_template": "B: {source}"}],
    )
    config = _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl"})
    assert [_train_instruction(config, "x", rotation_index=i) for i in range(2)] == ["A: x", "B: x"]


def test_rotating_train_prompt_rotates_across_training_rows(tmp_path):
    _write_templates(
        tmp_path / "templates.jsonl",
        [{"instruction_template": "A: {source}"}, {"instruction_template": "B: {source}"}],
    )
    config = _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl"})
    instructions = [_train_prompt(config, "hello", example_pool_index=i).instruction for i in range(3)]
    assert instructions == ["A: hello", "B: hello", "A: hello"]


def test_rotating_train_prompt_resolves_the_template_file_under_the_mt_dir(tmp_path):
    mt_dir = tmp_path / "mt"
    (mt_dir).mkdir()
    _write_templates(mt_dir / "shared.jsonl", [{"instruction_template": "shared: {source}"}])
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    config = LocalLLMConfig(
        exp_dir,
        {
            "data": {"corpus_pairs": []},
            "model": "google/gemma-2-2b-it",
            "train": {"prompt": {"type": "rotating", "template_file": "shared.jsonl"}},
        },
        SilNlpEnv.create_environment_with_mt_dir(mt_dir),
    )
    assert _train_instruction(config, "x") == "shared: x"


def test_rotating_train_prompt_warns_when_a_file_template_disagrees_with_num_examples(tmp_path, caplog):
    _write_templates(
        tmp_path / "templates.jsonl",
        [{"instruction_template": "A: {examples}{source}"}, {"instruction_template": "B: {source}"}],
    )
    with caplog.at_level(logging.WARNING):
        _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl", "num_examples": 2})
    assert any("templates.jsonl instruction_template[1]" in record.message for record in caplog.records)


def test_rotating_train_prompt_warns_when_a_file_template_has_a_placeholder_but_examples_are_off(tmp_path, caplog):
    _write_templates(
        tmp_path / "templates.jsonl",
        [{"instruction_template": "A: {source}"}, {"instruction_template": "B: {examples}{source}"}],
    )
    with caplog.at_level(logging.WARNING):
        _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl", "num_examples": 0})
    assert any("always renders as nothing" in record.message for record in caplog.records)


def test_rotating_train_prompt_rejects_translate_gemma_with_examples(tmp_path):
    _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {examples}{source}"}])
    with pytest.raises(RuntimeError, match="TranslateGemma"):
        _construct_llm_config(
            tmp_path,
            {"type": "rotating", "template_file": "templates.jsonl", "num_examples": 2},
            model="google/translategemma-4b-it",
        )


def test_infer_prompt_rejects_translate_gemma_with_examples(tmp_path):
    with pytest.raises(RuntimeError, match="TranslateGemma"):
        _construct_llm_config(tmp_path, {"num_examples": 2}, model="google/translategemma-4b-it", section="infer")


def test_fixed_prompts_can_always_be_rendered_through_the_chat_template(tmp_path):
    # Every local prompt reaches tokenizer.apply_chat_template, so a plain PromptMessages here
    # would fail only once training started.
    config = _construct_llm_config(tmp_path, {"type": "fixed"})
    assert isinstance(_train_prompt(config, "hello"), LocalLLMPromptMessages)
    assert isinstance(_infer_prompt(config, "hello"), LocalLLMPromptMessages)


def test_rotating_prompts_can_always_be_rendered_through_the_chat_template(tmp_path):
    # Every local prompt reaches tokenizer.apply_chat_template, so a plain PromptMessages here
    # would fail only once training started.
    _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {source}"}])
    config = _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl"})
    assert isinstance(_train_prompt(config, "hello"), LocalLLMPromptMessages)
    assert isinstance(_infer_prompt(config, "hello"), LocalLLMPromptMessages)


def test_the_local_model_reads_the_plain_training_corpus(tmp_path):
    # Local fine-tuning always preprocesses with tokenize: false, so the plain files are the
    # detokenized ones; a stale detok file from another model must not be picked up.
    (tmp_path / "train.src.txt").write_text("the cat sat\n", encoding="utf-8")
    (tmp_path / "train.trg.txt").write_text("le chat\n", encoding="utf-8")
    (tmp_path / "train.src.detok.txt").write_text("something else\n", encoding="utf-8")
    (tmp_path / "train.trg.detok.txt").write_text("autre chose\n", encoding="utf-8")

    config = _construct_llm_config(tmp_path, {"num_examples": 1})
    assert "le chat" in _train_instruction(config, "the cat sat too")


def test_rotating_train_prompt_rotates_eval_rows_too(tmp_path):
    _write_templates(
        tmp_path / "templates.jsonl",
        [{"instruction_template": "A: {source}"}, {"instruction_template": "B: {source}"}],
    )
    config = _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl"})
    instructions = [_train_prompt(config, "hello", rotation_index=i).instruction for i in range(3)]
    assert instructions == ["A: hello", "B: hello", "A: hello"]


def test_check_example_corpora_rejects_a_missing_training_corpus(tmp_path):
    # A fine-tune whose examples silently vanished trains on the wrong prompt shape, so this
    # fails before the model is loaded rather than when the first row is encoded.
    config = _construct_llm_config(tmp_path, {"num_examples": 2})
    with pytest.raises(RuntimeError, match="Run preprocessing"):
        config.check_example_corpora()


def test_check_example_corpora_checks_the_infer_prompt_too(tmp_path):
    config = _construct_llm_config(tmp_path, {"num_examples": 2}, section="infer")
    with pytest.raises(RuntimeError, match="Run preprocessing"):
        config.check_example_corpora()


def test_check_example_corpora_passes_without_examples(tmp_path):
    _construct_llm_config(tmp_path, {"num_examples": 0}).check_example_corpora()


def test_check_example_corpora_passes_once_the_corpus_is_there(tmp_path):
    (tmp_path / "train.src.txt").write_text("the cat sat\n", encoding="utf-8")
    (tmp_path / "train.trg.txt").write_text("le chat\n", encoding="utf-8")
    _construct_llm_config(tmp_path, {"num_examples": 1}).check_example_corpora()
