import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from jinja2.exceptions import UndefinedError

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config import CheckpointType, Config
from silnlp.nmt.config import Language
from silnlp.nmt.config_utils import is_local_llm_config
from silnlp.nmt.example_retrieval import Example, create_example_formatter
from silnlp.nmt.llm_config import PromptMessages, PromptTemplate
from silnlp.nmt.local_llm_config import (
    ChatPromptBuilder,
    ChatPromptMessages,
    DataCollatorForCausalLM,
    InterleavedTrainDataset,
    LocalLLMConfig,
    LocalLLMModel,
    TranslateGemmaPromptMessages,
    _render_turns,
    build_generation_kwargs,
)


def test_is_local_llm_config_explicit_model_type():
    assert is_local_llm_config({"model_type": "local_llm", "model": "anything"})
    assert is_local_llm_config({"model_type": "llm", "model": "anything"})
    assert not is_local_llm_config({"model_type": "nmt", "model": "google/gemma-2-2b-it"})
    assert not is_local_llm_config({"model_type": "remote_llm", "model": "anything"})


def test_is_local_llm_config_prefix_fallback():
    assert is_local_llm_config({"model": "google/gemma-2-2b-it"})
    assert is_local_llm_config({"model": "tencent/Hunyuan-MT-7B"})
    assert not is_local_llm_config({"model": "facebook/nllb-200-distilled-1.3B"})
    assert not is_local_llm_config({"model": "google/madlad400-3b-mt"})


def test_prompt_messages_to_chat_messages():
    prompt = ChatPromptMessages(
        system_message="You are a translator.", instruction="Translate: hello", target="bonjour"
    )
    assert prompt.to_chat_messages() == [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate: hello"},
        {"role": "assistant", "content": "bonjour"},
    ]


def test_prompt_messages_folds_system_message_into_user_turn():
    prompt = ChatPromptMessages(system_message="You are a translator.", instruction="Translate: hello")
    assert prompt.to_folded_chat_messages() == [
        {"role": "user", "content": "You are a translator.\n\nTranslate: hello"}
    ]


def test_prompt_messages_without_system_message():
    prompt = ChatPromptMessages(system_message="", instruction="Translate: hello")
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


def test_build_generation_kwargs_beam_search():
    infer = {"max_new_tokens": 256, "num_beams": 4, "do_sample": False, "temperature": 0.7}
    gen_kwargs = build_generation_kwargs(infer, num_return_sequences=2, pad_token_id=0)
    assert gen_kwargs["num_beams"] == 4
    assert gen_kwargs["num_return_sequences"] == 2
    assert "do_sample" not in gen_kwargs
    assert "temperature" not in gen_kwargs


def test_build_generation_kwargs_sampling_does_not_set_num_beams():
    infer = {"max_new_tokens": 256, "num_beams": 4, "do_sample": True, "temperature": 0.7}
    gen_kwargs = build_generation_kwargs(infer, num_return_sequences=3, pad_token_id=0)
    assert gen_kwargs["do_sample"] is True
    assert gen_kwargs["temperature"] == 0.7
    assert gen_kwargs["num_return_sequences"] == 3
    assert "num_beams" not in gen_kwargs


def test_build_generation_kwargs_rejects_more_drafts_than_beams():
    infer = {"max_new_tokens": 256, "num_beams": 1, "do_sample": False, "temperature": 0.7}
    with pytest.raises(RuntimeError, match="num_beams"):
        build_generation_kwargs(infer, num_return_sequences=2, pad_token_id=0)


def test_data_collator_pad_to_multiple_of():
    collator = DataCollatorForCausalLM(_StubTokenizer(pad_token_id=0), pad_to_multiple_of=4)
    features = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "attention_mask": [1, 1, 1]}]
    batch = collator(features)
    assert batch["input_ids"].shape[1] == 4
    assert batch["labels"].tolist() == [[-100, 6, 7, -100]]


@dataclass
class _StubLocalLLMConfig:
    model: str
    data: dict
    _train_prompt_builder: object = None
    _infer_prompt_builder: object = None

    lang_name = LocalLLMConfig.lang_name
    language = LocalLLMConfig.language
    build_prompt_messages = LocalLLMConfig.build_prompt_messages


def _stub_config(model="google/gemma-2-2b-it", lang_codes=None, **prompt_overrides):
    builder = _prompt_builder(**prompt_overrides)
    return _StubLocalLLMConfig(
        model=model,
        data={"lang_codes": lang_codes if lang_codes is not None else {}},
        _train_prompt_builder=builder,
        _infer_prompt_builder=builder,
    )


def _prompt_builder(
    instruction_template="Translate from {src_lang} to {trg_lang}.\n\n{source}",
    system_message="",
    num_examples=0,
    pool=None,
    example_format="text",
):
    template = PromptTemplate(system_message, instruction_template, create_example_formatter(example_format))
    return ChatPromptBuilder([template], num_examples, pool)


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


def test_language_resolves_configured_name_and_falls_back_to_iso():
    config = _stub_config(lang_codes={"en": "English"})
    assert config.language("en") == Language("en", "English")
    assert config.language("fr") == Language("fr", "fr")


def test_build_prompt_messages_translate_gemma_uses_structured_content():
    config = _stub_config(model="google/translategemma-4b-it")
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"), target="bonjour")
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
    config = _stub_config(lang_codes={"en": "English", "fr": "French"})
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"))
    assert prompt == ChatPromptMessages(
        system_message="", instruction="Translate from English to French.\n\nhello", target=None
    )


def test_build_prompt_messages_zero_examples_matches_legacy_output_exactly():
    # Pins the exact zero-shot text, since existing fine-tuned checkpoints depend on it.
    config = _stub_config(
        lang_codes={"en": "English", "fr": "French"},
        instruction_template=LocalLLMConfig.DEFAULT_INSTRUCTION_TEMPLATE,
    )
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"))
    assert prompt.instruction == "Translate the following text from English to French.\n\nhello"


def test_build_prompt_messages_splices_rendered_examples_before_source():
    pool = _FakePool([Example("cat", "chat"), Example("dog", "chien")])
    config = _stub_config(
        lang_codes={"en": "English", "fr": "French"},
        instruction_template="Translate from {src_lang} to {trg_lang}.\n\n{examples}{source}",
        num_examples=2,
        pool=pool,
        example_format={"type": "text", "template": "Source: {source}\nTarget: {target}\n\n"},
    )
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"))
    assert prompt.instruction == (
        "Translate from English to French.\n\n" "Source: cat\nTarget: chat\n\n" "Source: dog\nTarget: chien\n\n" "hello"
    )
    assert pool.calls == [("hello", 2, None)]


def test_build_prompt_messages_passes_pool_index_through_for_training():
    pool = _FakePool()
    config = _stub_config(instruction_template="{examples}{source}", num_examples=2, pool=pool)
    config.build_prompt_messages(
        "hello", config.language("en"), config.language("fr"), example_pool_index=3, training=True
    )
    assert pool.calls == [("hello", 2, 3)]


def test_build_prompt_messages_uses_the_infer_builder_outside_training():
    train_pool, infer_pool = _FakePool(), _FakePool()
    config = _StubLocalLLMConfig(
        model="google/gemma-2-2b-it",
        data={"lang_codes": {}},
        _train_prompt_builder=_prompt_builder("{examples}{source}", num_examples=1, pool=train_pool),
        _infer_prompt_builder=_prompt_builder("{examples}{source}", num_examples=5, pool=infer_pool),
    )
    config.build_prompt_messages("hello", config.language("en"), config.language("fr"))
    assert train_pool.calls == []
    assert infer_pool.calls == [("hello", 5, None)]


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
    config = _stub_config(
        model="google/translategemma-4b-it", lang_codes={"en": "English", "tst": "Test Language"}
    )
    tokenizer = _StubTranslateGemmaTokenizer()
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("tst"))

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
    peft_config = LocalLLMModel._build_adapter_config(
        {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"}, use_dora=False
    )
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.modules_to_save is None
    assert peft_config.use_dora is False


def test_build_adapter_config_passes_through_modules_to_save():
    peft_config = LocalLLMModel._build_adapter_config(
        {
            "rank": 64,
            "alpha": 256,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "modules_to_save": ["embed_tokens", "lm_head"],
        },
        use_dora=False,
    )
    assert peft_config.r == 64
    assert peft_config.lora_alpha == 256
    assert peft_config.modules_to_save == ["embed_tokens", "lm_head"]


def test_build_adapter_config_dora():
    adapter = {"rank": 64, "alpha": 256, "dropout": 0.05, "target_modules": "all-linear"}
    peft_config = LocalLLMModel._build_adapter_config(adapter, use_dora=True)
    assert peft_config.use_dora is True


@dataclass
class _MethodStub:
    params: dict

    finetune_method = LocalLLMConfig.finetune_method
    uses_quantization = LocalLLMConfig.uses_quantization
    uses_dora = LocalLLMConfig.uses_dora


def test_finetune_method_axes():
    # (method, quantized, dora)
    cases = [
        ("full", False, False),
        ("lora", False, False),
        ("qlora", True, False),
        ("dora", False, True),
        ("qdora", True, True),
    ]
    for method, quantized, dora in cases:
        stub = _MethodStub(params={"finetune_method": method})
        assert stub.finetune_method == method
        assert stub.uses_quantization is quantized
        assert stub.uses_dora is dora


def test_finetune_method_is_case_insensitive():
    assert _MethodStub(params={"finetune_method": "QDoRA"}).uses_dora is True


def test_finetune_method_invalid_raises():
    with pytest.raises(ValueError, match="Unknown finetune_method"):
        _ = _MethodStub(params={"finetune_method": "bogus"}).finetune_method


def test_normalize_deprecated_keys_renames_lora_to_adapter():
    config = {"params": {"finetune_method": "lora", "lora": {"rank": 8}}}
    LocalLLMConfig._normalize_deprecated_keys(config)
    assert "lora" not in config["params"]
    assert config["params"]["adapter"] == {"rank": 8}


def test_normalize_deprecated_keys_prefers_explicit_adapter():
    config = {"params": {"lora": {"rank": 8}, "adapter": {"rank": 64}}}
    LocalLLMConfig._normalize_deprecated_keys(config)
    # An explicit adapter wins; the deprecated lora key is left untouched rather than clobbering it.
    assert config["params"]["adapter"] == {"rank": 64}


@dataclass
class _InstructionDataStub:
    train: dict
    _environment: SilNlpEnv
    exp_dir: Path = Path(".")

    instruction_datasets = LocalLLMConfig.instruction_datasets
    instruction_data_size = LocalLLMConfig.instruction_data_size
    instruction_mix_ratio = LocalLLMConfig.instruction_mix_ratio
    instruction_data_paths = LocalLLMConfig.instruction_data_paths
    instruction_jsonl_filename = Config.instruction_jsonl_filename
    _open_append = Config._open_append
    _write_instruction_data = LocalLLMConfig._write_instruction_data


def test_instruction_datasets_defaults_to_empty():
    stub = _InstructionDataStub(train={"instruction_data": {"datasets": [], "size": 100000}}, _environment=None)
    assert stub.instruction_datasets == []
    assert stub.instruction_data_size == 100000
    assert stub.instruction_data_paths() == []
    assert stub._write_instruction_data() == 0


def test_instruction_data_paths_resolved_under_mt_dir_instructions(tmp_path):
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path)
    stub = _InstructionDataStub(
        train={"instruction_data": {"datasets": ["dolly", "no_robots"], "size": 100000}},
        _environment=environment,
    )
    assert stub.instruction_data_paths() == [
        tmp_path / "instructions" / "dolly.jsonl",
        tmp_path / "instructions" / "no_robots.jsonl",
    ]


def test_instruction_data_size_rejects_negative():
    stub = _InstructionDataStub(train={"instruction_data": {"datasets": [], "size": -1}}, _environment=None)
    with pytest.raises(ValueError, match="non-negative"):
        stub.instruction_data_size


def test_instruction_mix_ratio_default():
    stub = _InstructionDataStub(train={"instruction_data": {"mix_ratio": 0.1}}, _environment=None)
    assert stub.instruction_mix_ratio == 0.1


def test_instruction_mix_ratio_rejects_negative():
    stub = _InstructionDataStub(train={"instruction_data": {"mix_ratio": -0.1}}, _environment=None)
    with pytest.raises(ValueError, match="non-negative"):
        stub.instruction_mix_ratio


class _StubRejectingSystemTokenizer:
    chat_template = "some template"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
        assert tokenize is True
        assert return_dict is False
        if any(m["role"] == "system" for m in messages):
            raise ValueError("this template does not support a separate system role")
        return [len(m["content"]) for m in messages]


def test_render_turns_happy_path():
    turns = [{"role": "user", "content": "hi"}]
    assert _render_turns(_StubRejectingSystemTokenizer(), turns, add_generation_prompt=True) == [2]


def test_render_turns_folds_leading_system_turn_into_first_user_turn_on_failure():
    turns = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    result = _render_turns(_StubRejectingSystemTokenizer(), turns, add_generation_prompt=True)
    # folded: [{"role": "user", "content": "sys\n\nhi"}, {"role": "assistant", "content": "yo"}]
    assert result == [len("sys\n\nhi"), len("yo")]


class _AlwaysFailingTokenizer:
    chat_template = "some template"

    def apply_chat_template(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_render_turns_reraises_when_there_is_no_system_turn_to_fold():
    with pytest.raises(RuntimeError, match="boom"):
        _render_turns(_AlwaysFailingTokenizer(), [{"role": "user", "content": "hi"}], add_generation_prompt=True)


def test_render_turns_reraises_when_system_turn_has_nothing_to_fold_into():
    with pytest.raises(RuntimeError, match="boom"):
        _render_turns(
            _AlwaysFailingTokenizer(), [{"role": "system", "content": "sys"}], add_generation_prompt=True
        )


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


@dataclass
class _TotalTrainExamplesStub:
    _num_devices: int

    _estimate_total_train_examples = LocalLLMModel._estimate_total_train_examples


def test_estimate_total_train_examples_uses_max_steps_when_set():
    stub = _TotalTrainExamplesStub(_num_devices=2)
    training_args = SimpleNamespace(
        max_steps=100, per_device_train_batch_size=4, gradient_accumulation_steps=8, num_train_epochs=3.0
    )
    assert stub._estimate_total_train_examples(training_args, translation_size=1000) == 100 * 4 * 8 * 2


def test_estimate_total_train_examples_falls_back_to_num_train_epochs_when_max_steps_unset():
    stub = _TotalTrainExamplesStub(_num_devices=1)
    training_args = SimpleNamespace(
        max_steps=-1, per_device_train_batch_size=4, gradient_accumulation_steps=8, num_train_epochs=3.0
    )
    assert stub._estimate_total_train_examples(training_args, translation_size=1000) == 3000


def _write_jsonl_fixture(path: Path, examples: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for turns, output in examples:
            f.write(json.dumps({"turns": turns, "output": output}) + "\n")


def test_write_instruction_data_mixes_evenly_and_uses_undersized_datasets_whole(tmp_path):
    mt_dir = tmp_path / "mt"
    instructions_dir = mt_dir / "instructions"
    instructions_dir.mkdir(parents=True)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    _write_jsonl_fixture(
        instructions_dir / "a.jsonl",
        [([{"role": "user", "content": f"a-in-{i}"}], f"a-out-{i}") for i in range(10)],
    )
    # "b" has fewer lines than its even share of the requested size (4), so all of it is used.
    _write_jsonl_fixture(
        instructions_dir / "b.jsonl",
        [([{"role": "user", "content": f"b-in-{i}"}], f"b-out-{i}") for i in range(3)],
    )

    stub = _InstructionDataStub(
        train={"instruction_data": {"datasets": ["a", "b"], "size": 8}},
        _environment=SilNlpEnv.create_environment_with_mt_dir(mt_dir),
        exp_dir=exp_dir,
    )

    count = stub._write_instruction_data()
    assert count == 4 + 3

    lines = (exp_dir / "instruction.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == count
    examples = [json.loads(line) for line in lines]

    a_selected = {e["output"] for e in examples if e["output"].startswith("a-out-")}
    b_selected = {e["output"] for e in examples if e["output"].startswith("b-out-")}
    assert len(a_selected) == 4
    assert a_selected.issubset({f"a-out-{i}" for i in range(10)})
    assert b_selected == {f"b-out-{i}" for i in range(3)}

    # inputs and outputs stay aligned within each example
    for e in examples:
        assert e["turns"][0]["content"].split("-in-")[1] == e["output"].split("-out-")[1]


def test_write_instruction_data_missing_file_raises(tmp_path):
    mt_dir = tmp_path / "mt"
    (mt_dir / "instructions").mkdir(parents=True)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    stub = _InstructionDataStub(
        train={"instruction_data": {"datasets": ["missing"], "size": 10}},
        _environment=SilNlpEnv.create_environment_with_mt_dir(mt_dir),
        exp_dir=exp_dir,
    )
    with pytest.raises(RuntimeError, match="does not exist"):
        stub._write_instruction_data()


class _StubInferenceModel:
    def eval(self):
        return self

    def to(self, device):
        return self


class _StubInferenceProvider:
    def __init__(self):
        self.last_checkpoint_path = "unset"

    def create_model_for_inference(self, checkpoint_path):
        self.last_checkpoint_path = checkpoint_path
        return _StubInferenceModel()


@dataclass
class _InferenceModelStub:
    _config: SimpleNamespace
    _provider: _StubInferenceProvider

    _create_inference_model = LocalLLMModel._create_inference_model

    def get_checkpoint_path(self, ckpt):
        return (Path("/fake/checkpoint-999"), 999)


def test_create_inference_model_resolves_checkpoint_when_model_dir_exists(tmp_path):
    model_dir = tmp_path / "run"
    model_dir.mkdir()
    provider = _StubInferenceProvider()
    stub = _InferenceModelStub(_config=SimpleNamespace(model_dir=model_dir), _provider=provider)

    stub._create_inference_model(CheckpointType.LAST)

    assert provider.last_checkpoint_path == Path("/fake/checkpoint-999")


def test_create_inference_model_falls_back_to_base_model_with_no_checkpoints(tmp_path):
    # This is what lets the chat script load a bare model name (with no experiment directory,
    # so model_dir can never exist) as the pristine pretrained model.
    provider = _StubInferenceProvider()
    stub = _InferenceModelStub(_config=SimpleNamespace(model_dir=tmp_path / "run"), _provider=provider)

    stub._create_inference_model(CheckpointType.LAST)

    assert provider.last_checkpoint_path is None


# --- train.prompt ------------------------------------------------------------------------


def _write_templates(path: Path, entries) -> Path:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def test_train_prompt_defaults_to_a_fixed_zero_shot_prompt(tmp_path):
    config = _construct_llm_config(tmp_path, {})
    builder = config.train_prompt_builder
    assert builder.num_examples == 0
    assert builder.templates[0].instruction_template == LocalLLMConfig.DEFAULT_INSTRUCTION_TEMPLATE
    assert builder.pool is None


def test_train_and_infer_prompts_are_configured_independently(tmp_path):
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
    assert config.train_prompt_builder.num_examples == 4
    assert config.infer_prompt_builder.num_examples == 1


def test_train_prompt_rejects_an_unknown_type(tmp_path):
    with pytest.raises(ValueError, match="Unknown train.prompt.type"):
        _construct_llm_config(tmp_path, {"type": "wobbling"})


def test_train_prompt_rejects_a_template_file_with_a_fixed_prompt(tmp_path):
    with pytest.raises(ValueError, match="only valid with train.prompt.type"):
        _construct_llm_config(tmp_path, {"type": "fixed", "template_file": "templates.jsonl"})


def test_rotating_train_prompt_requires_a_template_file(tmp_path):
    with pytest.raises(ValueError, match="requires train.prompt.template_file"):
        _construct_llm_config(tmp_path, {"type": "rotating"})


@pytest.mark.parametrize("key", ["system_message", "instruction_template", "example_format"])
def test_rotating_train_prompt_rejects_the_fixed_only_keys(tmp_path, key):
    _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {source}"}])
    with pytest.raises(ValueError, match='only valid with train.prompt.type: "fixed"'):
        _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl", key: "anything"})


def test_rotating_train_prompt_reads_its_templates_from_the_file(tmp_path):
    _write_templates(
        tmp_path / "templates.jsonl",
        [{"instruction_template": "A: {source}"}, {"instruction_template": "B: {source}"}],
    )
    config = _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl"})
    assert [t.instruction_template for t in config.train_prompt_builder.templates] == ["A: {source}", "B: {source}"]


def test_rotating_train_prompt_rotates_across_training_rows(tmp_path):
    _write_templates(
        tmp_path / "templates.jsonl",
        [{"instruction_template": "A: {source}"}, {"instruction_template": "B: {source}"}],
    )
    config = _construct_llm_config(tmp_path, {"type": "rotating", "template_file": "templates.jsonl"})
    instructions = [
        config.build_prompt_messages(
            "hello", config.language("en"), config.language("fr"), example_pool_index=i, training=True
        ).instruction
        for i in range(3)
    ]
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
    assert config.train_prompt_builder.templates[0].instruction_template == "shared: {source}"


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
        _construct_llm_config(
            tmp_path, {"num_examples": 2}, model="google/translategemma-4b-it", section="infer"
        )


@pytest.mark.parametrize("prompt_type", ["fixed", "rotating"])
def test_prompts_can_always_be_rendered_through_the_chat_template(tmp_path, prompt_type):
    # Every local prompt reaches tokenizer.apply_chat_template, so a plain PromptMessages here
    # would fail only once training started.
    overrides = {"type": prompt_type}
    if prompt_type == "rotating":
        _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {source}"}])
        overrides["template_file"] = "templates.jsonl"
    config = _construct_llm_config(tmp_path, overrides)
    for training in (True, False):
        prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"), training=training)
        assert isinstance(prompt, ChatPromptMessages)
