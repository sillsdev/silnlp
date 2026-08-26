import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from jinja2.exceptions import UndefinedError

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config import CheckpointType, Config
from silnlp.nmt.config_utils import is_llm_config
from silnlp.nmt.llm_config import (
    DataCollatorForCausalLM,
    InterleavedTrainDataset,
    Language,
    LLMConfig,
    LLMModel,
    PromptMessages,
    TranslateGemmaPromptMessages,
    build_generation_kwargs,
)


def test_is_llm_config_explicit_model_type():
    assert is_llm_config({"model_type": "llm", "model": "anything"})
    assert not is_llm_config({"model_type": "nmt", "model": "google/gemma-2-2b-it"})


def test_is_llm_config_prefix_fallback():
    assert is_llm_config({"model": "google/gemma-2-2b-it"})
    assert is_llm_config({"model": "tencent/Hunyuan-MT-7B"})
    assert not is_llm_config({"model": "facebook/nllb-200-distilled-1.3B"})
    assert not is_llm_config({"model": "google/madlad400-3b-mt"})


def test_prompt_messages_to_chat_messages():
    prompt = PromptMessages(system_message="You are a translator.", instruction="Translate: hello", target="bonjour")
    assert prompt.to_chat_messages() == [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate: hello"},
        {"role": "assistant", "content": "bonjour"},
    ]


def test_prompt_messages_folds_system_message_into_user_turn():
    prompt = PromptMessages(system_message="You are a translator.", instruction="Translate: hello")
    assert prompt.to_folded_chat_messages() == [
        {"role": "user", "content": "You are a translator.\n\nTranslate: hello"}
    ]


def test_prompt_messages_without_system_message():
    prompt = PromptMessages(system_message="", instruction="Translate: hello")
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


class _FakeExamplePromptBuilder:
    """Verifies delegation, not retrieval (see test_example_retrieval.py for that)."""

    def __init__(self, text: str = ""):
        self._text = text
        self.calls = []

    def render(self, source, src_lang_name, trg_lang_name, pool_index=None):
        self.calls.append((source, src_lang_name, trg_lang_name, pool_index))
        return self._text


@dataclass
class _StubLLMConfig:
    model: str
    params: dict
    data: dict
    _example_prompt_builder: object = field(default_factory=_FakeExamplePromptBuilder)

    lang_name = LLMConfig.lang_name
    language = LLMConfig.language
    build_prompt_messages = LLMConfig.build_prompt_messages


def test_language_resolves_configured_name_and_falls_back_to_iso():
    config = _StubLLMConfig(model="google/gemma-2-2b-it", params={}, data={"lang_codes": {"en": "English"}})
    assert config.language("en") == Language("en", "English")
    assert config.language("fr") == Language("fr", "fr")


def test_build_prompt_messages_translate_gemma_uses_structured_content():
    config = _StubLLMConfig(
        model="google/translategemma-4b-it",
        params={"prompt": {"instruction_template": "Translate from {src_lang} to {trg_lang}.\n\n{source}"}},
        data={"lang_codes": {}},
    )
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
    config = _StubLLMConfig(
        model="google/gemma-2-2b-it",
        params={
            "prompt": {
                "instruction_template": "Translate from {src_lang} to {trg_lang}.\n\n{source}",
                "system_message": "",
            }
        },
        data={"lang_codes": {"en": "English", "fr": "French"}},
    )
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"))
    assert prompt == PromptMessages(
        system_message="", instruction="Translate from English to French.\n\nhello", target=None
    )


def test_build_prompt_messages_zero_examples_matches_legacy_output_exactly():
    # Pins the exact zero-shot text, since existing fine-tuned checkpoints depend on it.
    config = _StubLLMConfig(
        model="google/gemma-2-2b-it",
        params={
            "prompt": {
                "system_message": "",
                "instruction_template": "Translate the following text from {src_lang} to {trg_lang}.\n\n{examples}{source}",
            }
        },
        data={"lang_codes": {"en": "English", "fr": "French"}},
    )
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"))
    assert prompt.instruction == "Translate the following text from English to French.\n\nhello"


def test_build_prompt_messages_splices_example_prompt_builder_output_before_source():
    builder = _FakeExamplePromptBuilder(text="Source: cat\nTarget: chat\n\nSource: dog\nTarget: chien\n\n")
    config = _StubLLMConfig(
        model="google/gemma-2-2b-it",
        params={
            "prompt": {
                "system_message": "",
                "instruction_template": "Translate from {src_lang} to {trg_lang}.\n\n{examples}{source}",
            }
        },
        data={"lang_codes": {"en": "English", "fr": "French"}},
        _example_prompt_builder=builder,
    )
    prompt = config.build_prompt_messages("hello", config.language("en"), config.language("fr"))
    assert prompt.instruction == (
        "Translate from English to French.\n\n" "Source: cat\nTarget: chat\n\n" "Source: dog\nTarget: chien\n\n" "hello"
    )
    assert builder.calls == [("hello", "English", "French", None)]


def test_build_prompt_messages_passes_pool_index_through_to_example_prompt_builder():
    builder = _FakeExamplePromptBuilder()
    config = _StubLLMConfig(
        model="google/gemma-2-2b-it",
        params={"prompt": {"system_message": "", "instruction_template": "{examples}{source}"}},
        data={"lang_codes": {}},
        _example_prompt_builder=builder,
    )
    config.build_prompt_messages("hello", config.language("en"), config.language("fr"), example_pool_index=3)
    assert builder.calls == [("hello", "en", "fr", 3)]


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
    config = _StubLLMConfig(
        model="google/translategemma-4b-it",
        params={"prompt": {"instruction_template": "unused"}},
        data={"lang_codes": {"en": "English", "tst": "Test Language"}},
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


def _construct_llm_config(tmp_path: Path, prompt_overrides: dict, model: str = "google/gemma-2-2b-it") -> LLMConfig:
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path)
    return LLMConfig(
        tmp_path,
        {"data": {"corpus_pairs": []}, "model": model, "params": {"prompt": prompt_overrides}},
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
    with pytest.raises(ValueError, match="Unknown params.prompt.example_selection.method"):
        _construct_llm_config(
            tmp_path,
            {
                "num_examples": 2,
                "instruction_template": "Translate {src_lang} to {trg_lang}.\n\n{examples}{source}",
                "example_selection": {"method": "bogus"},
            },
        )


def test_build_adapter_config_plain_lora():
    peft_config = LLMModel._build_adapter_config(
        {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"}, use_dora=False
    )
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.modules_to_save is None
    assert peft_config.use_dora is False


def test_build_adapter_config_passes_through_modules_to_save():
    peft_config = LLMModel._build_adapter_config(
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
    peft_config = LLMModel._build_adapter_config(adapter, use_dora=True)
    assert peft_config.use_dora is True


@dataclass
class _MethodStub:
    params: dict

    finetune_method = LLMConfig.finetune_method
    uses_quantization = LLMConfig.uses_quantization
    uses_dora = LLMConfig.uses_dora


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
    LLMConfig._normalize_deprecated_keys(config)
    assert "lora" not in config["params"]
    assert config["params"]["adapter"] == {"rank": 8}


def test_normalize_deprecated_keys_prefers_explicit_adapter():
    config = {"params": {"lora": {"rank": 8}, "adapter": {"rank": 64}}}
    LLMConfig._normalize_deprecated_keys(config)
    # An explicit adapter wins; the deprecated lora key is left untouched rather than clobbering it.
    assert config["params"]["adapter"] == {"rank": 64}


@dataclass
class _InstructionDataStub:
    params: dict
    _environment: SilNlpEnv
    exp_dir: Path = Path(".")

    instruction_datasets = LLMConfig.instruction_datasets
    instruction_data_size = LLMConfig.instruction_data_size
    instruction_mix_ratio = LLMConfig.instruction_mix_ratio
    instruction_data_paths = LLMConfig.instruction_data_paths
    instruction_src_filename = Config.instruction_src_filename
    instruction_trg_filename = Config.instruction_trg_filename
    _open_append = Config._open_append
    _write_instruction_data = LLMConfig._write_instruction_data


def test_instruction_datasets_defaults_to_empty():
    stub = _InstructionDataStub(params={"instruction_data": {"datasets": [], "size": 100000}}, _environment=None)
    assert stub.instruction_datasets == []
    assert stub.instruction_data_size == 100000
    assert stub.instruction_data_paths() == []
    assert stub._write_instruction_data() == 0


def test_instruction_data_paths_resolved_under_mt_dir_instructions(tmp_path):
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path)
    stub = _InstructionDataStub(
        params={"instruction_data": {"datasets": ["dolly", "no_robots"], "size": 100000}},
        _environment=environment,
    )
    assert stub.instruction_data_paths() == [
        (tmp_path / "instructions" / "dolly.input.txt", tmp_path / "instructions" / "dolly.output.txt"),
        (tmp_path / "instructions" / "no_robots.input.txt", tmp_path / "instructions" / "no_robots.output.txt"),
    ]


def test_instruction_data_size_rejects_negative():
    stub = _InstructionDataStub(params={"instruction_data": {"datasets": [], "size": -1}}, _environment=None)
    with pytest.raises(ValueError, match="non-negative"):
        stub.instruction_data_size


def test_instruction_mix_ratio_default():
    stub = _InstructionDataStub(params={"instruction_data": {"mix_ratio": 0.1}}, _environment=None)
    assert stub.instruction_mix_ratio == 0.1


def test_instruction_mix_ratio_rejects_negative():
    stub = _InstructionDataStub(params={"instruction_data": {"mix_ratio": -0.1}}, _environment=None)
    with pytest.raises(ValueError, match="non-negative"):
        stub.instruction_mix_ratio


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

    _estimate_total_train_examples = LLMModel._estimate_total_train_examples


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


def _write_lines(path: Path, lines: list) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_write_instruction_data_mixes_evenly_and_uses_undersized_datasets_whole(tmp_path):
    mt_dir = tmp_path / "mt"
    instructions_dir = mt_dir / "instructions"
    instructions_dir.mkdir(parents=True)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    _write_lines(instructions_dir / "a.input.txt", [f"a-in-{i}" for i in range(10)])
    _write_lines(instructions_dir / "a.output.txt", [f"a-out-{i}" for i in range(10)])
    # "b" has fewer lines than its even share of the requested size (4), so all of it is used.
    _write_lines(instructions_dir / "b.input.txt", [f"b-in-{i}" for i in range(3)])
    _write_lines(instructions_dir / "b.output.txt", [f"b-out-{i}" for i in range(3)])

    stub = _InstructionDataStub(
        params={"instruction_data": {"datasets": ["a", "b"], "size": 8}},
        _environment=SilNlpEnv.create_environment_with_mt_dir(mt_dir),
        exp_dir=exp_dir,
    )

    count = stub._write_instruction_data()
    assert count == 4 + 3

    src_lines = (exp_dir / "instruction.src.txt").read_text(encoding="utf-8").splitlines()
    trg_lines = (exp_dir / "instruction.trg.txt").read_text(encoding="utf-8").splitlines()
    assert len(src_lines) == len(trg_lines) == count
    # inputs and outputs stay aligned line-for-line
    for src_line, trg_line in zip(src_lines, trg_lines):
        assert src_line.split("-in-")[0] == trg_line.split("-out-")[0]

    a_selected = {line for line in src_lines if line.startswith("a-in-")}
    b_selected = {line for line in src_lines if line.startswith("b-in-")}
    assert len(a_selected) == 4
    assert a_selected.issubset({f"a-in-{i}" for i in range(10)})
    assert b_selected == {f"b-in-{i}" for i in range(3)}


def test_write_instruction_data_missing_file_raises(tmp_path):
    mt_dir = tmp_path / "mt"
    (mt_dir / "instructions").mkdir(parents=True)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    stub = _InstructionDataStub(
        params={"instruction_data": {"datasets": ["missing"], "size": 10}},
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

    _create_inference_model = LLMModel._create_inference_model

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
