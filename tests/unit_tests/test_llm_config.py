from dataclasses import dataclass

import pytest
from jinja2.exceptions import UndefinedError

from silnlp.nmt.config_utils import is_llm_config
from silnlp.nmt.llm_config import (
    DataCollatorForCausalLM,
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


@dataclass
class _StubLLMConfig:
    model: str
    params: dict
    data: dict

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
