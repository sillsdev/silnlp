from dataclasses import dataclass

from jinja2.exceptions import UndefinedError

from silnlp.nmt.config_utils import is_llm_config
from silnlp.nmt.llm_config import DataCollatorForCausalLM, LLMConfig


def test_is_llm_config_explicit_model_type():
    assert is_llm_config({"model_type": "llm", "model": "anything"})
    assert not is_llm_config({"model_type": "nmt", "model": "google/gemma-2-2b-it"})


def test_is_llm_config_prefix_fallback():
    assert is_llm_config({"model": "google/gemma-2-2b-it"})
    assert is_llm_config({"model": "tencent/Hunyuan-MT-7B"})
    assert not is_llm_config({"model": "facebook/nllb-200-distilled-1.3B"})
    assert not is_llm_config({"model": "google/madlad400-3b-mt"})


def test_fold_system_message_merges_into_user_turn():
    messages = [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate: hello"},
    ]
    folded = LLMConfig._fold_system_message(messages)
    assert folded is not None
    assert len(folded) == 1
    assert folded[0]["role"] == "user"
    assert folded[0]["content"] == "You are a translator.\n\nTranslate: hello"
    # The original messages must not be mutated.
    assert messages[0]["role"] == "system"


def test_fold_system_message_no_system_returns_none():
    messages = [{"role": "user", "content": "Translate: hello"}]
    assert LLMConfig._fold_system_message(messages) is None


@dataclass
class _StubLLMConfig:
    model: str
    params: dict
    data: dict

    lang_name = LLMConfig.lang_name
    build_prompt_messages = LLMConfig.build_prompt_messages
    apply_prompt_template = LLMConfig.apply_prompt_template
    _render_translate_gemma_prompt = LLMConfig._render_translate_gemma_prompt


def test_build_prompt_messages_translate_gemma_uses_structured_content():
    config = _StubLLMConfig(
        model="google/translategemma-4b-it",
        params={"prompt": {"instruction_template": "Translate from {src_lang} to {trg_lang}.\n\n{source}"}},
        data={"lang_codes": {}},
    )
    messages = config.build_prompt_messages("hello", "en", "fr", target="bonjour")
    assert messages == [
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
    messages = config.build_prompt_messages("hello", "en", "fr")
    assert messages == [{"role": "user", "content": "Translate from English to French.\n\nhello"}]


class _StubTranslateGemmaTokenizer:
    chat_template = "{# a real chat template would render this #}"
    bos_token = "<bos>"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        # Mimics the real template's behavior for a language code outside its fixed lookup table.
        raise UndefinedError("'dict object' has no attribute 'ded'")

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


def test_apply_prompt_template_translate_gemma_falls_back_for_unrecognized_language_code():
    config = _StubLLMConfig(
        model="google/translategemma-4b-it",
        params={"prompt": {"instruction_template": "unused"}},
        data={"lang_codes": {"en": "English"}},
    )
    tokenizer = _StubTranslateGemmaTokenizer()
    messages = config.build_prompt_messages("hello", "en", "ded")

    text = config.apply_prompt_template(tokenizer, messages, add_generation_prompt=True, tokenize=False)
    assert text == (
        "<bos><start_of_turn>user\n"
        "You are a professional English (en) to ded (ded) translator. Your goal is to accurately convey "
        "the meaning and nuances of the original English text while adhering to ded grammar, vocabulary, "
        "and cultural sensitivities.\n"
        "Produce only the ded translation, without any additional explanations or commentary. Please "
        "translate the following English text into ded:\n\n\nhello<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    token_ids = config.apply_prompt_template(tokenizer, messages, add_generation_prompt=True, tokenize=True)
    assert token_ids == [ord(c) for c in text]


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


def test_data_collator_pad_to_multiple_of():
    collator = DataCollatorForCausalLM(_StubTokenizer(pad_token_id=0), pad_to_multiple_of=4)
    features = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "attention_mask": [1, 1, 1]}]
    batch = collator(features)
    assert batch["input_ids"].shape[1] == 4
    assert batch["labels"].tolist() == [[-100, 6, 7, -100]]
