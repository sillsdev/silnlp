from dataclasses import dataclass

import pytest

from silnlp.nmt.config_utils import is_llm_config
from silnlp.nmt.llm_config import DataCollatorForCausalLM, PromptMessages, build_generation_kwargs


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
