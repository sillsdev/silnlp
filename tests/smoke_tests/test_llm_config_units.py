from dataclasses import dataclass

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
