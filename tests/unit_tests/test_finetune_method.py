import pytest

from silnlp.nmt.finetune_method import FinetuneMethod


def test_full_fine_tuning_uses_no_adapter():
    method = FinetuneMethod("full")

    assert method.is_full()
    assert not method.uses_quantization()
    assert not method.uses_dora()


def test_lora_is_an_unquantized_adapter():
    method = FinetuneMethod("lora")

    assert not method.is_full()
    assert not method.uses_quantization()
    assert not method.uses_dora()


def test_qlora_is_a_quantized_adapter():
    method = FinetuneMethod("qlora")

    assert not method.is_full()
    assert method.uses_quantization()
    assert not method.uses_dora()


def test_dora_is_an_unquantized_dora_adapter():
    method = FinetuneMethod("dora")

    assert not method.is_full()
    assert not method.uses_quantization()
    assert method.uses_dora()


def test_qdora_is_a_quantized_dora_adapter():
    method = FinetuneMethod("qdora")

    assert not method.is_full()
    assert method.uses_quantization()
    assert method.uses_dora()


def test_the_method_is_named_by_its_lower_case_form():
    assert str(FinetuneMethod("QDoRA")) == "qdora"


def test_the_method_is_recognized_whatever_its_case():
    assert FinetuneMethod("QDoRA").uses_dora()
    assert FinetuneMethod("QLoRA").uses_quantization()


def test_an_unknown_method_is_rejected_with_the_valid_ones_named():
    with pytest.raises(ValueError, match=r"Unknown finetune_method 'bogus'\. Valid options: full, lora, qlora"):
        FinetuneMethod("bogus").is_full()


def test_an_unknown_method_is_only_rejected_once_it_is_asked_about():
    # An experiment that only translates should not be stopped by a misspelt training setting.
    method = FinetuneMethod("bogus")

    with pytest.raises(ValueError):
        method.uses_quantization()
