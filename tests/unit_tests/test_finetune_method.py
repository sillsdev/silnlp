import pytest

from silnlp.nmt.finetune_method import FinetuneMethod


@pytest.mark.parametrize(
    "name,adapter,quantized,dora",
    [
        ("full", False, False, False),
        ("lora", True, False, False),
        ("qlora", True, True, False),
        ("dora", True, False, True),
        ("qdora", True, True, True),
    ],
)
def test_each_method_is_placed_on_all_three_axes(name, adapter, quantized, dora):
    method = FinetuneMethod(name)

    assert method.is_full() is (not adapter)
    assert method.uses_adapter() is adapter
    assert method.uses_quantization() is quantized
    assert method.uses_dora() is dora


def test_the_method_is_named_by_its_lower_case_form():
    assert str(FinetuneMethod("QDoRA")) == "qdora"


def test_the_method_is_recognized_whatever_its_case():
    assert FinetuneMethod("QDoRA").uses_dora()
    assert FinetuneMethod("QLoRA").uses_quantization()


def test_an_unknown_method_is_rejected_with_the_valid_ones_named():
    with pytest.raises(ValueError, match="Unknown finetune_method 'bogus'"):
        FinetuneMethod("bogus")

    with pytest.raises(ValueError, match="full, lora, qlora, dora, qdora"):
        FinetuneMethod("bogus")
