from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from silnlp.nmt.decoder_inputs import DecoderInputs

LABEL_PADDING = -100


@dataclass
class StubConfig:
    decoder_start_token_id: Optional[int] = 2
    pad_token_id: Optional[int] = 1


@dataclass
class StubModel:
    config: StubConfig


def decoder_inputs(**config) -> DecoderInputs:
    return DecoderInputs(StubModel(StubConfig(**config)))


def test_labels_are_shifted_one_place_right_behind_the_start_token():
    labels = torch.tensor([[10, 11, 12]])
    assert decoder_inputs().from_labels(labels).tolist() == [[2, 10, 11]]


def test_every_row_of_the_batch_is_shifted():
    labels = torch.tensor([[10, 11], [20, 21]])
    assert decoder_inputs().from_labels(labels).tolist() == [[2, 10], [2, 20]]


def test_label_padding_becomes_the_pad_token_because_the_decoder_cannot_consume_it():
    labels = torch.tensor([[10, LABEL_PADDING, LABEL_PADDING]])
    assert decoder_inputs().from_labels(labels).tolist() == [[2, 10, 1]]


def test_the_labels_are_left_untouched():
    labels = torch.tensor([[10, 11]])
    decoder_inputs().from_labels(labels)
    assert labels.tolist() == [[10, 11]]


def test_a_model_without_a_decoder_start_token_cannot_be_fed():
    with pytest.raises(ValueError, match="decoder_start_token_id"):
        decoder_inputs(decoder_start_token_id=None).from_labels(torch.tensor([[10]]))


def test_a_model_without_a_pad_token_cannot_be_fed():
    with pytest.raises(ValueError, match="pad_token_id"):
        decoder_inputs(pad_token_id=None).from_labels(torch.tensor([[10]]))


def test_the_token_ids_are_read_when_the_batch_is_built_not_when_configured():
    # _configure_model sets decoder_start_token_id on the model after the collator exists.
    model = StubModel(StubConfig(decoder_start_token_id=None))
    inputs = DecoderInputs(model)
    model.config.decoder_start_token_id = 5
    assert inputs.from_labels(torch.tensor([[10, 11]])).tolist() == [[5, 10]]
