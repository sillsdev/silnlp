from typing import Any

from torch import Tensor

LABEL_PADDING = -100


class DecoderInputs:
    """The decoder inputs of a training batch: its labels shifted one place right."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def from_labels(self, labels: Tensor) -> Tensor:
        config = self._model.config
        if config.decoder_start_token_id is None:
            raise ValueError("The model's decoder_start_token_id has to be defined.")
        if config.pad_token_id is None:
            raise ValueError("The model's pad_token_id has to be defined.")

        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = config.decoder_start_token_id
        # The loss ignores the label padding, but the decoder cannot be fed it.
        shifted.masked_fill_(shifted == LABEL_PADDING, config.pad_token_id)
        return shifted
