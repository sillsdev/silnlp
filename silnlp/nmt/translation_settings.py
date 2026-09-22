from typing import Optional


class TranslationSettings:
    """How a sequence-to-sequence model is asked to translate: in what size batches, by beam search or
    by sampling, and how several drafts of one sentence are produced."""

    def __init__(self, infer: dict, params: dict) -> None:
        self._infer = infer
        self._params = params

    def batch_size(self) -> int:
        return self._infer["infer_batch_size"]

    def multiple_translations_method(self) -> str:
        return self._infer.get("multiple_translations_method")

    def beam_count(self) -> Optional[int]:
        # An unset number of beams falls back to whatever the training arguments were given.
        beams: Optional[int] = self._infer.get("num_beams")
        return beams if beams is not None else self._params.get("generation_num_beams")

    def temperature(self) -> Optional[float]:
        return self._infer.get("temperature")


class ModelSettings:
    """The settings a sequence-to-sequence model is constructed with, rather than trained by."""

    def __init__(self, params: dict) -> None:
        self._params = params

    def dropout(self) -> float:
        return self._params["dropout"]

    def attention_dropout(self) -> float:
        return self._params["attention_dropout"]

    def activation_dropout(self) -> float:
        return self._params["activation_dropout"]

    def attention_implementation(self) -> str:
        return self._params["attn_implementation"]


class CheckpointRetention:
    """What a finished training run keeps in each of its checkpoints."""

    def __init__(self, train: dict) -> None:
        self._train = train

    def keeps_optimizer_state(self) -> bool:
        return not self._train["delete_checkpoint_optimizer_state"]

    def keeps_tokenizers(self) -> bool:
        return not self._train["delete_checkpoint_tokenizer"]
