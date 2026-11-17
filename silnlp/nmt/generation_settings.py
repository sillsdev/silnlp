from typing import Any, Dict, Optional


class GenerationSettings:
    """How a decoder-only model is asked to generate: by beam search, or by sampling."""

    def __init__(self, infer: dict) -> None:
        self._infer = infer

    def batch_size(self) -> int:
        return self._infer["infer_batch_size"]

    def as_keyword_arguments(self, num_return_sequences: int, pad_token_id: Optional[int]) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self._infer["max_new_tokens"],
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id,
        }
        if self._infer.get("do_sample"):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self._infer["temperature"]
        else:
            gen_kwargs["num_beams"] = self._beams_for(num_return_sequences)
        return gen_kwargs

    def _beams_for(self, num_return_sequences: int) -> int:
        num_beams: int = self._infer["num_beams"]
        if num_return_sequences > num_beams:
            raise RuntimeError(
                f"Beam search cannot return {num_return_sequences} drafts with num_beams set to {num_beams}. "
                "Increase num_beams to at least num_drafts or set do_sample to true."
            )
        return num_beams
