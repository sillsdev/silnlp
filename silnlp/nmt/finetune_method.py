class FinetuneMethod:
    """How an LLM is fine-tuned: in full, or through a low-rank adapter that may be quantized, DoRA, or both."""

    _FULL = "full"
    _ADAPTER = ("lora", "qlora", "dora", "qdora")
    _QUANTIZED = ("qlora", "qdora")
    _DORA = ("dora", "qdora")
    _VALID = (_FULL,) + _ADAPTER

    def __init__(self, configured: str) -> None:
        self._name = configured.lower()
        if self._name not in self._VALID:
            raise ValueError(
                f"Unknown finetune_method '{self._name}'. Valid options: {', '.join(self._VALID)}."
            )

    def __str__(self) -> str:
        return self._name

    def is_full(self) -> bool:
        return self._name == self._FULL

    def uses_adapter(self) -> bool:
        return self._name in self._ADAPTER

    def uses_quantization(self) -> bool:
        return self._name in self._QUANTIZED

    def uses_dora(self) -> bool:
        return self._name in self._DORA
