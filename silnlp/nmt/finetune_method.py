class FinetuneMethod:
    """How an LLM is fine-tuned: in full, or through a low-rank adapter that may be quantized, DoRA, or both.
    The configured name is checked whenever it is asked about, so an experiment that never trains is not
    stopped by a misspelt method."""

    _FULL = "full"
    _ADAPTER = ("lora", "qlora", "dora", "qdora")
    _QUANTIZED = ("qlora", "qdora")
    _DORA = ("dora", "qdora")
    _VALID = (_FULL,) + _ADAPTER

    def __init__(self, configured: str) -> None:
        self._configured = configured

    def __str__(self) -> str:
        return self._name()

    def is_full(self) -> bool:
        return self._name() == self._FULL

    def uses_quantization(self) -> bool:
        return self._name() in self._QUANTIZED

    def uses_dora(self) -> bool:
        return self._name() in self._DORA

    def _name(self) -> str:
        name = self._configured.lower()
        if name not in self._VALID:
            raise ValueError(f"Unknown finetune_method '{name}'. Valid options: {', '.join(self._VALID)}.")
        return name
