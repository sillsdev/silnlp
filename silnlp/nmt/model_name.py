from pathlib import Path


class ModelName:
    _NLLB = "facebook/nllb-200"
    _MADLAD = "google/madlad400"
    _SEQ2SEQ_FAMILIES = (_NLLB, _MADLAD)
    _T5_FAMILIES = (_MADLAD,)
    _DECODER_ONLY_PREFIXES = (
        "google/gemma",
        "google/translate-gemma",
        "google/translategemma",
        "tencent/Hunyuan",
        "Hunyuan-MT",
    )
    _TRANSLATE_GEMMA_PREFIXES = ("google/translate-gemma", "google/translategemma")
    _SENTENCE_PIECE_SETTINGS = {
        _NLLB: {"type": "BPE", "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]},
        _MADLAD: {"type": "Unigram", "special_tokens": ["<unk>", "<s>", "</s>"], "unk_token": "<unk>"},
    }

    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name

    def is_nllb(self) -> bool:
        return self._family() == self._NLLB

    def is_madlad(self) -> bool:
        return self._family() == self._MADLAD

    def is_t5(self) -> bool:
        return self._family() in self._T5_FAMILIES

    def looks_decoder_only(self) -> bool:
        return self._name.startswith(self._DECODER_ONLY_PREFIXES)

    def uses_translate_gemma_template(self) -> bool:
        return self._name.lower().startswith(self._TRANSLATE_GEMMA_PREFIXES)

    def same_family_as(self, other: "ModelName") -> bool:
        return self._family() == other._family()

    def tokenizer_assets_dir(self, assets_dir: Path) -> Path:
        return assets_dir / "tokenizers" / self._family()

    def sentence_piece_settings(self) -> dict:
        return self._SENTENCE_PIECE_SETTINGS[self._family()]

    def _family(self) -> str:
        for family in self._SEQ2SEQ_FAMILIES:
            if self._name.startswith(family):
                return family
        return ""
