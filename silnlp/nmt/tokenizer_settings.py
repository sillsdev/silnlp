from pathlib import Path
from typing import Optional


class TokenizerSettings:
    """The tokenizer section of the config: which sides of the corpus extend the vocabulary, and by how much."""

    def __init__(self, settings: Optional[dict]) -> None:
        self._settings = settings

    def updates_source(self) -> bool:
        return bool(self._settings) and bool(self._settings.get("update_src"))

    def updates_target(self) -> bool:
        return bool(self._settings) and bool(self._settings.get("update_trg"))

    def updates_either(self) -> bool:
        return self.updates_source() or self.updates_target()

    def updates_both(self) -> bool:
        return self.updates_source() and self.updates_target()

    def shares_vocab(self) -> bool:
        return bool(self._settings) and bool(self._settings.get("share_vocab"))

    def trains_tokens(self) -> bool:
        return bool(self._settings) and bool(self._settings.get("trained_tokens"))

    def source_vocab_size(self) -> int:
        return self._settings.get("src_vocab_size")

    def target_vocab_size(self) -> int:
        return self._settings.get("trg_vocab_size")

    def shared_vocab_size(self) -> int:
        return self.source_vocab_size() + self.target_vocab_size()


class TokenizerSource:
    """Where an experiment's tokenizer is loaded from, which changes once the experiment has one of its own."""

    _SENTENCE_PIECE_MODELS = ("sentencepiece.bpe.model", "spiece.model")
    _CONFIG = "tokenizer_config.json"

    def __init__(
        self,
        exp_dir: Path,
        tokenizer_assets_dir: Path,
        parent_dir: Optional[Path],
        model: str,
        settings: TokenizerSettings,
    ) -> None:
        self._exp_dir = exp_dir
        self._assets_dir = tokenizer_assets_dir
        self._parent_dir = parent_dir
        self._model = model
        self._settings = settings

    def holds_unconverted_sentence_piece_model(self) -> bool:
        if not self._settings.updates_either():
            return False
        has_model = any((self._exp_dir / name).is_file() for name in self._SENTENCE_PIECE_MODELS)
        return has_model and not (self._exp_dir / self._CONFIG).is_file()

    def has_tokenizer_assets(self) -> bool:
        return (self._assets_dir / self._CONFIG).is_file()

    def path_for_building(self) -> str:
        if not self._settings.updates_either() and (self._exp_dir / self._CONFIG).is_file():
            return str(self._exp_dir)
        if self._settings.updates_either() and self.has_tokenizer_assets():
            return str(self._assets_dir)
        return self._inherited_path()

    def path_for_loading(self) -> str:
        if (self._exp_dir / self._CONFIG).is_file():
            return str(self._exp_dir)
        return self._inherited_path()

    def _inherited_path(self) -> str:
        return str(self._parent_dir) if self._parent_dir is not None else self._model
