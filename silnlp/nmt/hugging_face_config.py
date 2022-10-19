from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from .config import CheckpointType, Config, NMTModel
from .tokenizer import Tokenizer


class HuggingFaceConfig(Config):
    def create_model(self, mixed_precision: bool = False) -> NMTModel:
        return HuggingFaceNMTModel(self, mixed_precision)

    def create_tokenizer(self) -> Tokenizer:
        ...

    def _build_vocabs(self) -> None:
        ...


class HuggingFaceNMTModel(NMTModel):
    def __init__(self, config: HuggingFaceConfig, mixed_precision: bool) -> None:
        ...

    def train(self, num_devices: int = 1) -> None:
        ...

    def save_effective_config(self, path: Path) -> None:
        ...

    def translate_text_files(
        self,
        input_paths: List[Union[Path, Sequence[Path]]],
        translation_paths: List[Path],
        checkpoint: Union[CheckpointType, str, int] = CheckpointType.BEST,
    ) -> None:
        ...

    def translate(
        self,
        sentences: Iterable[Union[str, Sequence[str]]],
        src_iso: Optional[str] = None,
        trg_iso: Optional[str] = None,
        checkpoint: Union[CheckpointType, str, int] = CheckpointType.BEST,
    ) -> Iterable[str]:
        ...

    def get_checkpoint_step(self, checkpoint: Union[CheckpointType, str, int]) -> int:
        ...
