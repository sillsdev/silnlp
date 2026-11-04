from pathlib import Path
from typing import Optional

from machine.scripture import book_number_to_id

from ..common.paratext import book_file_name_digits

INFERENCE_DIRECTORY = "infer"
UNTRAINED_MODEL_STEP = "base"


class DraftFiles:
    """The drafts a translation run writes into an experiment directory, and the translate config
    that says which ones to produce."""

    def __init__(self, exp_dir: Path) -> None:
        self._exp_dir = exp_dir

    def translate_config(self) -> Path:
        return self._exp_dir / "translate_config.yml"

    def inference_directory(
        self, step: str, src_project: Optional[str] = None, trg_project: Optional[str] = None
    ) -> Path:
        directory = self._exp_dir / INFERENCE_DIRECTORY / step
        if src_project is not None:
            directory = directory / src_project
        if trg_project is not None:
            directory = directory / trg_project
        return directory

    def draft(self, step: str, src_project: str, book_num: int) -> Path:
        book = book_number_to_id(book_num)
        return self.inference_directory(step, src_project) / f"{book_file_name_digits(book_num)}{book}.SFM"
