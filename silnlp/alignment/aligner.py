from abc import ABC, abstractmethod

from .lexicon import Lexicon


class Aligner(ABC):
    def __init__(self, id: str, model_dir: str) -> None:
        self._id = id
        self._model_dir = model_dir

    @property
    def id(self) -> str:
        return self._id

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def has_inverse_model(self) -> bool:
        return True

    @abstractmethod
    def train(self, src_file_path: str, trg_file_path: str) -> None:
        pass

    @abstractmethod
    def align(self, out_file_path: str, sym_heuristic: str = "grow-diag-final-and") -> None:
        pass

    @abstractmethod
    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        pass

    @abstractmethod
    def get_inverse_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        pass

    @abstractmethod
    def extract_lexicon(self, out_file_path: str) -> None:
        pass

