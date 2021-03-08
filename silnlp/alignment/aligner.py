import abc

from .lexicon import Lexicon


class Aligner(abc.ABC):
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

    @abc.abstractmethod
    def train(self, src_file_path: str, trg_file_path: str) -> None:
        pass

    @abc.abstractmethod
    def align(self, out_file_path: str, sym_heuristic: str = "grow-diag-final-and") -> None:
        pass

    @abc.abstractmethod
    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        pass

    @abc.abstractmethod
    def get_inverse_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        pass

    @abc.abstractmethod
    def extract_lexicon(self, out_file_path: str) -> None:
        pass

