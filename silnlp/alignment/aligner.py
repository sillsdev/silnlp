import abc
from nlp.alignment.config import get_aligner_name


class Aligner(abc.ABC):
    def __init__(self, id: str, model_dir: str) -> None:
        self._id = id
        self._model_dir = model_dir

    @property
    def name(self) -> str:
        return get_aligner_name(self._id)

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @abc.abstractmethod
    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        pass
