import abc


class Aligner(abc.ABC):
    def __init__(self, name: str, model_dir: str) -> None:
        self._name = name
        self._model_dir = model_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @abc.abstractmethod
    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        pass