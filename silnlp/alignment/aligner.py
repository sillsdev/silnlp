import abc


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

    @abc.abstractmethod
    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        pass
