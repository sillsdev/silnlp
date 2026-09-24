from pathlib import Path


class PredictionFile:
    """What a checkpoint wrote for one test set, and where the drafts go when it produced several."""

    _EXTENSION = ".txt"

    def __init__(self, path: Path) -> None:
        self._path = path

    def draft(self, index: int) -> Path:
        head, separator, tail = self._path.name.partition(f"{self._EXTENSION}.")
        if separator == "":
            raise ValueError(f"{self._path.name} does not name the checkpoint it came from")
        # The checkpoint step follows the extension, so the draft goes before it rather than last.
        return self._path.with_name(f"{head}.{index}{self._EXTENSION}.{tail}")
