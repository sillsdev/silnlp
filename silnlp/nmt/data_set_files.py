from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional, TextIO


class DataSetFile:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: Optional[TextIO] = None

    def __enter__(self) -> "DataSetFile":
        return self

    def __exit__(self, *exc_info) -> None:
        if self._file is not None:
            self._file.close()

    def write_line(self, line: str) -> None:
        # A split with no rows must leave no file, or the test step would score an empty test set.
        if self._file is None:
            self._file = self._path.open("a", encoding="utf-8", newline="\n")
        self._file.write(line + "\n")


class DataSetFileOpener:
    def __init__(self, stack: ExitStack) -> None:
        self._stack = stack

    def open(self, *paths: Path) -> List[DataSetFile]:
        return [self._stack.enter_context(DataSetFile(path)) for path in paths]


class SplitFiles:
    """The parallel files of one data set split, which must stay aligned line for line."""

    def __init__(self, files: List[DataSetFile], blank_files: List[DataSetFile]) -> None:
        self._files = files
        self._blank_files = blank_files

    def write_row(self, *lines: str) -> None:
        for file, line in zip(self._files, lines, strict=True):
            file.write_line(line)
        for file in self._blank_files:
            file.write_line("")
