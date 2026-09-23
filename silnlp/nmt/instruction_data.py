import logging
from pathlib import Path
from typing import List

from ..common.corpus import split_corpus
from .experiment_data_set_writer import DataSetWriter
from .experiment_files import ExperimentFiles

LOGGER = logging.getLogger(__name__)


class InstructionData:
    """The instruction examples mixed into an LLM's training, drawn evenly from each of its datasets."""

    def __init__(self, files: ExperimentFiles, output_path: Path, dataset_paths: List[Path], size: int) -> None:
        self._files = files
        self._output_path = output_path
        self._dataset_paths = dataset_paths
        self._size = size

    def delete_previous(self) -> None:
        for path in self._output_path.parent.glob(f"{self._output_path.stem}.*"):
            path.unlink()

    def write(self) -> int:
        if len(self._dataset_paths) == 0:
            return 0

        base_share, remainder = divmod(self._size, len(self._dataset_paths))

        count = 0
        with self._files.open_for_append(self._output_path) as out_file:
            for i, dataset_path in enumerate(self._dataset_paths):
                if not dataset_path.is_file():
                    raise RuntimeError(
                        f"Instruction data file {dataset_path} does not exist. Run "
                        "scripts/prepare_instruction_data.py to generate it."
                    )
                # A dataset smaller than its even share is used in full, without repetition (see split_corpus).
                share = base_share + (1 if i < remainder else 0)
                corpus_size = self._count_nonblank_lines(dataset_path)
                if corpus_size < share:
                    LOGGER.warning(
                        f"Instruction dataset {dataset_path.stem} has {corpus_size} examples, fewer than its "
                        f"even share of {share}; all of it is used, and the shortfall is not made up by the "
                        "other datasets, so the mix will not be even."
                    )
                selected_indices = split_corpus(corpus_size, share)

                index = 0
                with dataset_path.open("r", encoding="utf-8") as in_file:
                    for raw_line in in_file:
                        line = raw_line.strip()
                        if not line:
                            continue
                        if selected_indices is None or index in selected_indices:
                            out_file.write(line + "\n")
                            count += 1
                        index += 1

        LOGGER.info(f"instruction data size: {count}")
        return count

    def _count_nonblank_lines(self, path: Path) -> int:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


class InstructionDataSetWriter(DataSetWriter):
    """Writes an experiment's data sets, and then the instruction data that is trained alongside them."""

    def __init__(self, data_sets: DataSetWriter, instruction_data: InstructionData) -> None:
        self._data_sets = data_sets
        self._instruction_data = instruction_data

    def write(self, stats: bool) -> int:
        self._instruction_data.delete_previous()
        return self._data_sets.write(stats) + self._instruction_data.write()
