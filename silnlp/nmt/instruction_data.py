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


class InstructionDataSettings:
    """The train.instruction_data section. Each setting is checked when it is read, so a bad value stops
    only the step that uses it."""

    def __init__(self, section: dict, mt_dir: Path) -> None:
        self._section = section
        self._mt_dir = mt_dir

    def paths(self) -> List[Path]:
        # Kept apart from the corpus files so instruction data never merges into the translation corpora.
        return [self._mt_dir / "instructions" / f"{name}.jsonl" for name in self._section["datasets"]]

    def size(self) -> int:
        size = int(self._section["size"])
        if size < 0:
            raise ValueError(f"train.instruction_data.size must be non-negative, got {size}.")
        return size

    def mix_ratio(self) -> float:
        ratio = float(self._section["mix_ratio"])
        if ratio < 0:
            raise ValueError(f"train.instruction_data.mix_ratio must be non-negative, got {ratio}.")
        if ratio > 1:
            raise ValueError(f"train.instruction_data.mix_ratio must be no greater than 1, got {ratio}.")
        return ratio
