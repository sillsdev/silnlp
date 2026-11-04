import json
import logging
from pathlib import Path
from typing import List

import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.experiment_preprocessor import DataSetWriter
from silnlp.nmt.instruction_data import InstructionData, InstructionDataSetWriter


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    return directory


@pytest.fixture
def instructions_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "mt" / "instructions"
    directory.mkdir(parents=True)
    return directory


def write_dataset(instructions_dir: Path, name: str, line_count: int) -> Path:
    path = instructions_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for i in range(line_count):
            example = {"turns": [{"role": "user", "content": f"{name}-in-{i}"}], "output": f"{name}-out-{i}"}
            file.write(json.dumps(example) + "\n")
    return path


def instruction_data_for(environment, exp_dir: Path, dataset_paths: List[Path], size: int) -> InstructionData:
    inventory = CorpusInventory([], include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)
    return InstructionData(files, exp_dir / "instruction.jsonl", dataset_paths, size)


def test_no_datasets_write_nothing(environment, exp_dir):
    assert instruction_data_for(environment, exp_dir, [], size=100000).write() == 0
    assert not (exp_dir / "instruction.jsonl").exists()


def test_the_size_is_shared_evenly_and_a_small_dataset_is_used_whole(environment, exp_dir, instructions_dir):
    # "b" has fewer lines than its even share of the requested size (4), so all of it is used.
    paths = [write_dataset(instructions_dir, "a", 10), write_dataset(instructions_dir, "b", 3)]

    count = instruction_data_for(environment, exp_dir, paths, size=8).write()

    assert count == 4 + 3
    lines = (exp_dir / "instruction.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == count
    outputs = {json.loads(line)["output"] for line in lines}
    assert len({output for output in outputs if output.startswith("a-out-")}) == 4
    assert {output for output in outputs if output.startswith("b-out-")} == {"b-out-0", "b-out-1", "b-out-2"}


def test_an_examples_input_stays_with_its_output(environment, exp_dir, instructions_dir):
    paths = [write_dataset(instructions_dir, "a", 10)]

    instruction_data_for(environment, exp_dir, paths, size=5).write()

    for line in (exp_dir / "instruction.jsonl").read_text(encoding="utf-8").splitlines():
        example = json.loads(line)
        assert example["turns"][0]["content"].split("-in-")[1] == example["output"].split("-out-")[1]


def test_a_dataset_smaller_than_its_share_is_warned_about(environment, exp_dir, instructions_dir, caplog):
    paths = [write_dataset(instructions_dir, "a", 10), write_dataset(instructions_dir, "b", 3)]

    with caplog.at_level(logging.WARNING):
        instruction_data_for(environment, exp_dir, paths, size=8).write()

    warnings = [record.message for record in caplog.records if "even share" in record.message]
    assert len(warnings) == 1
    assert "b has 3 examples" in warnings[0]
    assert "share of 4" in warnings[0]


def test_a_dataset_that_exactly_fills_its_share_is_not_warned_about(environment, exp_dir, instructions_dir, caplog):
    paths = [write_dataset(instructions_dir, "a", 10), write_dataset(instructions_dir, "b", 4)]

    with caplog.at_level(logging.WARNING):
        instruction_data_for(environment, exp_dir, paths, size=8).write()

    assert not any("even share" in record.message for record in caplog.records)


def test_a_dataset_file_that_does_not_exist_is_refused(environment, exp_dir, instructions_dir):
    with pytest.raises(RuntimeError, match="does not exist"):
        instruction_data_for(environment, exp_dir, [instructions_dir / "missing.jsonl"], size=10).write()


def test_only_the_earlier_instruction_files_are_deleted(environment, exp_dir):
    for name in ("instruction.jsonl", "instruction.tmp", "train.src.txt", "config.yml"):
        (exp_dir / name).write_text("", encoding="utf-8")

    instruction_data_for(environment, exp_dir, [], size=0).delete_previous()

    assert sorted(path.name for path in exp_dir.iterdir()) == ["config.yml", "train.src.txt"]


class RecordingDataSetWriter(DataSetWriter):
    def __init__(self, exp_dir: Path, train_count: int) -> None:
        self._exp_dir = exp_dir
        self._train_count = train_count
        self.stats: List[bool] = []
        self.found_stale_instruction_data = False

    def write(self, stats: bool) -> int:
        self.stats.append(stats)
        self.found_stale_instruction_data = (self._exp_dir / "instruction.jsonl").exists()
        return self._train_count


def test_the_instruction_examples_are_counted_with_the_training_sentences(environment, exp_dir, instructions_dir):
    data_sets = RecordingDataSetWriter(exp_dir, train_count=40)
    instruction_data = instruction_data_for(environment, exp_dir, [write_dataset(instructions_dir, "a", 10)], 5)

    assert InstructionDataSetWriter(data_sets, instruction_data).write(stats=False) == 45


def test_the_data_sets_are_told_whether_to_write_statistics(environment, exp_dir):
    data_sets = RecordingDataSetWriter(exp_dir, train_count=0)

    InstructionDataSetWriter(data_sets, instruction_data_for(environment, exp_dir, [], 0)).write(stats=True)

    assert data_sets.stats == [True]


def test_instruction_data_from_an_earlier_run_is_cleared_before_anything_is_written(
    environment, exp_dir, instructions_dir
):
    (exp_dir / "instruction.jsonl").write_text("stale\n", encoding="utf-8")
    data_sets = RecordingDataSetWriter(exp_dir, train_count=0)
    instruction_data = instruction_data_for(environment, exp_dir, [write_dataset(instructions_dir, "a", 3)], 3)

    InstructionDataSetWriter(data_sets, instruction_data).write(stats=False)

    assert not data_sets.found_stale_instruction_data
    assert "stale" not in (exp_dir / "instruction.jsonl").read_text(encoding="utf-8")


def local_llm_config(tmp_path: Path, instruction_data: dict):
    from silnlp.common.environment import SilNlpEnv
    from silnlp.nmt.local_llm_config import LocalLLMConfig

    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path / "mt")
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(exist_ok=True)
    config = {
        "data": {"corpus_pairs": [], "lang_codes": {}, "terms": {"train": False, "include_glosses": False}},
        "model": "google/gemma-2-2b-it",
        "train": {"instruction_data": instruction_data},
    }
    return LocalLLMConfig(exp_dir, config, environment), exp_dir


def test_a_local_llm_experiment_writes_its_instruction_data_when_preprocessed(tmp_path, instructions_dir):
    config, exp_dir = local_llm_config(tmp_path, {"datasets": ["a"], "size": 4})
    write_dataset(instructions_dir, "a", 10)

    config.create_data_set_writer(force_align=False).write(stats=False)

    assert len((exp_dir / "instruction.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_a_local_llm_experiment_without_instruction_datasets_writes_no_instruction_file(tmp_path):
    config, exp_dir = local_llm_config(tmp_path, {"datasets": [], "size": 4})

    config.create_data_set_writer(force_align=False).write(stats=False)

    assert not (exp_dir / "instruction.jsonl").exists()


def test_a_local_llm_experiment_clears_instruction_data_left_by_an_earlier_run(tmp_path):
    config, exp_dir = local_llm_config(tmp_path, {"datasets": [], "size": 4})
    (exp_dir / "instruction.jsonl").write_text("stale\n", encoding="utf-8")

    config.create_data_set_writer(force_align=False).write(stats=False)

    assert not (exp_dir / "instruction.jsonl").exists()
