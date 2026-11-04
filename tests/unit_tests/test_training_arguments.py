from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from silnlp.nmt.training_arguments import TrainingArgumentsMapping


class Strategy(Enum):
    STEPS = "steps"


@dataclass
class StandInTrainingArguments:
    max_steps: int = 5000
    learning_rate: float = 0.0002
    eval_strategy: Strategy = Strategy.STEPS
    early_stopping: Optional[int] = None


def mapping() -> TrainingArgumentsMapping:
    return TrainingArgumentsMapping({"train": {"max_steps"}, "params": {"learning_rate"}})


def config() -> dict:
    return {"train": {"max_steps": 100}, "params": {"learning_rate": 0.001}}


def test_the_named_config_values_are_collected_into_one_flat_set_of_arguments():
    args = mapping().collect(config(), precision_args={}, clearml_queue=None)

    assert args["max_steps"] == 100
    assert args["learning_rate"] == 0.001


def test_a_value_the_config_does_not_set_is_left_out():
    args = mapping().collect({"train": {}, "params": {"learning_rate": 0.001}}, {}, None)

    assert "max_steps" not in args


def test_a_value_explicitly_set_to_nothing_is_left_out():
    args = mapping().collect({"train": {"max_steps": None}, "params": {}}, {}, None)

    assert "max_steps" not in args


def test_the_precision_flags_override_whatever_the_config_asked_for():
    args = mapping().collect(config(), precision_args={"max_steps": 7}, clearml_queue=None)

    assert args["max_steps"] == 7


def test_reporting_is_off_unless_the_run_was_queued():
    assert mapping().collect(config(), {}, clearml_queue=None)["report_to"] == "none"
    assert mapping().collect(config(), {}, clearml_queue="gpu")["report_to"] == "all"


def test_the_effective_config_records_what_the_arguments_settled_on(tmp_path: Path):
    path = tmp_path / "effective-config.yml"
    mapping().write_effective_config(path, config(), StandInTrainingArguments())

    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["train"]["max_steps"] == 5000
    assert written["params"]["learning_rate"] == 0.0002


def test_the_original_config_is_not_changed_by_writing_the_effective_one(tmp_path: Path):
    original = config()
    mapping().write_effective_config(tmp_path / "out.yml", original, StandInTrainingArguments())

    assert original == config()


def test_an_enumerated_value_is_written_as_its_plain_value(tmp_path: Path):
    path = tmp_path / "out.yml"
    TrainingArgumentsMapping({"eval": {"eval_strategy"}}).write_effective_config(
        path, {"eval": {"eval_strategy": "no"}}, StandInTrainingArguments()
    )

    assert yaml.safe_load(path.read_text(encoding="utf-8"))["eval"]["eval_strategy"] == "steps"


def test_an_argument_that_settled_on_nothing_is_dropped_from_the_effective_config(tmp_path: Path):
    path = tmp_path / "out.yml"
    TrainingArgumentsMapping({"eval": {"early_stopping"}}).write_effective_config(
        path, {"eval": {"early_stopping": 3}}, StandInTrainingArguments()
    )

    assert "early_stopping" not in yaml.safe_load(path.read_text(encoding="utf-8"))["eval"]
