import json
from pathlib import Path

import pytest

from silnlp.nmt.checkpoints import CheckpointDirectory, CheckpointType


def make_checkpoints(model_dir: Path, steps: list) -> None:
    for step in steps:
        (model_dir / f"checkpoint-{step}").mkdir(parents=True)


def write_trainer_state(model_dir: Path, **state) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "trainer_state.json").open("w", encoding="utf-8") as file:
        json.dump(state, file)


def test_latest_compares_steps_numerically(tmp_path: Path):
    make_checkpoints(tmp_path, [2, 5, 10])
    assert CheckpointDirectory(tmp_path).latest().step == 10


def test_latest_ignores_files_named_like_checkpoints(tmp_path: Path):
    make_checkpoints(tmp_path, [2])
    (tmp_path / "checkpoint-99").write_text("")
    assert CheckpointDirectory(tmp_path).latest().step == 2


def test_latest_is_none_when_there_are_no_checkpoints(tmp_path: Path):
    assert CheckpointDirectory(tmp_path).latest() is None


def test_steps_are_sorted_numerically(tmp_path: Path):
    make_checkpoints(tmp_path, [10, 2, 5])
    assert CheckpointDirectory(tmp_path).steps() == [2, 5, 10]


def test_steps_is_empty_when_the_directory_does_not_exist(tmp_path: Path):
    assert CheckpointDirectory(tmp_path / "missing").steps() == []


def test_exists_reflects_the_directory(tmp_path: Path):
    assert CheckpointDirectory(tmp_path).exists()
    assert not CheckpointDirectory(tmp_path / "missing").exists()


@pytest.mark.parametrize("specifier", ["last", "LAST", "Last", CheckpointType.LAST])
def test_resolve_last(tmp_path: Path, specifier):
    make_checkpoints(tmp_path, [2, 10])
    checkpoint = CheckpointDirectory(tmp_path).resolve(specifier)
    assert checkpoint.step == 10
    assert checkpoint.path == tmp_path / "checkpoint-10"


def test_resolve_last_without_checkpoints_is_an_error(tmp_path: Path):
    with pytest.raises(ValueError):
        CheckpointDirectory(tmp_path).resolve(CheckpointType.LAST)


@pytest.mark.parametrize("specifier", ["best", "BEST", CheckpointType.BEST])
def test_resolve_best(tmp_path: Path, specifier):
    make_checkpoints(tmp_path, [2, 10])
    write_trainer_state(tmp_path, best_model_checkpoint="checkpoint-2")
    checkpoint = CheckpointDirectory(tmp_path).resolve(specifier)
    assert checkpoint.step == 2
    assert checkpoint.path == tmp_path / "checkpoint-2"


def test_resolve_best_keeps_only_the_name_of_a_recorded_path(tmp_path: Path):
    # trainer_state.json records the absolute path of the machine that trained the model.
    make_checkpoints(tmp_path, [2])
    write_trainer_state(tmp_path, best_model_checkpoint="/some/other/machine/run/checkpoint-2")
    assert CheckpointDirectory(tmp_path).resolve(CheckpointType.BEST).path == tmp_path / "checkpoint-2"


@pytest.mark.parametrize("specifier", [500, "500"])
def test_resolve_a_specific_step_does_not_require_it_to_exist(tmp_path: Path, specifier):
    checkpoint = CheckpointDirectory(tmp_path).resolve(specifier)
    assert checkpoint.step == 500
    assert checkpoint.path == tmp_path / "checkpoint-500"


@pytest.mark.parametrize("specifier", ["avg", "AVG", "checkpoint-avg", CheckpointType.AVERAGE])
def test_resolve_average_is_unsupported(tmp_path: Path, specifier):
    with pytest.raises(ValueError):
        CheckpointDirectory(tmp_path).resolve(specifier)


@pytest.mark.parametrize("specifier", ["average", "newest", ""])
def test_resolve_an_unrecognized_name_is_an_error(tmp_path: Path, specifier):
    # Only "avg", "best" and "last" are recognized; anything else has to be a step number.
    with pytest.raises(ValueError):
        CheckpointDirectory(tmp_path).resolve(specifier)


def test_has_best_is_false_without_a_trainer_state(tmp_path: Path):
    assert not CheckpointDirectory(tmp_path).has_best()


def test_has_best_is_false_when_no_best_was_recorded(tmp_path: Path):
    write_trainer_state(tmp_path, best_model_checkpoint=None)
    assert not CheckpointDirectory(tmp_path).has_best()


def test_has_best_is_true_when_a_best_was_recorded(tmp_path: Path):
    write_trainer_state(tmp_path, best_model_checkpoint="checkpoint-2")
    assert CheckpointDirectory(tmp_path).has_best()


def test_planned_final_comes_from_the_configured_step_count(tmp_path: Path):
    # The parent experiment's final checkpoint is named after max_steps, whether or not it is on disk.
    write_trainer_state(tmp_path, max_steps=5000)
    checkpoint = CheckpointDirectory(tmp_path).planned_final()
    assert checkpoint.step == 5000
    assert checkpoint.path == tmp_path / "checkpoint-5000"


def test_discard_optimizer_state_leaves_the_weights(tmp_path: Path):
    make_checkpoints(tmp_path, [2, 10])
    for step in (2, 10):
        for name in ("optimizer.pt", "rng_state.pth", "scaler.pt", "scheduler.pt", "model.safetensors"):
            (tmp_path / f"checkpoint-{step}" / name).write_text("")

    CheckpointDirectory(tmp_path).discard_optimizer_state()

    for step in (2, 10):
        checkpoint_dir = tmp_path / f"checkpoint-{step}"
        assert [path.name for path in sorted(checkpoint_dir.iterdir())] == ["model.safetensors"]


def test_discard_optimizer_state_tolerates_a_checkpoint_without_one(tmp_path: Path):
    make_checkpoints(tmp_path, [2])
    CheckpointDirectory(tmp_path).discard_optimizer_state()
    assert list((tmp_path / "checkpoint-2").iterdir()) == []


def test_discard_tokenizers_leaves_the_weights(tmp_path: Path):
    make_checkpoints(tmp_path, [2])
    for name in (
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "model.safetensors",
    ):
        (tmp_path / "checkpoint-2" / name).write_text("")

    CheckpointDirectory(tmp_path).discard_tokenizers()

    assert [path.name for path in sorted((tmp_path / "checkpoint-2").iterdir())] == ["model.safetensors"]


def test_discarding_ignores_anything_that_is_not_a_checkpoint(tmp_path: Path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "optimizer.pt").write_text("")
    CheckpointDirectory(tmp_path).discard_optimizer_state()
    assert (tmp_path / "logs" / "optimizer.pt").is_file()
