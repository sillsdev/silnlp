import json
from pathlib import Path

import pytest
import yaml

from silnlp.nmt.model_name import ModelName
from silnlp.nmt.parent_model import ParentModel


@pytest.fixture
def parent_dir(environment) -> Path:
    directory = environment.get_mt_exp_dir("parent")
    (directory / "run").mkdir(parents=True)
    return directory


def configure(parent_dir: Path, model: str) -> None:
    (parent_dir / "config.yml").write_text(yaml.dump({"model": model}), encoding="utf-8")


def trainer_state(parent_dir: Path, best: str = None, max_steps: int = 1000) -> None:
    state = {"max_steps": max_steps}
    if best is not None:
        state["best_model_checkpoint"] = best
    (parent_dir / "run" / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")


def test_the_parents_best_checkpoint_is_used_when_it_has_one(environment, parent_dir):
    configure(parent_dir, "facebook/nllb-200-distilled-600M")
    trainer_state(parent_dir, best="/somewhere/else/checkpoint-400")

    checkpoint = ParentModel("parent", environment).checkpoint_for(ModelName("facebook/nllb-200-distilled-1.3B"))

    assert checkpoint.path.endswith("run/checkpoint-400")


def test_the_planned_final_checkpoint_is_used_when_there_is_no_best(environment, parent_dir):
    configure(parent_dir, "facebook/nllb-200-distilled-600M")
    trainer_state(parent_dir, max_steps=5000)

    checkpoint = ParentModel("parent", environment).checkpoint_for(ModelName("facebook/nllb-200-distilled-1.3B"))

    assert checkpoint.path.endswith("run/checkpoint-5000")


def test_the_family_comes_from_the_parents_own_config(environment, parent_dir):
    configure(parent_dir, "facebook/nllb-200-distilled-600M")
    trainer_state(parent_dir)

    checkpoint = ParentModel("parent", environment).checkpoint_for(ModelName("facebook/nllb-200-distilled-1.3B"))

    assert checkpoint.family.is_nllb()


def test_a_parent_from_another_model_family_is_rejected(environment, parent_dir):
    configure(parent_dir, "google/madlad400-3b-mt")
    trainer_state(parent_dir)

    with pytest.raises(ValueError, match="Unmatched model prefix"):
        ParentModel("parent", environment).checkpoint_for(ModelName("facebook/nllb-200-distilled-1.3B"))


def test_a_parent_that_never_finished_training_is_reported_before_its_family_is_judged(environment, parent_dir):
    # The checkpoint is resolved first, so a parent with no training run fails on that rather than on its family.
    configure(parent_dir, "google/madlad400-3b-mt")

    with pytest.raises(FileNotFoundError):
        ParentModel("parent", environment).checkpoint_for(ModelName("facebook/nllb-200-distilled-1.3B"))
