import logging
from dataclasses import dataclass

import yaml

from ..common.environment import SilNlpEnv
from .checkpoints import CheckpointDirectory
from .model_name import ModelName

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParentCheckpoint:
    path: str
    family: ModelName


class ParentModel:
    """The model an experiment continues from: another experiment's best checkpoint, or the one its
    training run was planned to end at. It has to belong to the same model family as this experiment."""

    def __init__(self, exp_name: str, environment: SilNlpEnv) -> None:
        self._exp_dir = environment.get_mt_exp_dir(exp_name)

    def checkpoint_for(self, model_name: ModelName) -> ParentCheckpoint:
        family = self._configured_family()
        checkpoints = CheckpointDirectory(self._exp_dir / "run")
        checkpoint = checkpoints.best() if checkpoints.has_best() else checkpoints.planned_final()
        LOGGER.info("Using parent model. This might be different from the model specified in config.")
        if not family.same_family_as(model_name):
            LOGGER.error("The parent model and the config model are not in the same type.")
            raise ValueError(f"Unmatched model prefix {family} and {model_name}")
        return ParentCheckpoint(str(checkpoint.path), family)

    def _configured_family(self) -> ModelName:
        with (self._exp_dir / "config.yml").open("r", encoding="utf-8") as file:
            return ModelName(yaml.safe_load(file).get("model"))
