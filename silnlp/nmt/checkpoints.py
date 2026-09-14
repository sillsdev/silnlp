import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union


class CheckpointType(Enum):
    LAST = auto()
    BEST = auto()
    AVERAGE = auto()
    OTHER = auto()


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    step: int


class CheckpointDirectory:
    _PREFIX = "checkpoint-"
    _TRAINER_STATE = "trainer_state.json"
    _OPTIMIZER_STATE_FILES = ("optimizer.pt", "rng_state.pth", "scaler.pt", "scheduler.pt")
    _TOKENIZER_FILES = (
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
    )

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir

    def exists(self) -> bool:
        return self._model_dir.exists()

    def resolve(self, specifier: Union[CheckpointType, str, int]) -> Checkpoint:
        checkpoint_type, step = self._interpret(specifier)
        if checkpoint_type is CheckpointType.BEST:
            return self.best()
        if checkpoint_type is CheckpointType.LAST:
            latest = self.latest()
            if latest is None:
                raise ValueError(f"No checkpoints found in {self._model_dir}.")
            return latest
        if checkpoint_type is CheckpointType.OTHER and step is not None:
            return self._at_step(step)
        raise ValueError(f"Unsupported checkpoint type: {checkpoint_type}.")

    def latest(self) -> Optional[Checkpoint]:
        paths = self._checkpoint_paths()
        if len(paths) == 0:
            return None
        return self._at_path(max(paths, key=self._step_of))

    def best(self) -> Checkpoint:
        trainer_state = self._read_trainer_state()
        # The recorded path is the one the training machine used, so only its name transfers.
        return self._at_path(self._model_dir / Path(trainer_state["best_model_checkpoint"]).name)

    def has_best(self) -> bool:
        if not (self._model_dir / self._TRAINER_STATE).is_file():
            return False
        return self._read_trainer_state().get("best_model_checkpoint") is not None

    def steps(self) -> List[int]:
        return sorted(self._step_of(path) for path in self._checkpoint_paths())

    def planned_final(self) -> Checkpoint:
        return self._at_step(self._read_trainer_state()["max_steps"])

    def discard_optimizer_state(self) -> None:
        self._discard(self._OPTIMIZER_STATE_FILES)

    def discard_tokenizers(self) -> None:
        self._discard(self._TOKENIZER_FILES)

    def _interpret(self, specifier: Union[CheckpointType, str, int]) -> Tuple[CheckpointType, Optional[int]]:
        if isinstance(specifier, CheckpointType):
            return specifier, None
        if isinstance(specifier, int):
            return CheckpointType.OTHER, specifier
        name = specifier.lower()
        if "avg" in name:
            return CheckpointType.AVERAGE, None
        if "best" in name:
            return CheckpointType.BEST, None
        if "last" in name:
            return CheckpointType.LAST, None
        return CheckpointType.OTHER, int(specifier)

    def _discard(self, file_names: Iterable[str]) -> None:
        for checkpoint_path in self._checkpoint_paths():
            for file_name in file_names:
                path = checkpoint_path / file_name
                if path.is_file():
                    path.unlink()

    def _checkpoint_paths(self) -> List[Path]:
        return [path for path in self._model_dir.glob(f"{self._PREFIX}*") if path.is_dir()]

    def _at_path(self, path: Path) -> Checkpoint:
        return Checkpoint(path, self._step_of(path))

    def _at_step(self, step: int) -> Checkpoint:
        return Checkpoint(self._model_dir / f"{self._PREFIX}{step}", step)

    def _step_of(self, path: Path) -> int:
        return int(path.name[len(self._PREFIX) :])

    def _read_trainer_state(self) -> dict:
        with (self._model_dir / self._TRAINER_STATE).open("r", encoding="utf-8") as file:
            return json.load(file)
