from pathlib import Path
from typing import Any, List, Optional, cast

from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .experiment_files import ExperimentFiles
from .pretrained_tokenizer import PretrainedTokenizer


class TokenizedBatchEncoder:
    """Turns sentences that are already split into tokens into padded model inputs."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def encode(self, batch_tokens: List[List[str]], return_tensors: Optional[Any] = None) -> BatchEncoding:
        input_ids = [cast(List[int], self._tokenizer.convert_tokens_to_ids(tokens)) for tokens in batch_tokens]
        return self._tokenizer.pad({"input_ids": input_ids}, padding=False, return_tensors=return_tensors)


class Seq2SeqTrainingDataSets:
    """The data sets a training run reads: the experiment's parallel text, encoded with its tokenizer."""

    def __init__(self, files: ExperimentFiles, pretrained: PretrainedTokenizer) -> None:
        self._files = files
        self._pretrained = pretrained

    def training(self, training_args: Any) -> Optional[Dataset]:
        return self._encoded(self._files.train_source(), self._files.train_target(), training_args, "train")

    def validation(self, training_args: Any) -> Optional[Dataset]:
        return self._encoded(
            self._files.validation_source(), self._files.validation_target(), training_args, "validation"
        )

    def _encoded(self, source: Path, target: Path, training_args: Any, name: str) -> Optional[Dataset]:
        data_set = self._read(source, target)
        if data_set is None:
            return None
        with training_args.main_process_first(desc=f"{name} dataset map encoding"):
            return data_set.map(
                self._encode,
                batched=True,
                remove_columns=data_set.column_names,
                desc=f"Encoding {name} dataset",
            )

    def _read(self, source: Path, target: Path) -> Optional[Dataset]:
        if not source.is_file() or not target.is_file():
            return None
        data = []
        with (
            open(source, "r", encoding="utf-8-sig") as source_file,
            open(target, "r", encoding="utf-8-sig") as target_file,
        ):
            for source_line, target_line in zip(source_file, target_file):
                data.append({"src": source_line.strip(), "trg": target_line.strip()})
        return Dataset.from_dict({"translation": data})

    def _encode(self, examples: dict) -> dict:
        encoder = TokenizedBatchEncoder(self._pretrained.load())
        model_inputs = encoder.encode([example["src"].split() for example in examples["translation"]])
        labels = encoder.encode([example["trg"].split() for example in examples["translation"]])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
