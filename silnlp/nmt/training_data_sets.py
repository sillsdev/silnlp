from pathlib import Path
from typing import Any, List, Optional, cast

from datasets.arrow_dataset import Dataset
from machine.corpora import TextFileTextCorpus
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .causal_lm_tokenizer import CausalLMTokenizer
from .experiment_files import ExperimentFiles
from .experiment_languages import ExperimentLanguages
from .pretrained_tokenizer import PretrainedTokenizer
from .prompt_messages import PromptBuilder


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


class CausalLMTrainingDataSets:
    """The data sets a decoder-only training run reads: each sentence pair rendered as the prompt the
    model is given and the completion it is scored against."""

    _LABEL_PADDING = -100

    def __init__(
        self,
        files: ExperimentFiles,
        tokenizer: CausalLMTokenizer,
        prompts: PromptBuilder,
        languages: ExperimentLanguages,
        max_sequence_length: int,
    ) -> None:
        self._files = files
        self._tokenizer = tokenizer
        self._prompts = prompts
        self._languages = languages
        self._max_sequence_length = max_sequence_length

    def training(self) -> Optional[Dataset]:
        return self._encoded(self._files.train_source(), self._files.train_target())

    def validation(self) -> Optional[Dataset]:
        return self._encoded(self._files.validation_source(), self._files.validation_target())

    def _encoded(self, source: Path, target: Path) -> Optional[Dataset]:
        data_set = self._read(source, target)
        if data_set is None:
            return None
        return data_set.map(self._encode, remove_columns=data_set.column_names)

    def _read(self, source: Path, target: Path) -> Optional[Dataset]:
        if not source.is_file() or not target.is_file():
            return None
        corpus = TextFileTextCorpus(source).align_rows(TextFileTextCorpus(target))
        sources: List[str] = []
        targets: List[str] = []
        for row in corpus:
            sources.append(row.source_text)
            targets.append(row.target_text)
        if len(sources) == 0:
            return None
        return Dataset.from_dict({"src": sources, "trg": targets})

    def _encode(self, example: dict) -> dict:
        tokenizer = self._tokenizer.load()
        source_language = self._languages.of(self._languages.training_source_iso())
        target_language = self._languages.of(self._languages.training_target_iso())
        prompt = self._prompts.build(example["src"], source_language, target_language)
        prompt_ids = prompt.apply_prompt_template(tokenizer, add_generation_prompt=True, tokenize=True)
        completion_ids = tokenizer(example["trg"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + completion_ids)[: self._max_sequence_length]
        labels = ([self._LABEL_PADDING] * len(prompt_ids) + completion_ids)[: self._max_sequence_length]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
