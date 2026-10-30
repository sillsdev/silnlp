import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainingArguments

from .causal_lm_trainer import DataCollatorForCausalLM
from .corpora import read_parallel_text_pairs
from .experiment_files import ExperimentFiles
from .experiment_languages import ExperimentLanguages, Language
from .instruction_data import InstructionDataSettings
from .local_llm_prompts import LocalLLMPromptMessages, LocalLLMPrompts

LOGGER = logging.getLogger(__name__)


class OverlengthRowFilter:
    """Drops encoded rows longer than max_seq_length to avoid potential NaN"""

    def __init__(self, max_seq_length: int) -> None:
        self._max_seq_length = max_seq_length

    def apply(self, dataset: Dataset, name: str) -> Dataset:
        # Not loaded from the cache, so a re-run still warns about every dropped row.
        kept = dataset.filter(self._fits, with_indices=True, fn_kwargs={"name": name}, load_from_cache_file=False)
        if len(kept) == 0:
            raise RuntimeError(
                f"All {len(dataset)} {name} rows are longer than params.max_seq_length ({self._max_seq_length})."
            )
        return kept

    def _fits(self, row: dict, index: int, name: str) -> bool:
        length = len(row["input_ids"])
        if length <= self._max_seq_length:
            return True
        LOGGER.warning(
            f"Dropping {name} row {index}: its {length} tokens exceed params.max_seq_length ({self._max_seq_length})."
        )
        return False


class InterleavedTrainDataset(TorchDataset):
    """Combines two datasets of different lengths into one, reshuffling each whenever it's exhausted."""

    def __init__(
        self,
        translation_dataset: Dataset,
        instruction_dataset: Dataset,
        translation_count: int,
        instruction_count: int,
        seed: Optional[int] = None,
    ) -> None:
        self._translation_dataset = translation_dataset
        self._instruction_dataset = instruction_dataset
        self._translation_count = translation_count
        self._instruction_count = instruction_count
        self._seed = random.randrange(2**31) if seed is None else seed
        self._lap_cache: Dict[Tuple[int, int], List[int]] = {}
        self._epoch = 0

    def __len__(self) -> int:
        return self._translation_count + self._instruction_count

    def start_epoch(self, epoch: int) -> None:
        # used to track the position in the instruction dataset, which doesn't repeat per epoch
        self._epoch = epoch

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self._translation_count:
            return self._lap_item(
                self._translation_dataset, salt=0, index=index + self._epoch * self._translation_count
            )
        offset = index - self._translation_count
        return self._lap_item(self._instruction_dataset, salt=1, index=offset + self._epoch * self._instruction_count)

    def _lap_item(self, dataset: Dataset, salt: int, index: int) -> dict:
        pool_size = len(dataset)
        lap, position = divmod(index, pool_size)
        return dataset[self._lap_permutation(pool_size, salt, lap)[position]]

    def _lap_permutation(self, pool_size: int, salt: int, lap: int) -> List[int]:
        key = (salt, lap)
        permutation = self._lap_cache.get(key)
        if permutation is None:
            rng = random.Random(self._seed + salt * 2_147_483_647 + lap * 999_999_937)
            permutation = list(range(pool_size))
            rng.shuffle(permutation)
            self._lap_cache[key] = permutation
        return permutation


class InterleavedEpochCallback(TrainerCallback):
    """Advances an InterleavedTrainDataset each epoch, which a map-style dataset cannot do alone."""

    def __init__(self, dataset: InterleavedTrainDataset) -> None:
        self._dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        # state.epoch is restored from the checkpoint, so a resumed run carries on rather than replays.
        self._dataset.start_epoch(int(state.epoch))


class CausalLMEncoder:
    """Turns a prompt and its completion into token ids, scoring only the completion."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, collator: DataCollatorForCausalLM) -> None:
        self._tokenizer = tokenizer
        self._collator = collator

    def prompt_with_completion(self, prompt: LocalLLMPromptMessages, completion: str) -> dict:
        prompt_ids = prompt.apply_prompt_template(self._tokenizer, add_generation_prompt=True, tokenize=True)
        return self._with_completion(prompt_ids, completion)

    def turns_with_completion(self, turns: List[Dict[str, str]], completion: str) -> dict:
        return self._with_completion(self._rendered(turns), completion)

    def _with_completion(self, prompt_ids: List[int], completion: str) -> dict:
        completion_ids = self._tokenizer(completion, add_special_tokens=False)["input_ids"] + [
            self._tokenizer.eos_token_id
        ]
        input_ids = prompt_ids + completion_ids
        labels = self._collator.mask_ignored_ids(prompt_ids, completion_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    def _rendered(self, turns: List[Dict[str, str]]) -> List[int]:
        """Render turns through the tokenizer's chat template, the same way a live conversation is tokenized."""
        try:
            return self._tokenizer.apply_chat_template(
                turns, add_generation_prompt=True, tokenize=True, return_dict=False
            )
        except Exception:
            # Some chat templates (e.g. Gemma) reject a system role, so fold it into the first user turn and retry.
            if turns and turns[0]["role"] == "system" and len(turns) > 1:
                folded = [{"role": "user", "content": f"{turns[0]['content']}\n\n{turns[1]['content']}"}] + list(
                    turns[2:]
                )
                return self._tokenizer.apply_chat_template(
                    folded, add_generation_prompt=True, tokenize=True, return_dict=False
                )
            raise


class InstructionMix:
    """How the instruction data is mixed into training: in what share, and over how long a run."""

    def __init__(self, path: Path, settings: InstructionDataSettings, num_devices: int, seed: int) -> None:
        self._path = path
        self._settings = settings
        self._num_devices = num_devices
        self._seed = seed

    def load(self) -> Optional[Dataset]:
        if not self._path.is_file() or self._path.stat().st_size == 0:
            return None
        return Dataset.from_json(str(self._path))

    def ratio(self) -> float:
        return self._settings.mix_ratio()

    def interleave(
        self, translations: Dataset, instructions: Dataset, training_args: TrainingArguments
    ) -> InterleavedTrainDataset:
        ratio = self._settings.mix_ratio()
        total_length = self._length(training_args, len(translations), ratio)
        translation_count = round(total_length / (1 + ratio))
        instruction_count = total_length - translation_count
        return InterleavedTrainDataset(
            translations, instructions, translation_count, instruction_count, seed=self._seed
        )

    def _length(self, training_args: TrainingArguments, translation_size: int, ratio: float) -> int:
        """How long the interleaved dataset must be; under an epoch budget the Trainer repeats it."""
        num_devices = max(1, self._num_devices)
        effective_batch_size = (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_devices
        )
        if training_args.max_steps and training_args.max_steps > 0:
            return training_args.max_steps * effective_batch_size
        return round(translation_size * (1 + ratio))


class CausalLMTrainingDataSets:
    """The experiment's sentence pairs as prompts and completions, with instruction data mixed into training."""

    def __init__(
        self,
        files: ExperimentFiles,
        prompts: LocalLLMPrompts,
        languages: ExperimentLanguages,
        length_filter: OverlengthRowFilter,
        instructions: InstructionMix,
    ) -> None:
        self._files = files
        self._prompts = prompts
        self._languages = languages
        self._length_filter = length_filter
        self._instructions = instructions

    def translations(self, encoder: CausalLMEncoder) -> Optional[Dataset]:
        data_set = self._read(self._files.train_source(), self._files.train_target())
        if data_set is None:
            return None
        src_lang, trg_lang = self._training_languages()

        def encode(example: dict, idx: int) -> dict:
            # idx locates this row in the example pool (same corpus, same order) for leave-one-out.
            prompt = self._prompts.build(example["src"], src_lang, trg_lang, example_pool_index=idx)
            return encoder.prompt_with_completion(prompt, example["trg"])

        data_set = data_set.map(encode, with_indices=True, remove_columns=data_set.column_names)
        return self._length_filter.apply(data_set, "train")

    def validation(self, encoder: CausalLMEncoder) -> Optional[Dataset]:
        data_set = self._read(self._files.validation_source(), self._files.validation_target())
        if data_set is None:
            return None
        src_lang, trg_lang = self._training_languages()

        def encode(example: dict, idx: int) -> dict:
            # Eval rows aren't in the example pool, so idx only rotates the prompt template here,
            # where for the training rows it also drives leave-one-out.
            prompt = self._prompts.build(example["src"], src_lang, trg_lang, rotation_index=idx)
            return encoder.prompt_with_completion(prompt, example["trg"])

        data_set = data_set.map(encode, with_indices=True, remove_columns=data_set.column_names)
        return self._length_filter.apply(data_set, "val")

    def mixed_with_instructions(
        self, translations: Optional[Dataset], encoder: CausalLMEncoder, training_args: TrainingArguments
    ) -> Optional[Union[Dataset, InterleavedTrainDataset]]:
        instructions = self._instructions.load()
        ratio = self._instructions.ratio()
        if instructions is None or translations is None or ratio <= 0:
            return translations
        instructions = instructions.map(
            lambda example: encoder.turns_with_completion(list(example["turns"]), example["output"]),
            remove_columns=instructions.column_names,
        )
        instructions = self._length_filter.apply(instructions, "instruction")
        return self._instructions.interleave(translations, instructions, training_args)

    def _read(self, source: Path, target: Path) -> Optional[Dataset]:
        pairs = read_parallel_text_pairs(source, target)
        if pairs is None:
            return None
        sources, targets = pairs
        return Dataset.from_dict({"src": sources, "trg": targets})

    def _training_languages(self) -> Tuple[Language, Language]:
        return (
            self._languages.of(self._languages.training_source_iso()),
            self._languages.of(self._languages.training_target_iso()),
        )
