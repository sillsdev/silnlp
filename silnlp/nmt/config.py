import logging
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import yaml
from machine.scripture import get_books

from ..common.corpus import (
    exclude_chapters,
    get_scripture_parallel_corpus,
    include_chapters,
    split_corpus,
    split_parallel_corpus,
)
from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import SentenceTranslationGroup
from ..common.utils import set_seed
from .checkpoints import Checkpoint, CheckpointDirectory, CheckpointType
from .corpora import (
    CorpusPair,
    DataFile,
    DataFileMapping,
    get_data_file_pairs,
    parse_corpus_pairs,
)
from .alignment_scores import AlignmentScores
from .basic_data_set_writer import BasicDataSetWriter
from .corpus_inventory import CorpusInventory
from .eval_data_set import EvalDataSet
from .eval_data_set_writer import ScriptureTestSetWriter, ScriptureValidationSetWriter
from .experiment_files import ExperimentFiles
from .sentence_noiser import SentenceNoiser
from .shared_test_set import SharedTestSet
from .terms import GlossLanguage, TermCategories
from .terms_data_set import TermsDataSet
from .terms_writer import TermsWriter
from .train_data_set import TrainDataSet
from .tokenization_statistics import TokenizationStatistics
from .tokenizer import Tokenizer

LOGGER = logging.getLogger((__package__ or "") + ".config")


@dataclass
class InferenceModelParams:
    checkpoint: Union[CheckpointType, str, int]
    src_lang: str
    trg_lang: str

    def __post_init__(self):
        if not isinstance(self.checkpoint, (CheckpointType, str, int)):
            raise ValueError("checkpoint must be a CheckpointType, string, or integer")
        if not isinstance(self.src_lang, str):
            raise ValueError("src_lang must be a string")
        if not isinstance(self.trg_lang, str):
            raise ValueError("trg_lang must be a string")


def warn_about_renamed_keys(config: dict, renamed: Dict[str, Dict[str, str]]) -> None:
    # Some config keys were renamed from huggingface 4.x to 5.x, so we need to warn team members if they have them in their config
    # rather than silently dropping the arguments. Can be removed once team is accustomed to 5.x.
    for section, keys in renamed.items():
        section_config = config.get(section)
        if not isinstance(section_config, dict):
            continue
        for old_name, new_name in keys.items():
            if old_name in section_config:
                LOGGER.warning(
                    f"{section}.{old_name} was renamed to {section}.{new_name} and is being ignored. "
                    f"Rename it to keep its effect.",
                )


def collect_training_args(
    config_root: dict,
    mapping: Dict[str, Set[str]],
    precision_args: Dict[str, Any],
    clearml_queue: Optional[str],
) -> Dict[str, Any]:
    """Collect the experiment config values named in ``mapping`` into a flat dict of
    TrainingArguments fields, with ``precision_args`` merged on top. Shared by the seq2seq and
    LLM models, which differ only in the args class, the mapping, and the precision flags."""
    args: Dict[str, Any] = {}
    for section, params in mapping.items():
        section_config: dict = config_root[section]
        for param in params:
            if param in section_config and section_config[param] is not None:
                args[param] = section_config[param]
    args.update(precision_args)
    args["report_to"] = "none" if clearml_queue is None else "all"
    return args


def write_effective_config(path: Path, config_root: dict, training_args: Any, mapping: Dict[str, Set[str]]) -> None:
    """Write the resolved experiment config, overlaying the effective values from ``training_args``
    (per ``mapping``) onto a copy of ``config_root``. Shared by the seq2seq and LLM models, which
    differ only in the args class and the mapping they pass in."""
    config = deepcopy(config_root)
    for section, params in mapping.items():
        section_config: dict = config[section]
        for param in params:
            value = getattr(training_args, param)
            if isinstance(value, Enum):
                value = value.value
            if value is None:
                section_config.pop(param, None)
            else:
                section_config[param] = value
    with path.open("w") as file:
        yaml.dump(config, file)


class NMTModel(ABC):
    def __init__(self, config: "Config") -> None:
        self._config = config
        self._checkpoints = CheckpointDirectory(config.model_dir)
        # The cached inference model is framework-specific (a torch model), so it is typed loosely
        # here to keep this base module free of transformers/torch imports.
        self._cached_inference_model: Optional[Any] = None
        self._inference_model_params: Optional[InferenceModelParams] = None

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def save_effective_config(self, path: Path) -> None:
        ...

    @abstractmethod
    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        ...

    @abstractmethod
    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        ...

    def resolve_checkpoint(self, ckpt: Union[CheckpointType, str, int]) -> Checkpoint:
        return self._checkpoints.resolve(ckpt)

    def has_best_checkpoint(self) -> bool:
        return self._checkpoints.has_best()

    def checkpoint_steps(self) -> List[int]:
        return self._checkpoints.steps()

    def has_been_trained(self) -> bool:
        return self._checkpoints.exists()

    def clear_cache(self) -> None:
        self._cached_inference_model = None
        self._inference_model_params = None

    def get_num_drafts(self) -> int:
        return self._config.infer.get("num_drafts", 1)


class Config(ABC):
    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        self.exp_dir = exp_dir
        self._environment = environment
        self.root = config

        data_config: dict = config["data"]
        self.corpus_pairs = parse_corpus_pairs(data_config.get("corpus_pairs", []), self._environment)
        self.inventory = CorpusInventory(
            self.corpus_pairs,
            include_glosses=data_config["terms"]["include_glosses"],
            environment=self._environment,
        )
        self.files = ExperimentFiles(exp_dir, self.inventory, multi_ref_eval=config["eval"]["multi_ref_eval"])

    @property
    def model(self) -> str:
        return self.root["model"]

    @property
    def model_dir(self) -> Path:
        return Path(self.train["output_dir"])

    @property
    def params(self) -> dict:
        return self.root["params"]

    @property
    def data(self) -> dict:
        return self.root["data"]

    @property
    def train(self) -> dict:
        return self.root["train"]

    @property
    def infer(self) -> dict:
        return self.root["infer"]

    @property
    def eval(self) -> dict:
        return self.root["eval"]

    @property
    def mirror(self) -> bool:
        return self.data["mirror"]

    @property
    def has_parent(self) -> bool:
        return "parent" in self.data

    def _disable_eval_if_no_val_split(self) -> None:
        """Turn off evaluation-related settings when there is no validation split. Shared by
        Config subclasses, which call this after merging their defaults."""
        if not self.inventory.has_validation_split():
            eval_config: dict = self.root["eval"]
            eval_config["eval_strategy"] = "no"
            eval_config["load_best_model_at_end"] = False
            eval_config["early_stopping"] = None
            eval_config["metric_for_best_model"] = None

    def set_seed(self) -> None:
        seed = self.data["seed"]
        set_seed(seed)

    def preprocess(self, stats: bool, force_align: bool = False) -> None:
        missing_files = self.inventory.missing_input_files()
        if len(missing_files) > 0:
            raise RuntimeError("These corpus files do not exist: " + ", ".join(str(f) for f in missing_files))

        if self.data["tokenize"]:
            self._build_vocabs(stats)
        tokenizer = self.create_tokenizer()
        self._build_corpora(tokenizer, stats, force_align)
        LOGGER.info("Preprocessing completed")

    @abstractmethod
    def create_model(
        self, mixed_precision: bool = True, num_devices: int = 1, clearml_queue: Optional[str] = None
    ) -> NMTModel:
        ...

    @abstractmethod
    def create_tokenizer(self) -> Tokenizer:
        ...

    def _build_corpora(self, tokenizer: Tokenizer, stats: bool, force_align: bool) -> int:
        self.files.delete_data_sets()

        train_count = 0
        terms_config = self.data["terms"]
        src_terms_files: List[Tuple[DataFile, List[str]]] = []
        trg_terms_files: List[Tuple[DataFile, List[str]]] = []
        for pair in self.corpus_pairs:
            if pair.is_scripture:
                train_count += self._write_scripture_data_sets(tokenizer, pair, force_align)
            else:
                train_count += BasicDataSetWriter(
                    self.files, self.inventory, tokenizer, mirror=self.mirror
                ).write(pair)

            if terms_config["dictionary"] or terms_config["train"]:
                for file in pair.src_terms_files:
                    src_terms_files.append((file, pair.tags))
                for file in pair.trg_terms_files:
                    trg_terms_files.append((file, pair.tags))

        terms_train_count = 0
        if terms_config["train"]:
            terms_train_count = TermsWriter(
                TermsDataSet(self.files, tokenizer, mirror=self.mirror),
                self._term_categories(),
                self._gloss_language(),
                self._term_filter_books(),
                self._environment,
            ).write(src_terms_files, trg_terms_files)
            LOGGER.info(f"terms train size: {terms_train_count}")
        train_count += terms_train_count

        dict_count = 0
        if terms_config["dictionary"]:
            dict_count = self._write_dictionary(tokenizer, src_terms_files, trg_terms_files)
            LOGGER.info(f"dictionary size: {dict_count}")

        if stats and self.data["tokenize"]:
            TokenizationStatistics(self.files).write()

        return train_count

    def _write_scripture_data_sets(
        self,
        tokenizer: Tokenizer,
        pair: CorpusPair,
        force_align: bool,
    ) -> int:
        src_corpora_str = ", ".join(sf.path.stem for sf in pair.src_files)
        if len(pair.src_files) > 1:
            src_corpora_str = f"[{src_corpora_str}]"
        trg_corpora_str = ", ".join(tf.path.stem for tf in pair.trg_files)
        if len(pair.trg_files) > 1:
            trg_corpora_str = f"[{trg_corpora_str}]"
        LOGGER.info(f"Preprocessing {src_corpora_str} -> {trg_corpora_str}")
        test_size = pair.size if pair.test_size is None else pair.test_size
        val_size = pair.size if pair.val_size is None else pair.val_size
        train_size = pair.size

        test_indices: Optional[Set[int]] = None
        val_indices: Optional[Set[int]] = None
        train_indices: Optional[Set[int]] = None

        validation = EvalDataSet()
        test = EvalDataSet()
        train_data_set = TrainDataSet(
            self.files, tokenizer, mixed_source=pair.mapping == DataFileMapping.MIXED_SRC
        )

        alignment_scores = AlignmentScores(self.exp_dir, self.data["aligner"], force=force_align)

        if pair.use_test_set_from != "":
            test = EvalDataSet(
                SharedTestSet(
                    pair.use_test_set_from, self.inventory, self._environment, self.exp_dir
                ).indices_by_iso_pair()
            )

        for src_file, trg_file in get_data_file_pairs(pair):
            train_data_set.note_language(src_file.project, src_file.iso)
            train_data_set.note_language(trg_file.project, trg_file.iso)
            corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path, environment=self._environment)
            if len(pair.src_noise) > 0:
                noiser = SentenceNoiser(pair.src_noise)
                corpus["source"] = [noiser.apply(x) for x in corpus["source"]]

            if len(pair.corpus_books) > 0:
                cur_train = include_chapters(corpus, pair.corpus_books)
                if len(pair.test_books) > 0:
                    cur_train = exclude_chapters(cur_train, pair.test_books)
            elif len(pair.test_books) > 0:
                cur_train = exclude_chapters(corpus, pair.test_books)
            else:
                cur_train = corpus
            corpus_count = len(cur_train)

            if pair.is_train and pair.score_threshold > 0:
                alignment_scores.add_to(
                    cur_train,
                    f"{src_file.iso}-{src_file.project}",
                    f"{trg_file.iso}-{trg_file.project}",
                )

            if pair.is_test:
                if len(pair.test_books) > 0:
                    cur_test = include_chapters(corpus, pair.test_books)
                    if test_indices is None:
                        test_indices = cur_test.index

                if pair.disjoint_test and test_indices is None:
                    indices: Set[int] = set(cur_train.index)
                    if pair.disjoint_val and val_indices is not None:
                        indices.difference_update(val_indices)
                    split_size = test_size
                    if isinstance(split_size, float):
                        split_size = int(split_size if split_size > 1 else corpus_count * split_size)
                    test_indices = set(random.sample(indices, min(split_size, len(indices))))

                if len(pair.test_books) > 0:
                    if test_size > 0:
                        _, cur_test = split_parallel_corpus(
                            cur_test,
                            test_size,
                            test.indices_for(src_file.iso, trg_file.iso, default=test_indices),
                        )
                else:
                    cur_train, cur_test = split_parallel_corpus(
                        cur_train, test_size, test.indices_for(src_file.iso, trg_file.iso, default=test_indices)
                    )

                cur_test.drop("score", axis=1, inplace=True, errors="ignore")
                if src_file.include_test:
                    test.add(src_file.iso, trg_file.iso, trg_file.project, pair.tags, cur_test)

            if pair.is_train and pair.score_threshold > 0:
                cur_train = alignment_scores.filter(cur_train, pair.score_threshold)

            if pair.is_val:
                if pair.disjoint_val and val_indices is None:
                    indices = set(cur_train.index)
                    if pair.disjoint_test and test_indices is not None:
                        indices.difference_update(test_indices)
                    split_size = val_size
                    if isinstance(split_size, float):
                        split_size = int(split_size if split_size > 1 else corpus_count * split_size)
                    val_indices = set(random.sample(indices, min(split_size, len(indices))))

                cur_train, cur_val = split_parallel_corpus(
                    cur_train, val_size, validation.indices_for(src_file.iso, trg_file.iso, default=val_indices)
                )

                validation.add(src_file.iso, trg_file.iso, trg_file.project, pair.tags, cur_val)

            if pair.is_train:
                cur_train["source_lang"] = src_file.iso
                cur_train["target_lang"] = trg_file.iso

                train_indices = split_corpus(set(cur_train.index), train_size)
                _, cur_train = split_parallel_corpus(cur_train, train_size, train_indices)

                if self.mirror:
                    train_data_set.add(
                        trg_file.project,
                        src_file.project,
                        pair.tags,
                        cur_train.rename(
                            columns={
                                "source": "target",
                                "target": "source",
                                "source_lang": "target_lang",
                                "target_lang": "source_lang",
                            }
                        ),
                    )

                train_data_set.add(src_file.project, trg_file.project, pair.tags, cur_train)

        train_count = train_data_set.write()

        val_count = ScriptureValidationSetWriter(
            self.files, tokenizer, multi_ref_eval=self.root["eval"]["multi_ref_eval"]
        ).write(validation)
        test_count = ScriptureTestSetWriter(self.files, self.inventory, tokenizer).write(test)
        LOGGER.info(f"train size: {train_count}," f" val size: {val_count}," f" test size: {test_count},")
        return train_count

    def _term_filter_books(self) -> Optional[Set[int]]:
        if "filter_books" not in self.data["terms"]:
            return None
        return get_books(self.data["terms"]["filter_books"])

    def _term_categories(self) -> TermCategories:
        return TermCategories(self.data["terms"]["categories"])

    def _gloss_language(self) -> GlossLanguage:
        return GlossLanguage(
            self.data["terms"]["include_glosses"], self.inventory.source_isos(), self.inventory.target_isos()
        )

    @abstractmethod
    def _build_vocabs(self, stats: bool = False) -> None:
        ...

    @abstractmethod
    def _write_dictionary(
        self,
        tokenizer: Tokenizer,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        ...
