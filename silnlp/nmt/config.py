import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Set, Union

from machine.scripture import get_books

from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import SentenceTranslationGroup
from ..common.utils import set_seed
from .alignment_scores import AlignmentScores
from .basic_data_set_writer import BasicDataSetWriter
from .checkpoints import Checkpoint, CheckpointDirectory, CheckpointType
from .corpora import parse_corpus_pairs
from .corpus_inventory import CorpusInventory
from .dictionary_writer import DictionaryWriter
from .experiment_data_set_writer import ExperimentDataSetWriter, ScriptureDataSetWriterFactory, TermsSettings
from .experiment_files import ExperimentFiles
from .experiment_settings import EvaluationSettings
from .terms import GlossLanguage, TermCategories
from .terms_data_set import TermsDataSet
from .terms_writer import TermsWriter
from .tokenizer import Tokenizer
from .vocabulary_builder import VocabularyBuilder

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


class NMTModel(ABC):
    def __init__(self, checkpoints: CheckpointDirectory, num_drafts: int) -> None:
        self._checkpoints = checkpoints
        self._num_drafts = num_drafts
        # The cached inference model is framework-specific (a torch model), so it is typed loosely
        # here to keep this base module free of transformers/torch imports.
        self._cached_inference_model: Optional[Any] = None
        self._inference_model_params: Optional[InferenceModelParams] = None

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def save_effective_config(self, path: Path) -> None: ...

    @abstractmethod
    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None: ...

    @abstractmethod
    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Generator[SentenceTranslationGroup, None, None]: ...

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
        return self._num_drafts


class Config(ABC):
    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        self.exp_dir = exp_dir
        self._environment = environment
        self.root = config

        data_config: dict = config["data"]
        self.corpus_pairs = parse_corpus_pairs(data_config.get("corpus_pairs", []), self._environment)
        self.corpus_inventory = CorpusInventory(
            self.corpus_pairs,
            include_glosses=data_config["terms"]["include_glosses"],
            environment=self._environment,
        )
        self.files = ExperimentFiles(exp_dir, self.corpus_inventory, multi_ref_eval=config["eval"]["multi_ref_eval"])

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
        EvaluationSettings(self.root["eval"]).disable_unless(self.corpus_inventory.has_validation_split())

    def set_seed(self) -> None:
        seed = self.data["seed"]
        set_seed(seed)

    @abstractmethod
    def create_model(
        self, mixed_precision: bool = True, num_devices: int = 1, clearml_queue: Optional[str] = None
    ) -> NMTModel: ...

    @abstractmethod
    def create_tokenizer(self) -> Tokenizer: ...

    def create_data_set_writer(self, force_align: bool) -> ExperimentDataSetWriter:
        tokenizer = self.create_tokenizer()
        return ExperimentDataSetWriter(
            self.corpus_pairs,
            self.files,
            ScriptureDataSetWriterFactory(
                self.files,
                self.corpus_inventory,
                tokenizer,
                AlignmentScores(self.exp_dir, self.data["aligner"], force=force_align),
                self._environment,
                self.exp_dir,
                mirror=self.mirror,
                multi_ref_eval=self.root["eval"]["multi_ref_eval"],
            ),
            BasicDataSetWriter(self.files, self.corpus_inventory, tokenizer, mirror=self.mirror),
            TermsWriter(
                TermsDataSet(self.files, tokenizer, mirror=self.mirror),
                self._term_categories(),
                self._gloss_language(),
                self._term_filter_books(),
                self._environment,
            ),
            self._dictionary_writer(tokenizer),
            TermsSettings(self.data["terms"]),
            tokenize=self.data["tokenize"],
        )

    def _term_filter_books(self) -> Optional[Set[int]]:
        if "filter_books" not in self.data["terms"]:
            return None
        return get_books(self.data["terms"]["filter_books"])

    def _term_categories(self) -> TermCategories:
        return TermCategories(self.data["terms"]["categories"])

    def _gloss_language(self) -> GlossLanguage:
        return GlossLanguage(
            self.data["terms"]["include_glosses"],
            self.corpus_inventory.source_isos(),
            self.corpus_inventory.target_isos(),
        )

    @abstractmethod
    def create_vocabulary_builder(self) -> VocabularyBuilder: ...

    @abstractmethod
    def _dictionary_writer(self, tokenizer: Tokenizer) -> DictionaryWriter: ...
