import itertools
import logging
import random
import re
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum, Flag, auto
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple, Union, cast

import pandas as pd
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, get_books, get_chapters
from machine.tokenization import LatinWordTokenizer
from tqdm import tqdm

from ..alignment.config import get_aligner_name
from ..alignment.utils import add_alignment_scores
from ..common.corpus import (
    Term,
    exclude_chapters,
    filter_parallel_corpus,
    get_mt_corpus_path,
    get_scripture_parallel_corpus,
    get_terms,
    get_terms_corpus,
    get_terms_data_frame,
    get_terms_glosses_path,
    get_terms_list,
    get_terms_renderings_path,
    include_chapters,
    load_corpus,
    split_corpus,
    split_parallel_corpus,
    write_corpus,
)
from ..common.environment import SIL_NLP_ENV
from ..common.translator import TranslationGroup
from ..common.utils import NoiseMethod, Side, create_noise_methods, get_mt_exp_dir, is_set, set_seed
from .augment import AugmentMethod, create_augment_methods
from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__package__ + ".config")

BASIC_DATA_PROJECT = "BASIC"

ALIGNMENT_SCORES_FILE = re.compile(r"([a-z]{2,3}-.+)_([a-z]{2,3}-.+)")


class CheckpointType(Enum):
    LAST = auto()
    BEST = auto()
    AVERAGE = auto()
    OTHER = auto()


class DataFileType(Flag):
    NONE = 0
    TRAIN = auto()
    TEST = auto()
    VAL = auto()
    DICT = auto()


class DataFileMapping(Enum):
    ONE_TO_ONE = auto()
    MIXED_SRC = auto()
    MANY_TO_MANY = auto()


@dataclass
class DataFile:
    path: Path
    iso: str = field(init=False)
    project: str = field(init=False)
    include_test: bool = True

    def __post_init__(self):
        file_name = self.path.stem
        parts = file_name.split("-")
        if len(parts) < 2:
            raise RuntimeError(f"The filename {file_name} needs to be of the format <iso>-<project>")
        self.iso = parts[0]
        self.project = (
            parts[1] if str(self.path.parent).startswith(str(SIL_NLP_ENV.mt_scripture_dir)) else BASIC_DATA_PROJECT
        )

    @property
    def is_scripture(self) -> bool:
        return self.project != BASIC_DATA_PROJECT


@dataclass
class CorpusPair:
    src_files: List[DataFile]
    trg_files: List[DataFile]
    type: DataFileType
    src_noise: List[NoiseMethod]
    augmentations: List[AugmentMethod]
    tags: List[str]
    size: Union[float, int]
    test_size: Optional[Union[float, int]]
    val_size: Optional[Union[float, int]]
    disjoint_test: bool
    disjoint_val: bool
    score_threshold: float
    corpus_books: Dict[int, List[int]]
    test_books: Dict[int, List[int]]
    use_test_set_from: str
    src_terms_files: List[DataFile]
    trg_terms_files: List[DataFile]
    is_lexical_data: bool
    mapping: DataFileMapping

    @property
    def is_train(self) -> bool:
        return is_set(self.type, DataFileType.TRAIN)

    @property
    def is_test(self) -> bool:
        return is_set(self.type, DataFileType.TEST)

    @property
    def is_val(self) -> bool:
        return is_set(self.type, DataFileType.VAL)

    @property
    def is_dictionary(self) -> bool:
        return is_set(self.type, DataFileType.DICT)

    @property
    def is_scripture(self) -> bool:
        return self.src_files[0].is_scripture


@dataclass
class IsoPairInfo:
    test_projects: Set[str] = field(default_factory=set)
    val_projects: Set[str] = field(default_factory=set)
    has_basic_test_data: bool = False

    @property
    def has_multiple_test_projects(self) -> bool:
        return len(self.test_projects) > 1

    @property
    def has_test_data(self) -> bool:
        return len(self.test_projects) > 0 or self.has_basic_test_data


def parse_corpus_pairs(corpus_pairs: List[dict]) -> List[CorpusPair]:
    pairs: List[CorpusPair] = []
    for pair in corpus_pairs:
        if "type" not in pair:
            pair["type"] = ["train", "test", "val"]
        type_strs: Union[str, List[str]] = pair["type"]
        if isinstance(type_strs, str):
            type_strs = type_strs.split(",")
        type = DataFileType.NONE
        for type_str in type_strs:
            type_str = type_str.strip().lower()
            if type_str == "train":
                type |= DataFileType.TRAIN
            elif type_str == "test":
                type |= DataFileType.TEST
            elif type_str == "val" or type_str == "validation":
                type |= DataFileType.VAL
            elif type_str == "dict" or type_str == "dictionary":
                type |= DataFileType.DICT

        src: Union[str, List[Union[dict, str]]] = pair["src"]
        src_files = []
        if isinstance(src, str):
            src = src.split(",")
        for file in src:
            if isinstance(file, str):
                src_files.append(DataFile(get_mt_corpus_path(file.strip())))
            else:
                src_files.append(DataFile(get_mt_corpus_path(file.pop("name"))))
                for k, v in file.items():
                    setattr(src_files[-1], k, v)
        trg: Union[str, List[Union[dict, str]]] = pair["trg"]
        trg_files = []
        if isinstance(trg, str):
            trg = trg.split(",")
        for file in trg:
            if isinstance(file, str):
                trg_files.append(DataFile(get_mt_corpus_path(file.strip())))
            else:
                trg_files.append(DataFile(get_mt_corpus_path(file.pop("name"))))
                for k, v in file.items():
                    setattr(trg_files[-1], k, v)
        is_scripture = src_files[0].is_scripture
        if not all(df.is_scripture == is_scripture for df in (src_files + trg_files)):
            raise RuntimeError("All corpora in a corpus pair must contain the same type of data.")

        tags: Union[str, List[str]] = pair.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",")]

        src_noise = create_noise_methods(pair.get("src_noise", []))
        augmentations = create_augment_methods(pair.get("augment", []))

        if "size" not in pair:
            pair["size"] = 1.0
        size: Union[float, int] = pair["size"]
        if "test_size" not in pair and is_set(type, DataFileType.TRAIN | DataFileType.TEST):
            pair["test_size"] = 0 if "test_books" in pair else 250
        test_size: Optional[Union[float, int]] = pair.get("test_size")
        if "val_size" not in pair and is_set(type, DataFileType.TRAIN | DataFileType.VAL):
            pair["val_size"] = 250
        val_size: Optional[Union[float, int]] = pair.get("val_size")

        if "disjoint_test" not in pair:
            pair["disjoint_test"] = True
        disjoint_test: bool = pair["disjoint_test"]
        if "disjoint_val" not in pair:
            pair["disjoint_val"] = True
        disjoint_val: bool = pair["disjoint_val"]
        score_threshold: float = pair.get("score_threshold", 0.0)
        corpus_books = get_chapters(pair.get("corpus_books", []))
        test_books = get_chapters(pair.get("test_books", []))
        use_test_set_from: str = pair.get("use_test_set_from", "")

        src_terms_files = get_terms_files(src_files) if is_set(type, DataFileType.TRAIN) else []
        trg_terms_files = get_terms_files(trg_files) if is_set(type, DataFileType.TRAIN) else []

        if "lexical" not in pair:
            pair["lexical"] = is_set(type, DataFileType.DICT)
        is_lexical_data: bool = pair["lexical"]

        if "mapping" not in pair:
            pair["mapping"] = DataFileMapping.ONE_TO_ONE.name.lower()
        mapping = DataFileMapping[pair["mapping"].upper()]
        if not is_scripture and mapping != DataFileMapping.ONE_TO_ONE:
            raise RuntimeError("Basic corpus pairs only support one-to-one mapping.")
        if mapping == DataFileMapping.ONE_TO_ONE and len(src_files) != len(trg_files):
            raise RuntimeError(
                "A corpus pair with one-to-one mapping must contain the same number of source and target corpora."
            )

        pairs.append(
            CorpusPair(
                src_files,
                trg_files,
                type,
                src_noise,
                augmentations,
                tags,
                size,
                test_size,
                val_size,
                disjoint_test,
                disjoint_val,
                score_threshold,
                corpus_books,
                test_books,
                use_test_set_from,
                src_terms_files,
                trg_terms_files,
                is_lexical_data,
                mapping,
            )
        )
    return pairs


def get_terms_files(files: List[DataFile]) -> List[DataFile]:
    terms_files: List[DataFile] = []
    for file in files:
        terms_path = get_terms_renderings_path(file.iso, file.project)
        if terms_path is None:
            continue
        terms_files.append(DataFile(terms_path))
    return terms_files


def get_terms_glosses_file_paths(terms_files: List[DataFile]) -> Set[Path]:
    glosses_file_paths: Set[Path] = set()
    for terms_file in terms_files:
        list_name = get_terms_list(terms_file.path)
        glosses_path = get_terms_glosses_path(list_name)
        if glosses_path.is_file():
            glosses_file_paths.add(glosses_path)
    return glosses_file_paths


def get_parallel_corpus_size(src_file_path: Path, trg_file_path: Path) -> int:
    count = 0
    with src_file_path.open("r", encoding="utf-8") as src_file, trg_file_path.open("r", encoding="utf-8") as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if len(src_line) > 0 and len(trg_line) > 0:
                count += 1
    return count


def get_data_file_pairs(corpus_pair: CorpusPair) -> Iterable[Tuple[DataFile, DataFile]]:
    if corpus_pair.mapping == DataFileMapping.ONE_TO_ONE:
        for file_pair in zip(corpus_pair.src_files, corpus_pair.trg_files):
            yield file_pair
    else:
        for src_file in corpus_pair.src_files:
            for trg_file in corpus_pair.trg_files:
                yield (src_file, trg_file)


class NMTModel(ABC):
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
        vref_paths: Optional[List[Path]] = None,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None: ...

    @abstractmethod
    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        vrefs: Optional[Iterable[VerseRef]] = None,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Iterable[TranslationGroup]: ...

    @abstractmethod
    def get_checkpoint_path(self, ckpt: Union[CheckpointType, str, int]) -> Tuple[Path, int]: ...


class Config(ABC):
    def __init__(self, exp_dir: Path, config: dict) -> None:
        self.exp_dir = exp_dir
        self.root = config

        data_config: dict = config["data"]
        self.corpus_pairs = parse_corpus_pairs(data_config.get("corpus_pairs", []))

        terms_config: dict = data_config["terms"]
        self.src_isos: Set[str] = set()
        self.val_src_isos: Set[str] = set()
        self.test_src_isos: Set[str] = set()
        self.trg_isos: Set[str] = set()
        self.val_trg_isos: Set[str] = set()
        self.test_trg_isos: Set[str] = set()
        self.src_file_paths: Set[Path] = set()
        self.trg_file_paths: Set[Path] = set()
        self._tags: Set[str] = set()
        self.has_scripture_data = False
        self._iso_pairs: Dict[Tuple[str, str], IsoPairInfo] = {}
        self.src_projects: Set[str] = set()
        self.trg_projects: Set[str] = set()
        for corpus_pair in self.corpus_pairs:
            pair_src_isos = {sf.iso for sf in corpus_pair.src_files}
            pair_trg_isos = {tf.iso for tf in corpus_pair.trg_files}
            self.src_isos.update(pair_src_isos)
            self.trg_isos.update(pair_trg_isos)
            if corpus_pair.is_val:
                self.val_src_isos.update(pair_src_isos)
                self.val_trg_isos.update(pair_trg_isos)
            if corpus_pair.is_test:
                self.test_src_isos.update(pair_src_isos)
                self.test_trg_isos.update(pair_trg_isos)
            self.src_file_paths.update(sf.path for sf in corpus_pair.src_files)
            self.trg_file_paths.update(tf.path for tf in corpus_pair.trg_files)
            if corpus_pair.is_scripture:
                self.has_scripture_data = True
                self.src_file_paths.update(sf.path for sf in corpus_pair.src_terms_files)
                self.trg_file_paths.update(tf.path for tf in corpus_pair.trg_terms_files)
                self.src_projects.update(sf.project for sf in corpus_pair.src_files)
                self.trg_projects.update(sf.project for sf in corpus_pair.trg_files)
                if terms_config["include_glosses"]:
                    for gloss_iso in ["fr", "en", "id", "es"]:
                        if gloss_iso in pair_src_isos or gloss_iso == terms_config["include_glosses"]:
                            self.src_file_paths.update(get_terms_glosses_file_paths(corpus_pair.src_terms_files))
                        if gloss_iso in pair_trg_isos:
                            self.trg_file_paths.update(get_terms_glosses_file_paths(corpus_pair.trg_terms_files))
            self._tags.update(f"<{tag}>" for tag in corpus_pair.tags)

            for src_file in corpus_pair.src_files:
                for trg_file in corpus_pair.trg_files:
                    iso_pair = self._iso_pairs.get((src_file.iso, trg_file.iso))
                    if iso_pair is None:
                        iso_pair = IsoPairInfo()
                        self._iso_pairs[(src_file.iso, trg_file.iso)] = iso_pair
                    if corpus_pair.is_scripture:
                        if corpus_pair.is_test:
                            iso_pair.test_projects.add(trg_file.project)
                        if corpus_pair.is_val:
                            iso_pair.val_projects.add(trg_file.project)
                    elif corpus_pair.is_test:
                        iso_pair.has_basic_test_data = True

        self._multiple_test_iso_pairs = sum(1 for iso_pair in self._iso_pairs.values() if iso_pair.has_test_data) > 1

    @property
    def default_test_src_iso(self) -> str:
        if len(self.test_src_isos) == 0:
            return ""
        return next(iter(self.test_src_isos))

    @property
    def default_val_src_iso(self) -> str:
        if len(self.val_src_isos) == 0:
            return ""
        return next(iter(self.val_src_isos))

    @property
    def default_test_trg_iso(self) -> str:
        if len(self.test_trg_isos) == 0:
            return ""
        return next(iter(self.test_trg_isos))

    @property
    def default_val_trg_iso(self) -> str:
        if len(self.val_trg_isos) == 0:
            return ""
        return next(iter(self.val_trg_isos))

    @property
    def model(self) -> str:
        return self.root["model"]

    @property
    @abstractmethod
    def model_dir(self) -> Path: ...

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
    def share_vocab(self) -> bool:
        return self.data["share_vocab"]

    @property
    def stats_max_size(self) -> int:
        return self.data["stats_max_size"]

    @property
    def has_parent(self) -> bool:
        return "parent" in self.data

    @property
    def has_val_split(self) -> bool:
        return any(
            pair.is_val and (pair.size if pair.val_size is None else pair.val_size) > 0 for pair in self.corpus_pairs
        )

    @property
    @abstractmethod
    def has_best_checkpoint(self) -> bool: ...

    def set_seed(self) -> None:
        seed = self.data["seed"]
        set_seed(seed)

    def preprocess(self, stats: bool, force_align: bool = False) -> None:
        # confirm that input file paths exist
        for file in self.src_file_paths | self.trg_file_paths:
            if not file.is_file():
                LOGGER.error(f"The source file {str(file)} does not exist.")
                return

        if self.data["tokenize"]:
            self._build_vocabs(stats)
        tokenizer = self.create_tokenizer()
        self._build_corpora(tokenizer, stats, force_align)
        LOGGER.info("Preprocessing completed")

    @abstractmethod
    def create_model(self, mixed_precision: bool = True, num_devices: int = 1) -> NMTModel: ...

    @abstractmethod
    def create_tokenizer(self) -> Tokenizer: ...

    def is_train_project(self, ref_file_path: Path) -> bool:
        trg_iso, trg_project = self._parse_ref_file_path(ref_file_path)
        for pair in self.corpus_pairs:
            if not pair.is_train:
                continue
            for df in pair.src_files + pair.trg_files:
                if df.iso == trg_iso and df.project == trg_project:
                    return True
        return False

    def is_ref_project(self, ref_projects: Set[str], ref_file_path: Path) -> bool:
        _, trg_project = self._parse_ref_file_path(ref_file_path)
        return trg_project in ref_projects

    def _parse_ref_file_path(self, ref_file_path: Path) -> Tuple[str, str]:
        parts = ref_file_path.name.split(".")
        if len(parts) == 5:
            return self.default_test_trg_iso, parts[3]
        return parts[2], parts[5]

    def _build_corpora(self, tokenizer: Tokenizer, stats: bool, force_align: bool) -> int:
        self._delete_files("train.*.txt")
        self._delete_files("val.*.txt")
        self._delete_files("test.*.txt")
        self._delete_files("dict.*.txt")

        train_count = 0
        terms_config = self.data["terms"]
        src_terms_files: List[Tuple[DataFile, str]] = []
        trg_terms_files: List[Tuple[DataFile, str]] = []
        for pair in self.corpus_pairs:
            if pair.is_scripture:
                train_count += self._write_scripture_data_sets(tokenizer, pair, force_align)
            else:
                train_count += self._write_basic_data_sets(tokenizer, pair)

            if terms_config["dictionary"] or terms_config["train"]:
                for file in pair.src_terms_files:
                    src_terms_files.append((file, self._get_tags_str(pair.tags)))
                for file in pair.trg_terms_files:
                    trg_terms_files.append((file, self._get_tags_str(pair.tags)))

        terms_train_count = 0
        if terms_config["train"]:
            terms_train_count = self._write_terms(tokenizer, src_terms_files, trg_terms_files)
            LOGGER.info(f"terms train size: {terms_train_count}")
        train_count += terms_train_count

        dict_count = 0
        if terms_config["dictionary"]:
            dict_count = self._write_dictionary(tokenizer, src_terms_files, trg_terms_files)
            LOGGER.info(f"dictionary size: {dict_count}")

        if stats and self.data["tokenize"]:
            self._calculate_tokenization_stats()

        return train_count

    def _calculate_tokenization_stats(self) -> None:
        LOGGER.info("Calculating tokenization statistics")

        stats_path = self.exp_dir / "tokenization_stats.csv"
        if stats_path.is_file():
            existing_stats = pd.read_csv(stats_path, header=[0, 1])
        else:
            existing_stats = pd.DataFrame({(" ", "Translation Side"): ["Source", "Target"]})

        src_tokens_per_verse, src_chars_per_token = [], []
        for src_tok_file in self.exp_dir.glob("*.src.txt"):
            with open(self.exp_dir / src_tok_file, "r+", encoding="utf-8") as f:
                for line in f:
                    src_tokens_per_verse.append(len(line.split()))
                    src_chars_per_token.extend([len(token) for token in line.split()])

        trg_tokens_per_verse, trg_chars_per_token = [], []
        for trg_tok_file in self.exp_dir.glob("*.trg.txt"):
            with open(self.exp_dir / trg_tok_file, "r+", encoding="utf-8") as f:
                for line in f:
                    trg_tokens_per_verse.append(len(line.split()))
                    trg_chars_per_token.extend([len(token) for token in line.split()])

        src_chars_per_verse, src_words_per_verse, src_chars_per_word = [], [], []
        for src_detok_file in self.exp_dir.glob("*.src.detok.txt"):
            with open(self.exp_dir / src_detok_file, "r+", encoding="utf-8") as f:
                for line in f:
                    src_chars_per_verse.append(len(line))
                    word_line = " ".join(LatinWordTokenizer().tokenize(line))
                    src_words_per_verse.append(len(word_line.split()))
                    src_chars_per_word.extend([len(word) for word in word_line.split()])

        trg_chars_per_verse, trg_words_per_verse, trg_chars_per_word = [], [], []
        for trg_detok_file in self.exp_dir.glob("*.trg.detok.txt"):
            with open(self.exp_dir / trg_detok_file, "r+", encoding="utf-8") as f:
                for line in f:
                    trg_chars_per_verse.append(len(line))
                    word_line = " ".join(LatinWordTokenizer().tokenize(line))
                    trg_words_per_verse.append(len(word_line.split()))
                    trg_chars_per_word.extend([len(word) for word in word_line.split()])

        def distribution_df(
            top_header: str,
            src_data: List[int],
            trg_data: List[int],
        ) -> pd.DataFrame:
            columns = pd.MultiIndex.from_product([[top_header], ["Min", "Max", "Median", "Mean", "Std Dev"]])
            distribution_data = [
                [
                    min(src_data),
                    max(src_data),
                    median(src_data),
                    Decimal(str(mean(src_data))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                    Decimal(str(stdev(src_data))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                ],
                [
                    min(trg_data),
                    max(trg_data),
                    median(trg_data),
                    Decimal(str(mean(trg_data))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                    Decimal(str(stdev(trg_data))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                ],
            ]
            return pd.DataFrame(distribution_data, columns=columns)

        top_header = "Tokens/Verse"
        tokens_verse_df = distribution_df(top_header, src_tokens_per_verse, trg_tokens_per_verse)
        num_verses_200_df = pd.DataFrame(
            {
                (top_header, "Num Verses >= 200 Tokens"): [
                    sum(seg_length >= 200 for seg_length in src_tokens_per_verse),
                    sum(seg_length >= 200 for seg_length in trg_tokens_per_verse),
                ]
            }
        )
        tokens_verse_df = pd.concat([tokens_verse_df, num_verses_200_df], axis=1)
        existing_stats = pd.concat([existing_stats, tokens_verse_df], axis=1)

        top_header = "Characters/Verse"
        chars_verse_df = distribution_df(top_header, src_chars_per_verse, trg_chars_per_verse)
        existing_stats = pd.concat([existing_stats, chars_verse_df], axis=1)

        top_header = "Characters/Token"
        chars_token_df = distribution_df(top_header, src_chars_per_token, trg_chars_per_token)
        existing_stats = pd.concat([existing_stats, chars_token_df], axis=1)

        top_header = "Words/Verse"
        words_verse_df = distribution_df(top_header, src_words_per_verse, trg_words_per_verse)
        existing_stats = pd.concat([existing_stats, words_verse_df], axis=1)

        top_header = "Characters/Word"
        chars_word_df = distribution_df(top_header, src_chars_per_word, trg_chars_per_word)
        existing_stats = pd.concat([existing_stats, chars_word_df], axis=1)

        existing_stats.to_csv(stats_path, index=False)
        existing_stats.to_excel(stats_path.with_suffix(".xlsx"))

    def _delete_files(self, pattern: str) -> None:
        for old_file_path in self.exp_dir.glob(pattern):
            old_file_path.unlink()

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

        train: Optional[pd.DataFrame] = None
        val: Dict[Tuple[str, str], pd.DataFrame] = {}
        test: Dict[Tuple[str, str], pd.DataFrame] = {}
        pair_val_indices: Dict[Tuple[str, str], Set[int]] = {}
        pair_test_indices: Dict[Tuple[str, str], Set[int]] = {}
        project_isos: Dict[str, str] = {}

        tags_str = self._get_tags_str(pair.tags)

        if pair.use_test_set_from != "":
            self._populate_pair_test_indices(pair.use_test_set_from, pair_test_indices)

        for src_file, trg_file in get_data_file_pairs(pair):
            project_isos[src_file.project] = src_file.iso
            project_isos[trg_file.project] = trg_file.iso
            corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path)
            if len(pair.src_noise) > 0:
                corpus["source"] = [self._noise(pair.src_noise, x) for x in corpus["source"]]

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
                pair_align_path = (
                    self.exp_dir / f"{src_file.iso}-{src_file.project}_{trg_file.iso}-{trg_file.project}.csv"
                )
                if pair_align_path.is_file() and not force_align:
                    LOGGER.info(f"Using pre-existing alignment scores from {pair_align_path}")
                    pair_scores = pd.read_csv(pair_align_path)
                    pair_scores["idx"] = cur_train.index
                    pair_scores.set_index("idx", inplace=True)
                    cur_train["score"] = pair_scores["score"]
                else:
                    aligner_id = self.data["aligner"]
                    LOGGER.info(f"Computing alignment scores using {get_aligner_name(aligner_id)}")
                    add_alignment_scores(cur_train, aligner_id)
                    cur_train.to_csv(pair_align_path, index=False)

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
                            pair_test_indices.get((src_file.iso, trg_file.iso), test_indices),
                        )
                else:
                    cur_train, cur_test = split_parallel_corpus(
                        cur_train, test_size, pair_test_indices.get((src_file.iso, trg_file.iso), test_indices)
                    )

                cur_test.drop("score", axis=1, inplace=True, errors="ignore")
                if src_file.include_test:
                    self._add_to_eval_data_set(
                        src_file.iso,
                        trg_file.iso,
                        trg_file.project,
                        tags_str,
                        test,
                        pair_test_indices,
                        cur_test,
                    )

            if pair.is_train and pair.score_threshold > 0:
                unfiltered_count = len(cur_train)
                cur_train = filter_parallel_corpus(cur_train, pair.score_threshold)
                cur_train = cur_train.drop("score", axis=1, errors="ignore")
                LOGGER.info(
                    f"Filtered out {unfiltered_count - len(cur_train)} verses pairs with alignment below {pair.score_threshold}."
                )

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
                    cur_train, val_size, pair_val_indices.get((src_file.iso, trg_file.iso), val_indices)
                )

                self._add_to_eval_data_set(
                    src_file.iso, trg_file.iso, trg_file.project, tags_str, val, pair_val_indices, cur_val
                )

            if pair.is_train:
                cur_train["source_lang"] = src_file.iso
                cur_train["target_lang"] = trg_file.iso

                train_indices = split_corpus(set(cur_train.index), train_size)
                _, cur_train = split_parallel_corpus(cur_train, train_size, train_indices)

                if self.mirror:
                    mirror_cur_train = cur_train.rename(
                        columns={
                            "source": "target",
                            "target": "source",
                            "source_lang": "target_lang",
                            "target_lang": "source_lang",
                        }
                    )
                    train = self._add_to_train_data_set(
                        trg_file.project,
                        src_file.project,
                        pair.mapping == DataFileMapping.MIXED_SRC,
                        tags_str,
                        train,
                        mirror_cur_train,
                    )

                train = self._add_to_train_data_set(
                    src_file.project,
                    trg_file.project,
                    pair.mapping == DataFileMapping.MIXED_SRC,
                    tags_str,
                    train,
                    cur_train,
                )

        train_count = 0
        if train is not None and len(train) > 0:
            train_count = self._write_train(
                tokenizer, train, pair.mapping == DataFileMapping.MIXED_SRC, project_isos, pair.augmentations
            )

        val_count = 0
        if len(val) > 0:
            for (src_iso, trg_iso), pair_val in val.items():
                tokenizer.set_src_lang(src_iso)
                tokenizer.set_trg_lang(trg_iso)
                self._append_corpus(self.val_src_filename(), tokenizer.tokenize_all(Side.SOURCE, pair_val["source"]))
                self._append_corpus(self.val_src_detok_filename(), pair_val["source"])
            val_count = sum(len(pair_val) for pair_val in val.values())
            self._write_val_trg(tokenizer, val)
            self._write_val_trg(None, val)
            val_vref = itertools.chain.from_iterable(pair_val["vref"] for pair_val in val.values())
            self._append_corpus(self.val_vref_filename(), (str(vr) for vr in val_vref))

        test_count = 0
        for (src_iso, trg_iso), pair_test in test.items():
            tokenizer.set_src_lang(src_iso)
            tokenizer.set_trg_lang(trg_iso)
            self._append_corpus(self.test_vref_filename(src_iso, trg_iso), (str(vr) for vr in pair_test["vref"]))
            self._append_corpus(
                self.test_src_filename(src_iso, trg_iso),
                tokenizer.tokenize_all(Side.SOURCE, pair_test["source"]),
            )
            self._append_corpus(self.test_src_detok_filename(src_iso, trg_iso), pair_test["source"])
            test_count += len(pair_test)

            columns: List[str] = [c for c in pair_test.columns if c.startswith("target")]
            test_projects = self._get_test_projects(src_iso, trg_iso)
            for column in columns:
                project = column[len("target_") :]
                self._append_corpus(
                    self.test_trg_filename(src_iso, trg_iso, project),
                    tokenizer.normalize_all(Side.TARGET, pair_test[column]),
                )
                test_projects.remove(project)
            if self._has_multiple_test_projects(src_iso, trg_iso):
                for project in test_projects:
                    self._fill_corpus(self.test_trg_filename(src_iso, trg_iso, project), len(pair_test))
        LOGGER.info(f"train size: {train_count}," f" val size: {val_count}," f" test size: {test_count},")
        return train_count

    def _populate_pair_test_indices(self, exp_name: str, pair_test_indices: Dict[Tuple[str, str], Set[int]]) -> None:
        vrefs: Dict[str, int] = {}
        for i, vref_str in enumerate(load_corpus(SIL_NLP_ENV.assets_dir / "vref.txt")):
            if vref_str != "":
                vrefs[vref_str] = i

        SIL_NLP_ENV.copy_experiment_from_bucket(exp_name, patterns="test*.vref.txt")
        exp_dir = get_mt_exp_dir(exp_name)
        vref_paths: List[Path] = list(exp_dir.glob("test*.vref.txt"))
        if len(vref_paths) == 0:
            if Path.samefile(exp_dir, self.exp_dir):
                # Having the same experiment and "use_test_set_from" directory will also result in no test vrefs, but more cryptically.
                LOGGER.warning('The experiment specified in "use_test_set_from" is the same as the current experiment.')
            else:
                LOGGER.warning(
                    f'The experiment specified in "use_test_set_from" does not contain any files matching "test*.vref.txt".'
                )
        for vref_path in vref_paths:
            stem = vref_path.stem
            test_indices: Set[int] = set()
            if stem == "test.vref":
                pair_test_indices[(self.default_test_src_iso, self.default_test_trg_iso)] = test_indices
            else:
                _, src_iso, trg_iso, _ = stem.split(".", maxsplit=4)
                pair_test_indices[(src_iso, trg_iso)] = test_indices

            for vref_str in load_corpus(vref_path):
                vref = VerseRef.from_string(vref_str, ORIGINAL_VERSIFICATION)
                if vref.has_multiple:
                    vref.simplify()
                test_indices.add(vrefs[str(vref)])

    def _add_to_eval_data_set(
        self,
        src_iso: str,
        trg_iso: str,
        trg_project: str,
        tags_str: str,
        dataset: Dict[Tuple[str, str], pd.DataFrame],
        pair_indices: Dict[Tuple[str, str], Set[int]],
        new_data: pd.DataFrame,
    ) -> None:
        if len(new_data) == 0:
            return

        self._insert_tags(tags_str, new_data)

        pair_data = dataset.get((src_iso, trg_iso))

        if (src_iso, trg_iso) not in pair_indices:
            pair_indices[(src_iso, trg_iso)] = set(new_data.index)

        new_data.rename(columns={"target": f"target_{trg_project}"}, inplace=True)
        if pair_data is None:
            pair_data = new_data
        else:
            pair_data = pair_data.combine_first(new_data)
            pair_data.fillna("", inplace=True)

        dataset[(src_iso, trg_iso)] = pair_data

    def _add_to_train_data_set(
        self,
        src_project: str,
        trg_project: str,
        mixed_src: bool,
        tags_str: str,
        train: Optional[pd.DataFrame],
        cur_train: pd.DataFrame,
    ) -> pd.DataFrame:
        self._insert_tags(tags_str, cur_train)
        if mixed_src:
            cur_train.drop("source_lang", axis=1, inplace=True, errors="ignore")
            cur_train.rename(columns={"source": f"source_{src_project}"}, inplace=True)
            cur_train.set_index(
                pd.MultiIndex.from_tuples(
                    map(lambda i: (trg_project, i), cur_train.index), names=["trg_project", "index"]
                ),
                inplace=True,
            )
            train = cur_train if train is None else train.combine_first(cur_train)
        else:
            train = cur_train if train is None else pd.concat([train, cur_train], ignore_index=True)
        return train

    def _insert_tags(self, tags_str: str, sentences: pd.DataFrame) -> None:
        if tags_str != "":
            cast(Any, sentences).loc[:, "source"] = tags_str + sentences.loc[:, "source"]

    def _add_to_terms_data_set(
        self, terms: Optional[pd.DataFrame], cur_terms: pd.DataFrame, tags_str: str = ""
    ) -> pd.DataFrame:
        if self.mirror:
            mirror_cur_terms = cur_terms.rename(
                columns={
                    "source": "target",
                    "target": "source",
                    "source_lang": "target_lang",
                    "target_lang": "source_lang",
                }
            )
            self._insert_tags(tags_str, mirror_cur_terms)
            terms = mirror_cur_terms if terms is None else pd.concat([terms, mirror_cur_terms], ignore_index=True)

        self._insert_tags(tags_str, cur_terms)
        return cur_terms if terms is None else pd.concat([terms, cur_terms], ignore_index=True)

    def _write_train(
        self,
        tokenizer: Tokenizer,
        train: pd.DataFrame,
        mixed_src: bool,
        project_isos: Dict[str, str],
        augmentations: List[AugmentMethod],
    ) -> int:
        train_count = 0
        train.fillna("", inplace=True)
        src_columns: List[str] = [c for c in train.columns if c.startswith("source")]
        with ExitStack() as stack:
            train_src_file = stack.enter_context(self._open_append(self.train_src_filename()))
            train_trg_file = stack.enter_context(self._open_append(self.train_trg_filename()))
            train_vref_file = stack.enter_context(self._open_append(self.train_vref_filename()))
            train_src_detok_file = stack.enter_context(self._open_append(self.train_src_detok_filename()))
            train_trg_detok_file = stack.enter_context(self._open_append(self.train_trg_detok_filename()))

            for _, row in train.iterrows():
                if mixed_src:
                    nonempty_src_columns: List[str] = [c for c in src_columns if row[c] != ""]
                    source_column = random.choice(nonempty_src_columns)
                    src_sentence = row[source_column]
                    src_project = source_column[7:]
                    tokenizer.set_src_lang(project_isos[src_project])
                else:
                    src_sentence = row["source"]
                    tokenizer.set_src_lang(row["source_lang"])
                trg_sentence = row["target"]
                vref = row["vref"]
                tokenizer.set_trg_lang(row["target_lang"])

                train_src_file.write(tokenizer.tokenize(Side.SOURCE, src_sentence) + "\n")
                train_trg_file.write(tokenizer.tokenize(Side.TARGET, trg_sentence) + "\n")
                train_vref_file.write(str(vref) + "\n")
                train_src_detok_file.write(src_sentence + "\n")
                train_trg_detok_file.write(trg_sentence + "\n")
                train_count += 1

                src_augments, trg_augments = self._augment_sentence(
                    augmentations, src_sentence, trg_sentence, tokenizer
                )
                for src_augment, trg_augment in zip(src_augments, trg_augments):
                    train_src_file.write(src_augment + "\n")
                    train_trg_file.write(trg_augment + "\n")
                    train_vref_file.write(str(vref) + "\n")
                    train_count += 1
        return train_count

    def _write_terms(
        self,
        tokenizer: Tokenizer,
        src_terms_files: List[Tuple[DataFile, str]],
        trg_terms_files: List[Tuple[DataFile, str]],
    ) -> int:

        try:
            filter_books = get_books(self.data["terms"]["filter_books"])
        except KeyError:
            filter_books = None

        terms = self._collect_terms(src_terms_files, trg_terms_files, filter_books)

        if terms is None:
            return 0
        terms = terms.drop_duplicates(subset=["source", "target"])

        train_count = 0
        with ExitStack() as stack:
            train_src_file = stack.enter_context(self._open_append(self.train_src_filename()))
            train_trg_file = stack.enter_context(self._open_append(self.train_trg_filename()))
            train_vref_file = stack.enter_context(self._open_append(self.train_vref_filename()))
            train_src_detok_file = stack.enter_context(self._open_append(self.train_src_detok_filename()))
            train_trg_detok_file = stack.enter_context(self._open_append(self.train_trg_detok_filename()))

            for _, term in terms.iterrows():
                src_term = term["source"]
                trg_term = term["target"]
                tokenizer.set_src_lang(term["source_lang"])
                tokenizer.set_trg_lang(term["target_lang"])

                src_term_variants = [
                    tokenizer.tokenize(Side.SOURCE, src_term, add_dummy_prefix=True),
                    tokenizer.tokenize(Side.SOURCE, src_term, add_dummy_prefix=False),
                ]
                trg_term_variants = [
                    tokenizer.tokenize(Side.TARGET, trg_term, add_dummy_prefix=True),
                    tokenizer.tokenize(Side.TARGET, trg_term, add_dummy_prefix=False),
                ]
                for stv, ttv in zip(src_term_variants, trg_term_variants):
                    train_src_file.write(stv + "\n")
                    train_trg_file.write(ttv + "\n")
                    train_vref_file.write("\n")
                    train_count += 1

                train_src_detok_file.write(src_term + "\n")
                train_trg_detok_file.write(trg_term + "\n")
        return train_count

    def _collect_terms(
        self,
        src_terms_files: List[Tuple[DataFile, str]],
        trg_terms_files: List[Tuple[DataFile, str]],
        filter_books: Optional[Set[int]] = None,
    ) -> Optional[pd.DataFrame]:
        terms_config = self.data["terms"]
        terms: Optional[pd.DataFrame] = None
        categories: Optional[Union[str, List[str]]] = terms_config["categories"]
        if isinstance(categories, str):
            categories = [cat.strip() for cat in categories.split(",")]
        if categories is not None and len(categories) == 0:
            return None
        categories_set: Optional[Set[str]] = None if categories is None else set(categories)

        if terms_config["include_glosses"]:
            if terms_config["include_glosses"] in ["en", "fr", "id", "es"]:
                gloss_iso = terms_config["include_glosses"]
            src_gloss_iso = list(self.src_isos.intersection(["en", "fr", "id", "es"]))
            trg_gloss_iso = list(self.trg_isos.intersection(["en", "fr", "id", "es"]))
            if src_gloss_iso:
                gloss_iso = src_gloss_iso[0]
            elif trg_gloss_iso:
                gloss_iso = trg_gloss_iso[0]
        else:
            gloss_iso = ""

        all_src_terms: List[Tuple[DataFile, Dict[str, Term], str]] = []
        for src_terms_file, tags_str in src_terms_files:
            all_src_terms.append((src_terms_file, get_terms(src_terms_file.path, iso=gloss_iso), tags_str))

        all_trg_terms: List[Tuple[DataFile, Dict[str, Term], str]] = []
        for trg_terms_file, tags_str in trg_terms_files:
            all_trg_terms.append((trg_terms_file, get_terms(trg_terms_file.path, iso=gloss_iso), tags_str))

        for src_terms_file, src_terms, tags_str in all_src_terms:
            for trg_terms_file, trg_terms, trg_tags_str in all_trg_terms:
                if src_terms_file.iso == trg_terms_file.iso:
                    continue
                cur_terms = get_terms_corpus(src_terms, trg_terms, categories_set, filter_books)
                cur_terms["source_lang"] = src_terms_file.iso
                cur_terms["target_lang"] = trg_terms_file.iso
                terms = self._add_to_terms_data_set(terms, cur_terms, tags_str)
        if gloss_iso is not "":
            if gloss_iso in self.trg_isos:
                for src_terms_file, src_terms, tags_str in all_src_terms:
                    cur_terms = get_terms_data_frame(src_terms, categories_set, filter_books)
                    cur_terms = cur_terms.rename(columns={"rendering": "source", "gloss": "target"})
                    cur_terms["source_lang"] = src_terms_file.iso
                    cur_terms["target_lang"] = gloss_iso
                    terms = self._add_to_terms_data_set(terms, cur_terms, tags_str)
            if gloss_iso in self.src_isos or gloss_iso == terms_config["include_glosses"]:
                for trg_terms_file, trg_terms, tags_str in all_trg_terms:
                    cur_terms = get_terms_data_frame(trg_terms, categories_set, filter_books)
                    cur_terms = cur_terms.rename(columns={"rendering": "target", "gloss": "source"})
                    cur_terms["source_lang"] = gloss_iso
                    cur_terms["target_lang"] = trg_terms_file.iso
                    terms = self._add_to_terms_data_set(terms, cur_terms, tags_str)
        return terms

    def _write_val_trg(self, tokenizer: Optional[Tokenizer], val: Dict[Tuple[str, str], pd.DataFrame]) -> None:
        with ExitStack() as stack:
            ref_files: List[TextIO] = []
            for (src_iso, trg_iso), pair_val in val.items():
                if tokenizer is not None:
                    tokenizer.set_src_lang(src_iso)
                    tokenizer.set_trg_lang(trg_iso)
                columns: List[str] = [c for c in pair_val.columns if c.startswith("target")]
                if self.root["eval"]["multi_ref_eval"]:
                    val_project_count = self._get_val_ref_count(src_iso, trg_iso)
                    for index in pair_val.index:
                        for ci in range(val_project_count):
                            if len(ref_files) == ci:
                                ref_files.append(stack.enter_context(self._open_append(self.val_trg_filename(ci))))
                            if ci < len(columns):
                                col = columns[ci]
                                if tokenizer is not None:
                                    ref_files[ci].write(
                                        tokenizer.tokenize(Side.TARGET, cast(str, pair_val.loc[index, col]).strip())
                                        + "\n"
                                    )
                                else:
                                    ref_files[ci].write(pair_val.loc[index, col] + "\n")
                            else:
                                ref_files[ci].write("\n")
                else:
                    for index in pair_val.index:
                        if len(ref_files) == 0:
                            fn = self.val_trg_filename() if tokenizer is not None else self.val_trg_detok_filename()
                            ref_files.append(stack.enter_context(self._open_append(fn)))
                        columns_with_data = [c for c in columns if cast(str, pair_val.loc[index, c]).strip() != ""]
                        col = random.choice(columns_with_data)
                        if tokenizer is not None:
                            ref_files[0].write(
                                tokenizer.tokenize(Side.TARGET, cast(str, pair_val.loc[index, col]).strip()) + "\n"
                            )
                        else:
                            ref_files[0].write(pair_val.loc[index, col] + "\n")

    def _append_corpus(self, filename: str, sentences: Iterable[str]) -> None:
        write_corpus(self.exp_dir / filename, sentences, append=True)

    def _fill_corpus(self, filename: str, size: int) -> None:
        write_corpus(self.exp_dir / filename, ("" for _ in range(size)), append=True)

    def _open_append(self, filename: str) -> TextIO:
        return (self.exp_dir / filename).open("a", encoding="utf-8", newline="\n")

    def _write_basic_data_sets(self, tokenizer: Tokenizer, pair: CorpusPair) -> int:
        total_train_count = 0
        for src_file, trg_file in zip(pair.src_files, pair.trg_files):
            total_train_count += self._write_basic_data_file_pair(tokenizer, pair, src_file, trg_file)
        return total_train_count

    def _write_basic_data_file_pair(
        self,
        tokenizer: Tokenizer,
        pair: CorpusPair,
        src_file: DataFile,
        trg_file: DataFile,
    ) -> int:
        LOGGER.info(f"Preprocessing {src_file.path.stem} -> {trg_file.path.stem}")
        tokenizer.set_src_lang(src_file.iso)
        tokenizer.set_trg_lang(trg_file.iso)
        corpus_size = get_parallel_corpus_size(src_file.path, trg_file.path)
        train_count = 0
        val_count = 0
        test_count = 0
        dict_count = 0
        with ExitStack() as stack:
            input_src_file = stack.enter_context(src_file.path.open("r", encoding="utf-8"))
            input_trg_file = stack.enter_context(trg_file.path.open("r", encoding="utf-8"))
            test_indices: Optional[Set[int]] = set()
            if pair.is_test:
                test_size = pair.size if pair.test_size is None else pair.test_size
                test_indices = split_corpus(corpus_size, test_size)

            val_indices: Optional[Set[int]] = set()
            if pair.is_val and test_indices is not None:
                val_size = pair.size if pair.val_size is None else pair.val_size
                val_indices = split_corpus(corpus_size, val_size, test_indices)

            train_indices: Optional[Set[int]] = set()
            if pair.is_train and test_indices is not None and val_indices is not None:
                train_size = pair.size
                train_indices = split_corpus(corpus_size, train_size, test_indices | val_indices)

            tags_str = self._get_tags_str(pair.tags)

            train_src_file = stack.enter_context(self._open_append(self.train_src_filename()))
            train_trg_file = stack.enter_context(self._open_append(self.train_trg_filename()))
            val_src_file = stack.enter_context(self._open_append(self.val_src_filename()))
            val_trg_file = stack.enter_context(self._open_append(self.val_trg_filename()))
            test_src_file = stack.enter_context(self._open_append(self.test_src_filename(src_file.iso, trg_file.iso)))
            test_trg_file = stack.enter_context(self._open_append(self.test_trg_filename(src_file.iso, trg_file.iso)))

            train_vref_file: Optional[TextIO] = None
            val_vref_file: Optional[TextIO] = None
            test_vref_file: Optional[TextIO] = None
            test_trg_project_files: List[TextIO] = []
            val_trg_ref_files: List[TextIO] = []
            dict_src_file: Optional[TextIO] = None
            dict_trg_file: Optional[TextIO] = None
            dict_vref_file: Optional[TextIO] = None
            if self.has_scripture_data:
                train_vref_file = stack.enter_context(self._open_append(self.train_vref_filename()))
                val_vref_file = stack.enter_context(self._open_append(self.val_vref_filename()))
                test_vref_file = stack.enter_context(
                    self._open_append(self.test_vref_filename(src_file.iso, trg_file.iso))
                )
                test_projects = self._get_test_projects(src_file.iso, trg_file.iso)
                if self._has_multiple_test_projects(src_file.iso, trg_file.iso):
                    test_trg_project_files = [
                        stack.enter_context(
                            self._open_append(self.test_trg_filename(src_file.iso, trg_file.iso, project))
                        )
                        for project in test_projects
                        if project != BASIC_DATA_PROJECT
                    ]
                val_ref_count = self._get_val_ref_count(src_file.iso, trg_file.iso)
                val_trg_ref_files = [
                    stack.enter_context(self._open_append(self.val_trg_filename(index)))
                    for index in range(1, val_ref_count)
                ]
            if pair.is_dictionary:
                dict_src_file = stack.enter_context(self._open_append(self.dict_src_filename()))
                dict_trg_file = stack.enter_context(self._open_append(self.dict_trg_filename()))
                dict_vref_file = stack.enter_context(self._open_append(self.dict_vref_filename()))

            index = 0
            for src_line, trg_line in tqdm(zip(input_src_file, input_trg_file)):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                if len(src_line) == 0 or len(trg_line) == 0:
                    continue

                src_sentence = tags_str + src_line
                trg_sentence = trg_line

                if pair.is_test and (test_indices is None or index in test_indices):
                    test_src_file.write(tokenizer.tokenize(Side.SOURCE, src_sentence) + "\n")
                    test_trg_file.write(tokenizer.normalize(Side.TARGET, trg_sentence) + "\n")
                    if test_vref_file is not None:
                        test_vref_file.write("\n")
                    for test_trg_project_file in test_trg_project_files:
                        test_trg_project_file.write("\n")
                    test_count += 1
                elif pair.is_val and (val_indices is None or index in val_indices):
                    val_src_file.write(tokenizer.tokenize(Side.SOURCE, src_sentence) + "\n")
                    val_trg_file.write(tokenizer.tokenize(Side.TARGET, trg_sentence) + "\n")
                    if val_vref_file is not None:
                        val_vref_file.write("\n")
                    for val_trg_ref_file in val_trg_ref_files:
                        val_trg_ref_file.write("\n")
                    val_count += 1
                elif pair.is_train and (train_indices is None or index in train_indices):
                    noised_src_sentence = tags_str + self._noise(pair.src_noise, src_line)
                    train_count += self._write_train_sentence_pair(
                        train_src_file,
                        train_trg_file,
                        train_vref_file,
                        tokenizer,
                        noised_src_sentence,
                        trg_sentence,
                        pair.is_lexical_data,
                        pair.augmentations,
                    )
                    if self.mirror:
                        tokenizer.set_src_lang(trg_file.iso)
                        tokenizer.set_trg_lang(src_file.iso)
                        mirror_src_sentence = tags_str + self._noise(pair.src_noise, trg_line)
                        mirror_trg_sentence = src_line
                        train_count += self._write_train_sentence_pair(
                            train_src_file,
                            train_trg_file,
                            train_vref_file,
                            tokenizer,
                            mirror_src_sentence,
                            mirror_trg_sentence,
                            pair.is_lexical_data,
                            pair.augmentations,
                        )
                        tokenizer.set_src_lang(src_file.iso)
                        tokenizer.set_trg_lang(trg_file.iso)

                if (
                    pair.is_dictionary
                    and dict_src_file is not None
                    and dict_trg_file is not None
                    and dict_vref_file is not None
                ):
                    src_variants = [
                        tokenizer.tokenize(Side.SOURCE, src_sentence, add_dummy_prefix=True, add_special_tokens=False),
                        tokenizer.tokenize(Side.SOURCE, src_sentence, add_dummy_prefix=False, add_special_tokens=False),
                    ]
                    trg_variants = [
                        tokenizer.tokenize(Side.TARGET, trg_sentence, add_dummy_prefix=True, add_special_tokens=False),
                        tokenizer.tokenize(Side.TARGET, trg_sentence, add_dummy_prefix=False, add_special_tokens=False),
                    ]
                    dict_src_file.write("\t".join(src_variants) + "\n")
                    dict_trg_file.write("\t".join(trg_variants) + "\n")
                    dict_vref_file.write("\n")
                    dict_count += 1

                index += 1

        LOGGER.info(
            f"train size: {train_count}, val size: {val_count}, test size: {test_count}, dict size: {dict_count}"
        )
        return train_count

    def _write_train_sentence_pair(
        self,
        src_file: TextIO,
        trg_file: TextIO,
        vref_file: Optional[TextIO],
        tokenizer: Tokenizer,
        src_sentence: str,
        trg_sentence: str,
        is_lexical: bool,
        augmentations: List[AugmentMethod],
    ) -> int:
        src_variants = [tokenizer.tokenize(Side.SOURCE, src_sentence, add_dummy_prefix=True)]
        trg_variants = [tokenizer.tokenize(Side.TARGET, trg_sentence, add_dummy_prefix=True)]

        if is_lexical:
            src_variants.append(tokenizer.tokenize(Side.SOURCE, src_sentence, add_dummy_prefix=False))
            trg_variants.append(tokenizer.tokenize(Side.TARGET, trg_sentence, add_dummy_prefix=False))

        src_augments, trg_augments = self._augment_sentence(augmentations, src_sentence, trg_sentence, tokenizer)
        src_variants.extend(src_augments)
        trg_variants.extend(trg_augments)

        for src_variant, trg_variant in zip(src_variants, trg_variants):
            src_file.write(src_variant + "\n")
            trg_file.write(trg_variant + "\n")
            if vref_file is not None:
                vref_file.write("\n")
        return len(src_variants)

    def _get_tags_str(self, tags: List[str]) -> str:
        tags_str = ""
        if len(tags) > 0:
            tags_str += " ".join(f"<{t}>" for t in tags) + " "
        return tags_str

    def _noise(self, src_noise: List[NoiseMethod], src_sentence: str) -> str:
        if len(src_noise) == 0:
            return src_sentence
        tokens = src_sentence.split()
        for noise_method in src_noise:
            tokens = noise_method(tokens)
        return " ".join(tokens)

    def _augment_sentence(
        self, methods: List[AugmentMethod], src: str, trg: str, tokenizer: Tokenizer
    ) -> Tuple[List[str], List[str]]:
        src_augments: List[str] = []
        trg_augments: List[str] = []
        for method in methods:
            src_delta, trg_delta = method.augment_sentence(src, trg, tokenizer)
            src_augments.extend(src_delta)
            trg_augments.extend(trg_delta)
        return src_augments, trg_augments

    def _get_val_ref_count(self, src_iso: str, trg_iso: str) -> int:
        if self.root["eval"]["multi_ref_eval"]:
            return len(self._iso_pairs[(src_iso, trg_iso)].val_projects)
        return 1

    def _get_test_projects(self, src_iso: str, trg_iso: str) -> Set[str]:
        iso_pair = self._iso_pairs[(src_iso, trg_iso)]
        test_projects = iso_pair.test_projects.copy()
        if iso_pair.has_basic_test_data:
            test_projects.add(BASIC_DATA_PROJECT)
        return test_projects

    def train_src_filename(self) -> str:
        return "train.src.txt"

    def train_src_detok_filename(self) -> str:
        return "train.src.detok.txt"

    def train_trg_filename(self) -> str:
        return "train.trg.txt"

    def train_trg_detok_filename(self) -> str:
        return "train.trg.detok.txt"

    def train_vref_filename(self) -> str:
        return "train.vref.txt"

    def val_src_filename(self) -> str:
        return "val.src.txt"

    def val_src_detok_filename(self) -> str:
        return "val.src.detok.txt"

    def val_vref_filename(self) -> str:
        return "val.vref.txt"

    def val_trg_filename(self, index: int = 0) -> str:
        return f"val.trg.txt.{index}" if self.root["eval"]["multi_ref_eval"] else "val.trg.txt"

    def val_trg_detok_filename(self, index: int = 0) -> str:
        return f"val.trg.detok.txt.{index}" if self.root["eval"]["multi_ref_eval"] else "val.trg.detok.txt"

    def test_src_filename(self, src_iso: str, trg_iso: str) -> str:
        return f"test.{src_iso}.{trg_iso}.src.txt" if self._multiple_test_iso_pairs else "test.src.txt"

    def test_src_detok_filename(self, src_iso: str, trg_iso: str) -> str:
        return f"test.{src_iso}.{trg_iso}.src.detok.txt" if self._multiple_test_iso_pairs else "test.src.detok.txt"

    def test_vref_filename(self, src_iso: str, trg_iso: str) -> str:
        return f"test.{src_iso}.{trg_iso}.vref.txt" if self._multiple_test_iso_pairs else "test.vref.txt"

    def test_trg_filename(self, src_iso: str, trg_iso: str, project: str = BASIC_DATA_PROJECT) -> str:
        prefix = f"test.{src_iso}.{trg_iso}" if self._multiple_test_iso_pairs else "test"
        has_multiple_test_projects = self._iso_pairs[(src_iso, trg_iso)].has_multiple_test_projects
        suffix = f".{project}" if has_multiple_test_projects else ""
        return f"{prefix}.trg.detok{suffix}.txt"

    def dict_src_filename(self) -> str:
        return "dict.src.txt"

    def dict_trg_filename(self) -> str:
        return "dict.trg.txt"

    def dict_vref_filename(self) -> str:
        return "dict.vref.txt"

    def _has_multiple_test_projects(self, src_iso: str, trg_iso: str) -> bool:
        return self._iso_pairs[(src_iso, trg_iso)].has_multiple_test_projects

    @abstractmethod
    def _build_vocabs(self, stats: bool = False) -> None: ...

    @abstractmethod
    def _write_dictionary(
        self,
        tokenizer: Tokenizer,
        src_terms_files: List[Tuple[DataFile, str]],
        trg_terms_files: List[Tuple[DataFile, str]],
    ) -> int: ...
