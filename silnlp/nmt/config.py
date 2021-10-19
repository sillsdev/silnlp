import argparse
import itertools
import logging
import os
import random
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple, Type, Union, cast

import pandas as pd
import sentencepiece as sp
import tensorflow as tf
import yaml
from machine.scripture import VerseRef, get_books
from opennmt import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN
from opennmt.data import Noise, Vocab, WordDropout, WordNoiser, tokens_to_words
from opennmt.inputters import TextInputter
from opennmt.models import Model, get_model_from_catalog

from ..alignment.machine_aligner import FastAlignMachineAligner
from ..alignment.utils import add_alignment_scores
from ..common.corpus import (
    exclude_books,
    filter_parallel_corpus,
    get_scripture_parallel_corpus,
    get_terms,
    get_terms_corpus,
    get_terms_data_frame,
    get_terms_glosses_path,
    get_terms_list,
    get_terms_renderings_path,
    include_books,
    load_corpus,
    split_corpus,
    split_parallel_corpus,
    write_corpus,
)
from ..common.environment import SIL_NLP_ENV, download_if_s3_path, download_if_s3_paths
from ..common.utils import (
    DeleteRandomToken,
    NoiseMethod,
    RandomTokenPermutation,
    ReplaceRandomToken,
    get_git_revision_hash,
    get_mt_exp_dir,
    is_set,
    merge_dict,
    set_seed,
)
from .runner import SILRunner
from .transformer import SILTransformer
from .utils import decode_sp, decode_sp_lines, encode_sp, encode_sp_lines, get_best_model_dir, get_last_checkpoint

_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL: Dict[int, int] = {
    logging.CRITICAL: 3,
    logging.ERROR: 2,
    logging.WARNING: 1,
    logging.INFO: 0,
    logging.DEBUG: 0,
    logging.NOTSET: 0,
}


LOGGER = logging.getLogger(__package__ + ".config")

_DEFAULT_NEW_CONFIG: dict = {
    "data": {
        "share_vocab": False,
        "character_coverage": 1.0,
        "mirror": False,
        "mixed_src": False,
        "seed": 111,
    },
    "train": {"maximum_features_length": 150, "maximum_labels_length": 150},
    "eval": {"multi_ref_eval": False},
    "params": {
        "length_penalty": 0.2,
        "dropout": 0.2,
        "transformer_dropout": 0.1,
        "transformer_attention_dropout": 0.1,
        "transformer_ffn_dropout": 0.1,
        "word_dropout": 0.1,
    },
}

BASIC_DATA_PROJECT = "BASIC"


# Different types of parent model checkpoints (last, best, average)
class CheckpointType(Enum):
    LAST = 1
    BEST = 2
    AVERAGE = 3


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

    def __post_init__(self):
        file_name = self.path.stem
        parts = file_name.split("-")
        if len(parts) < 2:
            raise RuntimeError(f"The filename {file_name} needs to be of the format <iso>-<project>")
        self.iso = parts[0]
        self.project = parts[1] if self.path.parent == SIL_NLP_ENV.mt_scripture_dir else BASIC_DATA_PROJECT

    @property
    def is_scripture(self) -> bool:
        return self.project != BASIC_DATA_PROJECT


@dataclass
class CorpusPair:
    src_files: List[DataFile]
    trg_files: List[DataFile]
    type: DataFileType
    src_noise: List[NoiseMethod]
    tags: List[str]
    size: Union[float, int]
    test_size: Optional[Union[float, int]]
    val_size: Optional[Union[float, int]]
    disjoint_test: bool
    disjoint_val: bool
    score_threshold: float
    corpus_books: Set[int]
    test_books: Set[int]
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


def create_noise_methods(params: List[dict]) -> List[NoiseMethod]:
    methods: List[NoiseMethod] = []
    for module in params:
        noise_type, args = next(iter(module.items()))
        if not isinstance(args, list):
            args = [args]
        noise_type = noise_type.lower()
        noise_method_class: Type[NoiseMethod]
        if noise_type == "dropout":
            noise_method_class = DeleteRandomToken
        elif noise_type == "replacement":
            noise_method_class = ReplaceRandomToken
        elif noise_type == "permutation":
            noise_method_class = RandomTokenPermutation
        else:
            raise ValueError("Invalid noise type: %s" % noise_type)
        methods.append(noise_method_class(*args))
    return methods


def get_corpus_path(corpus: str) -> Path:
    corpus_path = SIL_NLP_ENV.mt_corpora_dir / f"{corpus}.txt"
    if corpus_path.is_file():
        return corpus_path
    return SIL_NLP_ENV.mt_scripture_dir / f"{corpus}.txt"


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

        src: Union[str, List[str]] = pair["src"]
        if isinstance(src, str):
            src = src.split(",")
        src_files = [DataFile(get_corpus_path(sp.strip())) for sp in src]
        trg: Union[str, List[str]] = pair["trg"]
        if isinstance(trg, str):
            trg = trg.split(",")
        trg_files = [DataFile(get_corpus_path(tp.strip())) for tp in trg]
        is_scripture = src_files[0].is_scripture
        if not all(df.is_scripture == is_scripture for df in (src_files + trg_files)):
            raise RuntimeError("All corpora in a corpus pair must contain the same type of data.")

        tags: Union[str, List[str]] = pair.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",")]

        src_noise = create_noise_methods(pair.get("src_noise", []))

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
            pair["disjoint_test"] = False
        disjoint_test: bool = pair["disjoint_test"]
        if "disjoint_val" not in pair:
            pair["disjoint_val"] = False
        disjoint_val: bool = pair["disjoint_val"]
        score_threshold: float = pair.get("score_threshold", 0.0)
        corpus_books = get_books(pair.get("corpus_books", []))
        test_books = get_books(pair.get("test_books", []))
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


def convert_vocab(sp_vocab_path: Path, onmt_vocab_path: Path) -> None:
    special_tokens = [PADDING_TOKEN, START_OF_SENTENCE_TOKEN, END_OF_SENTENCE_TOKEN]

    vocab = Vocab(special_tokens)
    with sp_vocab_path.open("r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            token = line.rstrip("\r\n")
            index = token.rindex("\t")
            token = token[:index]
            if token in ("<unk>", "<s>", "</s>", "<range>"):  # Ignore special tokens
                continue
            vocab.add(token)
    vocab.pad_to_multiple(8)
    vocab.serialize(onmt_vocab_path)


def build_vocab(
    file_paths: Iterable[Path],
    vocab_size: int,
    casing: str,
    character_coverage: float,
    model_prefix: Path,
    vocab_path: Path,
    tags: Set[str],
) -> None:
    user_defined_symbols = "<blank>"
    for tag in tags:
        user_defined_symbols += f",{tag}"

    casing = casing.lower()
    normalization: str
    if casing == "lower":
        normalization = "nmt_nfkc_cf"
    elif casing == "preserve":
        normalization = "nmt_nfkc"
    else:
        raise RuntimeError("Invalid casing was specified in the config.")

    # use custom normalization that does not convert ZWJ and ZWNJ to spaces
    # allows properly handling of scripts like Devanagari
    normalization_path = Path(__file__).parent / f"{normalization}.tsv"
    file_paths = [fp for fp in file_paths]
    file_paths.sort()

    file_paths = download_if_s3_paths(file_paths)

    sp.SentencePieceTrainer.Train(
        normalization_rule_tsv=normalization_path,
        input=file_paths,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        user_defined_symbols=user_defined_symbols,
        character_coverage="%.4f" % character_coverage,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        control_symbols="<range>",
    )

    convert_vocab(model_prefix.with_suffix(".vocab"), vocab_path)


def get_checkpoint_path(model_dir: Path, checkpoint_type: CheckpointType) -> Tuple[Optional[Path], Optional[int]]:
    if checkpoint_type == CheckpointType.AVERAGE:
        # Get the checkpoint path and step count for the averaged checkpoint
        return get_last_checkpoint(model_dir / "avg")
    elif checkpoint_type == CheckpointType.BEST:
        # Get the checkpoint path and step count for the best checkpoint
        best_model_dir, step = get_best_model_dir(model_dir)
        return (best_model_dir / "ckpt", step)
    elif checkpoint_type == CheckpointType.LAST:
        return (None, None)
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {checkpoint_type}")


def get_data_file_pairs(corpus_pair: CorpusPair) -> Iterable[Tuple[DataFile, DataFile]]:
    if corpus_pair.mapping == DataFileMapping.ONE_TO_ONE:
        for file_pair in zip(corpus_pair.src_files, corpus_pair.trg_files):
            yield file_pair
    else:
        for src_file in corpus_pair.src_files:
            for trg_file in corpus_pair.trg_files:
                if src_file.iso == trg_file.iso:
                    continue
                yield (src_file, trg_file)


class Config:
    def __init__(self, exp_dir: Path, config: dict) -> None:
        config = merge_dict(
            {
                "model": "SILTransformerBase",
                "model_dir": str(exp_dir / "run"),
                "data": {
                    "train_features_file": str(exp_dir / "train.src.txt"),
                    "train_labels_file": str(exp_dir / "train.trg.txt"),
                    "eval_features_file": str(exp_dir / "val.src.txt"),
                    "eval_labels_file": str(exp_dir / "val.trg.txt"),
                    "share_vocab": True,
                    "character_coverage": 1.0,
                    "mirror": False,
                    "parent_use_best": False,
                    "parent_use_average": False,
                    "parent_use_vocab": False,
                    "seed": 111,
                    "tokenize": True,
                    "guided_alignment": False,
                    "guided_alignment_train_size": 1000000,
                    "stats_max_size": 100000,  # a little over the size of the bible
                    "terms": {
                        "train": True,
                        "dictionary": False,
                        "categories": "PN",
                        "include_glosses": True,
                    },
                },
                "train": {
                    "average_last_checkpoints": 0,
                    "maximum_features_length": 150,
                    "maximum_labels_length": 150,
                    "keep_checkpoint_max": 3,
                    "save_checkpoints_steps": 1000,
                },
                "eval": {
                    "external_evaluators": "bleu_multi_ref",
                    "steps": 1000,
                    "early_stopping": {"metric": "bleu", "min_improvement": 0.2, "steps": 4},
                    "export_on_best": "bleu",
                    "export_format": "checkpoint",
                    "max_exports_to_keep": 1,
                    "multi_ref_eval": False,
                    "use_dictionary": True,
                },
                "params": {
                    "length_penalty": 0.2,
                    "transformer_dropout": 0.1,
                    "transformer_attention_dropout": 0.1,
                    "transformer_ffn_dropout": 0.1,
                    "word_dropout": 0,
                    "guided_alignment_type": "mse",
                    "guided_alignment_weight": 0.3,
                },
            },
            config,
        )
        data_config: dict = config["data"]
        eval_config: dict = config["eval"]
        multi_ref_eval: bool = eval_config["multi_ref_eval"]
        if multi_ref_eval:
            data_config["eval_labels_file"] = str(exp_dir / "val.trg.txt.0")
        if data_config["share_vocab"]:
            data_config["source_vocabulary"] = str(exp_dir / "onmt.vocab")
            data_config["target_vocabulary"] = str(exp_dir / "onmt.vocab")
            if (
                "src_vocab_size" not in data_config
                and "trg_vocab_size" not in data_config
                and "vocab_size" not in data_config
            ):
                data_config["vocab_size"] = 24000
            if "src_casing" not in data_config and "trg_casing" not in data_config and "casing" not in data_config:
                data_config["casing"] = "lower"
        else:
            data_config["source_vocabulary"] = str(exp_dir / "src-onmt.vocab")
            data_config["target_vocabulary"] = str(exp_dir / "trg-onmt.vocab")
            if "vocab_size" not in data_config:
                if "src_vocab_size" not in data_config:
                    data_config["src_vocab_size"] = 8000
                if "trg_vocab_size" not in data_config:
                    data_config["trg_vocab_size"] = 8000
            if "casing" not in data_config:
                if "src_casing" not in data_config:
                    data_config["src_casing"] = "lower"
                if "trg_casing" not in data_config:
                    data_config["trg_casing"] = "lower"

        model: str = config["model"]
        if model.endswith("AlignmentEnhanced"):
            data_config["guided_alignment"] = True

        if data_config["guided_alignment"]:
            if config["params"]["word_dropout"] > 0:
                raise RuntimeError("Guided alignment will not work with word dropout enabled.")
            data_config["train_alignments"] = str(exp_dir / "train.alignments.txt")

        self.exp_dir = exp_dir
        self.root = config

        self.corpus_pairs = parse_corpus_pairs(data_config["corpus_pairs"])

        if any(
            p.is_dictionary or (len(p.src_terms_files) > 0 and data_config["terms"]["dictionary"])
            for p in self.corpus_pairs
        ):
            data_config["source_dictionary"] = str(exp_dir / self._dict_src_filename())
            data_config["target_dictionary"] = str(exp_dir / self._dict_trg_filename())

        terms_config: dict = data_config["terms"]
        self.src_isos: Set[str] = set()
        self.trg_isos: Set[str] = set()
        self.src_file_paths: Set[Path] = set()
        self.trg_file_paths: Set[Path] = set()
        self._tags: Set[str] = set()
        self._has_scripture_data = False
        self._iso_pairs: Dict[Tuple[str, str], IsoPairInfo] = {}
        self.src_projects: Set[str] = set()
        for corpus_pair in self.corpus_pairs:
            pair_src_isos = {sf.iso for sf in corpus_pair.src_files}
            pair_trg_isos = {tf.iso for tf in corpus_pair.trg_files}
            self.src_isos.update(pair_src_isos)
            self.trg_isos.update(pair_trg_isos)
            self.src_file_paths.update(sf.path for sf in corpus_pair.src_files)
            self.trg_file_paths.update(tf.path for tf in corpus_pair.trg_files)
            if corpus_pair.is_scripture:
                self._has_scripture_data = True
                self.src_file_paths.update(sf.path for sf in corpus_pair.src_terms_files)
                self.trg_file_paths.update(tf.path for tf in corpus_pair.trg_terms_files)
                self.src_projects.update(sf.project for sf in corpus_pair.src_files)
                if terms_config["include_glosses"]:
                    if "en" in pair_src_isos:
                        self.src_file_paths.update(get_terms_glosses_file_paths(corpus_pair.src_terms_files))
                    if "en" in pair_trg_isos:
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

        parent: Optional[str] = self.data.get("parent")
        self.parent_config: Optional[Config] = None
        if parent is not None:
            SIL_NLP_ENV.copy_experiment_from_bucket(parent, extensions=("config.yml", ".model", ".vocab"))
            self.parent_config = load_config(parent)
            freeze_layers: Optional[List[str]] = self.parent_config.params.get("freeze_layers")
            # do not freeze any word embeddings layer, because we will update them when we create the parent model
            if freeze_layers is not None:
                self.parent_config.params["freeze_layers"] = list()

        self.write_trg_tag: bool = (
            len(self.trg_isos) > 1
            or self.mirror
            or (self.parent_config is not None and self.parent_config.write_trg_tag)
        )

        if self.write_trg_tag:
            self._tags.update(f"<2{trg_iso}>" for trg_iso in self.trg_isos)
            if self.mirror:
                self._tags.update(f"<2{src_iso}>" for src_iso in self.src_isos)

    @property
    def default_src_iso(self) -> str:
        return next(iter(self.src_isos))

    @property
    def default_trg_iso(self) -> str:
        return next(iter(self.trg_isos))

    @property
    def model(self) -> str:
        return self.root["model"]

    @property
    def model_dir(self) -> Path:
        return Path(self.root["model_dir"])

    @property
    def params(self) -> dict:
        return self.root["params"]

    @property
    def data(self) -> dict:
        return self.root["data"]

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

    def set_seed(self) -> None:
        seed = self.data["seed"]
        set_seed(seed)
        tf.random.set_seed(seed)

    def preprocess(self, stats: bool) -> None:
        # confirm that input file paths exist
        for file in self.src_file_paths | self.trg_file_paths:
            if not file.is_file():
                LOGGER.error("The source file " + str(file) + " does not exist.")
                return

        self._build_vocabs()
        src_spp, trg_spp = self.create_sp_processors()
        train_count = self._build_corpora(src_spp, trg_spp, stats)
        if self.data["guided_alignment"]:
            LOGGER.info("Generating train alignments")
            self._create_train_alignments(train_count)
        LOGGER.info("Preprocessing completed")

    def create_sp_processors(self) -> Tuple[Optional[sp.SentencePieceProcessor], Optional[sp.SentencePieceProcessor]]:
        if not self.data["tokenize"]:
            return (None, None)
        if self.share_vocab:
            model_prefix = self.exp_dir / "sp"
            src_spp = sp.SentencePieceProcessor()
            src_spp.Load(str(model_prefix.with_suffix(".model")))

            trg_spp = src_spp
        else:
            src_spp = sp.SentencePieceProcessor()
            src_spp.Load(str(self.exp_dir / "src-sp.model"))

            trg_spp = sp.SentencePieceProcessor()
            trg_spp.Load(str(self.exp_dir / "trg-sp.model"))
        return (src_spp, trg_spp)

    def create_src_sp_processor(self) -> Optional[sp.SentencePieceProcessor]:
        if not self.data["tokenize"]:
            return None
        src_spp = sp.SentencePieceProcessor()
        src_spp.Load(str(self.exp_dir / "sp.model" if self.share_vocab else self.exp_dir / "src-sp.model"))
        return src_spp

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
            return self.default_trg_iso, parts[3]
        return parts[2], parts[5]

    def _build_corpora(
        self, src_spp: Optional[sp.SentencePieceProcessor], trg_spp: Optional[sp.SentencePieceProcessor], stats: bool
    ) -> int:
        self._delete_files("train.*.txt")
        self._delete_files("val.*.txt")
        self._delete_files("test.*.txt")
        self._delete_files("dict.*.txt")

        train_count = 0
        for pair in self.corpus_pairs:
            if pair.is_scripture:
                train_count += self._write_scripture_data_sets(src_spp, trg_spp, pair, stats)
            else:
                train_count += self._write_basic_data_sets(src_spp, trg_spp, pair)
        return train_count

    def _delete_files(self, pattern: str) -> None:
        for old_file_path in self.exp_dir.glob(pattern):
            old_file_path.unlink()

    def _write_scripture_data_sets(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        pair: CorpusPair,
        stats: bool,
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

        test_indices: Optional[Set[int]] = None
        val_indices: Optional[Set[int]] = None

        train: Optional[pd.DataFrame] = None
        val: Dict[Tuple[str, str], pd.DataFrame] = {}
        test: Dict[Tuple[str, str], pd.DataFrame] = {}
        pair_val_indices: Dict[Tuple[str, str], Set[int]] = {}
        pair_test_indices: Dict[Tuple[str, str], Set[int]] = {}
        terms: Optional[pd.DataFrame] = None

        if pair.use_test_set_from != "":
            self._populate_pair_test_indices(pair.use_test_set_from, pair_test_indices)

        with ExitStack() as stack:
            stats_file: Optional[TextIO] = None
            if stats and pair.size < self.stats_max_size:
                stats_file = stack.enter_context(self._open_append("corpus-stats.csv"))
                if stats_file.tell() == 0:
                    stats_file.write("src_project,trg_project,count,align_score,filtered_count,filtered_align_score\n")

            for src_file, trg_file in get_data_file_pairs(pair):
                corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path)
                if len(pair.corpus_books) > 0:
                    cur_train = include_books(corpus, pair.corpus_books)
                    if len(pair.corpus_books.intersection(pair.test_books)) > 0:
                        cur_train = exclude_books(cur_train, pair.test_books)
                elif len(pair.test_books) > 0:
                    cur_train = exclude_books(corpus, pair.test_books)
                else:
                    cur_train = corpus

                corpus_count = len(cur_train)
                if pair.is_train and (stats_file is not None or pair.score_threshold > 0):
                    LOGGER.info("Computing alignment scores")
                    add_alignment_scores(cur_train)
                    if stats_file is not None:
                        cur_train.to_csv(self.exp_dir / f"{src_file.project}_{trg_file.project}.csv", index=False)

                tags_str = self._get_tags_str(pair.tags, trg_file.iso)
                mirror_tags_str = self._get_tags_str(pair.tags, src_file.iso)

                if pair.is_test:
                    if pair.disjoint_test and test_indices is None:
                        indices: Set[int] = set(cur_train.index)
                        if pair.disjoint_val and val_indices is not None:
                            indices.difference_update(val_indices)
                        split_size = test_size
                        if isinstance(split_size, float):
                            split_size = int(split_size if split_size > 1 else corpus_count * split_size)
                        test_indices = set(random.sample(indices, min(split_size, len(indices))))

                    if len(pair.test_books) > 0:
                        cur_test = include_books(corpus, pair.test_books)
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
                    self._add_to_eval_dataset(
                        src_file.iso,
                        trg_file.iso,
                        trg_file.project,
                        tags_str,
                        test,
                        pair_test_indices,
                        cur_test,
                    )

                if pair.is_train:
                    alignment_score = mean(cur_train["score"]) if stats_file is not None else 0

                    filtered_count = 0
                    filtered_alignment_score = alignment_score
                    if pair.score_threshold > 0:
                        unfiltered_len = len(cur_train)
                        cur_train = filter_parallel_corpus(cur_train, pair.score_threshold)
                        filtered_count = unfiltered_len - len(cur_train)
                        filtered_alignment_score = mean(cur_train["score"]) if stats_file is not None else 0

                    if stats_file is not None:
                        LOGGER.info(
                            f"{src_file.project} -> {trg_file.project} stats -"
                            f" count: {corpus_count},"
                            f" alignment: {alignment_score:.4f},"
                            f" filtered count: {filtered_count},"
                            f" alignment (filtered): {filtered_alignment_score:.4f}"
                        )
                        stats_file.write(
                            f"{src_file.project},{trg_file.project},{corpus_count},{alignment_score:.4f},"
                            f"{filtered_count},{filtered_alignment_score:.4f}\n"
                        )
                    cur_train.drop("score", axis=1, inplace=True, errors="ignore")

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

                    self._add_to_eval_dataset(
                        src_file.iso, trg_file.iso, trg_file.project, tags_str, val, pair_val_indices, cur_val
                    )

                if pair.is_train:
                    if self.mirror:
                        mirror_cur_train = cur_train.rename(columns={"source": "target", "target": "source"})
                        train = self._add_to_train_dataset(
                            trg_file.project,
                            src_file.project,
                            pair.mapping == DataFileMapping.MIXED_SRC,
                            mirror_tags_str,
                            train,
                            mirror_cur_train,
                        )

                    train = self._add_to_train_dataset(
                        src_file.project,
                        trg_file.project,
                        pair.mapping == DataFileMapping.MIXED_SRC,
                        tags_str,
                        train,
                        cur_train,
                    )

        terms_config = self.data["terms"]
        if terms_config["train"] or terms_config["dictionary"]:
            categories: Optional[Union[str, List[str]]] = terms_config["categories"]
            if isinstance(categories, str):
                categories = [cat.strip() for cat in categories.split(",")]
            if categories is None or len(categories) > 0:
                categories_set: Optional[Set[str]] = None if categories is None else set(categories)
                dict_books = get_books(terms_config["dictionary_books"]) if "dictionary_books" in terms_config else None
                all_src_terms = [
                    (src_terms_file, get_terms(src_terms_file.path, iso=src_terms_file.iso))
                    for src_terms_file in pair.src_terms_files
                ]
                all_trg_terms = [
                    (trg_terms_file, get_terms(trg_terms_file.path, iso=trg_terms_file.iso))
                    for trg_terms_file in pair.trg_terms_files
                ]
                for src_terms_file, src_terms in all_src_terms:
                    for trg_terms_file, trg_terms in all_trg_terms:
                        if src_terms_file.iso == trg_terms_file.iso:
                            continue
                        tags_str = self._get_tags_str(pair.tags, trg_terms_file.iso)
                        mirror_tags_str = self._get_tags_str(pair.tags, src_terms_file.iso)
                        cur_terms = get_terms_corpus(src_terms, trg_terms, categories_set, dict_books)
                        terms = self._add_to_terms_dataset(tags_str, mirror_tags_str, terms, cur_terms)
                if terms_config["include_glosses"]:
                    if "en" in self.trg_isos:
                        for src_terms_file, src_terms in all_src_terms:
                            cur_terms = get_terms_data_frame(src_terms, categories_set, dict_books)
                            cur_terms = cur_terms.rename(columns={"rendering": "source", "gloss": "target"})
                            tags_str = self._get_tags_str(pair.tags, "en")
                            mirror_tags_str = self._get_tags_str(pair.tags, src_terms_file.iso)
                            terms = self._add_to_terms_dataset(tags_str, mirror_tags_str, terms, cur_terms)
                    if "en" in self.src_isos:
                        for trg_terms_file, trg_terms in all_trg_terms:
                            cur_terms = get_terms_data_frame(trg_terms, categories_set, dict_books)
                            cur_terms = cur_terms.rename(columns={"rendering": "target", "gloss": "source"})
                            tags_str = self._get_tags_str(pair.tags, trg_terms_file.iso)
                            mirror_tags_str = self._get_tags_str(pair.tags, "en")
                            terms = self._add_to_terms_dataset(tags_str, mirror_tags_str, terms, cur_terms)

        if train is None:
            return 0

        if pair.mapping == DataFileMapping.MIXED_SRC:
            train.fillna("", inplace=True)
            src_columns: List[str] = [c for c in train.columns if c.startswith("source")]

            def select_random_column(row: Any) -> pd.Series:
                nonempty_src_columns: List[str] = [c for c in src_columns if row[c] != ""]
                return row[random.choice(nonempty_src_columns)]

            train["source"] = train[src_columns].apply(select_random_column, axis=1)
            train.drop(src_columns, axis=1, inplace=True, errors="ignore")

        self._append_corpus(self._train_src_filename(), encode_sp_lines(src_spp, train["source"]))
        self._append_corpus(self._train_trg_filename(), encode_sp_lines(trg_spp, train["target"]))
        self._append_corpus(self._train_vref_filename(), (str(vr) for vr in train["vref"]))
        train_count = len(train)

        terms_train_count, dict_count = self._write_terms(src_spp, trg_spp, terms)
        train_count += terms_train_count

        val_count = 0
        if len(val) > 0:
            val_src = itertools.chain.from_iterable(pair_val["source"] for pair_val in val.values())
            self._append_corpus(self._val_src_filename(), encode_sp_lines(src_spp, val_src))
            val_count = sum(len(pair_val) for pair_val in val.values())
            self._write_val_corpora(trg_spp, val)
            val_vref = itertools.chain.from_iterable(pair_val["vref"] for pair_val in val.values())
            self._append_corpus(self._val_vref_filename(), (str(vr) for vr in val_vref))

        test_count = 0
        for (src_iso, trg_iso), pair_test in test.items():
            self._append_corpus(self._test_vref_filename(src_iso, trg_iso), (str(vr) for vr in pair_test["vref"]))
            self._append_corpus(
                self._test_src_filename(src_iso, trg_iso), encode_sp_lines(src_spp, pair_test["source"])
            )
            test_count += len(pair_test)

            columns: List[str] = [c for c in pair_test.columns if c.startswith("target")]
            test_projects = self._get_test_projects(src_iso, trg_iso)
            for column in columns:
                project = column[len("target_") :]
                self._append_corpus(
                    self._test_trg_filename(src_iso, trg_iso, project),
                    decode_sp_lines(encode_sp_lines(trg_spp, pair_test[column])),
                )
                test_projects.remove(project)
            for project in test_projects:
                self._fill_corpus(self._test_trg_filename(src_iso, trg_iso, project), len(pair_test))
        LOGGER.info(
            f"train size: {train_count},"
            f" val size: {val_count},"
            f" test size: {test_count},"
            f" dict size: {dict_count},"
            f" terms train size: {terms_train_count}"
        )
        return train_count

    def _populate_pair_test_indices(self, exp_name: str, pair_test_indices: Dict[Tuple[str, str], Set[int]]) -> None:
        vrefs: Dict[str, int] = {}
        for i, vref_str in enumerate(load_corpus(SIL_NLP_ENV.mt_scripture_dir / "vref.txt")):
            if vref_str != "":
                vrefs[vref_str] = i

        exp_dir = get_mt_exp_dir(exp_name)
        for vref_path in exp_dir.glob("test*.vref.txt"):
            stem = vref_path.stem
            test_indices: Set[int] = set()
            if stem == "test.vref":
                pair_test_indices[(self.default_src_iso, self.default_trg_iso)] = test_indices
            else:
                _, src_iso, trg_iso, _ = stem.split(".", maxsplit=4)
                pair_test_indices[(src_iso, trg_iso)] = test_indices

            for vref_str in load_corpus(vref_path):
                vref = VerseRef.from_string(vref_str)
                if vref.has_multiple:
                    vref.simplify()
                test_indices.add(vrefs[str(vref)])

    def _add_to_eval_dataset(
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

    def _add_to_train_dataset(
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
            cur_train.rename(columns={"source": f"source_{src_project}"}, inplace=True)
            cur_train.set_index(
                pd.MultiIndex.from_tuples(
                    map(lambda i: (trg_project, i), cur_train.index), names=["trg_project", "index"]
                ),
                inplace=True,
            )
            train = cur_train if train is None else train.combine_first(cur_train)
        else:
            train = pd.concat([train, cur_train], ignore_index=True)
        return train

    def _insert_tags(self, tags_str: str, sentences: pd.DataFrame) -> None:
        if tags_str != "":
            cast(Any, sentences).loc[:, "source"] = tags_str + sentences.loc[:, "source"]

    def _add_to_terms_dataset(
        self, tags_str: str, mirror_tags_str: str, terms: Optional[pd.DataFrame], cur_terms: pd.DataFrame
    ) -> pd.DataFrame:
        if self.mirror:
            mirror_cur_terms = cur_terms.rename(columns={"source": "target", "target": "source"})
            self._insert_tags(mirror_tags_str, mirror_cur_terms)
            terms = pd.concat([terms, mirror_cur_terms], ignore_index=True)

        self._insert_tags(tags_str, cur_terms)

        return pd.concat([terms, cur_terms], ignore_index=True)

    def _write_terms(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        terms: Optional[pd.DataFrame],
    ) -> Tuple[int, int]:
        train_count = 0
        dict_count = 0
        with ExitStack() as stack:
            terms_config = self.data["terms"]
            train_src_file: Optional[TextIO] = None
            train_trg_file: Optional[TextIO] = None
            train_vref_file: Optional[TextIO] = None
            if terms_config["train"] and terms is not None:
                train_src_file = stack.enter_context(self._open_append(self._train_src_filename()))
                train_trg_file = stack.enter_context(self._open_append(self._train_trg_filename()))
                train_vref_file = stack.enter_context(self._open_append(self._train_vref_filename()))

            dict_src_file: Optional[TextIO] = None
            dict_trg_file: Optional[TextIO] = None
            if terms_config["dictionary"]:
                dict_src_file = stack.enter_context(self._open_append(self._dict_src_filename()))
                dict_trg_file = stack.enter_context(self._open_append(self._dict_trg_filename()))

            if terms is not None:
                for _, term in terms.iterrows():
                    src_term = cast(str, term["source"])
                    trg_term = cast(str, term["target"])
                    dictionary = cast(bool, term["dictionary"])
                    src_term_variants = [
                        encode_sp(src_spp, src_term, add_dummy_prefix=True),
                        encode_sp(src_spp, src_term, add_dummy_prefix=False),
                    ]
                    trg_term_variants = [
                        encode_sp(trg_spp, trg_term, add_dummy_prefix=True),
                        encode_sp(trg_spp, trg_term, add_dummy_prefix=False),
                    ]

                    if train_src_file is not None and train_trg_file is not None and train_vref_file is not None:
                        for stv, ttv in zip(src_term_variants, trg_term_variants):
                            train_src_file.write(stv + "\n")
                            train_trg_file.write(ttv + "\n")
                            train_vref_file.write("\n")
                            train_count += 1

                    if dictionary and dict_src_file is not None and dict_trg_file is not None:
                        dict_src_file.write("\t".join(src_term_variants) + "\n")
                        dict_trg_file.write("\t".join(trg_term_variants) + "\n")
                        dict_count += 1
        return train_count, dict_count

    def _write_val_corpora(
        self, trg_spp: Optional[sp.SentencePieceProcessor], val: Dict[Tuple[str, str], pd.DataFrame]
    ) -> None:
        with ExitStack() as stack:
            ref_files: List[TextIO] = []
            for (src_iso, trg_iso), pair_val in val.items():
                columns: List[str] = [c for c in pair_val.columns if c.startswith("target")]
                if self.root["eval"]["multi_ref_eval"]:
                    val_project_count = self._get_val_ref_count(src_iso, trg_iso)
                    for index in pair_val.index:
                        for ci in range(val_project_count):
                            if len(ref_files) == ci:
                                ref_files.append(stack.enter_context(self._open_append(self._val_trg_filename(ci))))
                            if ci < len(columns):
                                col = columns[ci]
                                ref_files[ci].write(
                                    encode_sp(trg_spp, cast(str, pair_val.loc[index, col]).strip()) + "\n"
                                )
                            else:
                                ref_files[ci].write("\n")
                else:
                    for index in pair_val.index:
                        if len(ref_files) == 0:
                            ref_files.append(stack.enter_context(self._open_append(self._val_trg_filename())))
                        columns_with_data = [c for c in columns if cast(str, pair_val.loc[index, c]).strip() != ""]
                        col = random.choice(columns_with_data)
                        ref_files[0].write(encode_sp(trg_spp, cast(str, pair_val.loc[index, col]).strip()) + "\n")

    def _append_corpus(self, filename: str, sentences: Iterable[str]) -> None:
        write_corpus(self.exp_dir / filename, sentences, append=True)

    def _fill_corpus(self, filename: str, size: int) -> None:
        write_corpus(self.exp_dir / filename, ("" for _ in range(size)), append=True)

    def _open_append(self, filename: str) -> TextIO:
        return (self.exp_dir / filename).open("a", encoding="utf-8", newline="\n")

    def _write_basic_data_sets(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        pair: CorpusPair,
    ) -> int:
        total_train_count = 0
        for src_file, trg_file in zip(pair.src_files, pair.trg_files):
            total_train_count += self._write_basic_data_file_pair(src_spp, trg_spp, pair, src_file, trg_file)
        return total_train_count

    def _write_basic_data_file_pair(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        pair: CorpusPair,
        src_file: DataFile,
        trg_file: DataFile,
    ) -> int:
        LOGGER.info(f"Preprocessing {src_file.path.stem} -> {trg_file.path.stem}")
        corpus_size = get_parallel_corpus_size(src_file.path, trg_file.path)
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

            train_count = 0
            val_count = 0
            test_count = 0
            dict_count = 0
            tags_str = self._get_tags_str(pair.tags, trg_file.iso)
            mirror_tags_str = self._get_tags_str(pair.tags, src_file.iso)

            train_src_file = stack.enter_context(self._open_append(self._train_src_filename()))
            train_trg_file = stack.enter_context(self._open_append(self._train_trg_filename()))
            val_src_file = stack.enter_context(self._open_append(self._val_src_filename()))
            val_trg_file = stack.enter_context(self._open_append(self._val_trg_filename()))
            test_src_file = stack.enter_context(self._open_append(self._test_src_filename(src_file.iso, trg_file.iso)))
            test_trg_file = stack.enter_context(self._open_append(self._test_trg_filename(src_file.iso, trg_file.iso)))

            train_vref_file: Optional[TextIO] = None
            val_vref_file: Optional[TextIO] = None
            test_vref_file: Optional[TextIO] = None
            test_trg_project_files: List[TextIO] = []
            val_trg_ref_files: List[TextIO] = []
            dict_src_file: Optional[TextIO] = None
            dict_trg_file: Optional[TextIO] = None
            if self._has_scripture_data:
                train_vref_file = stack.enter_context(self._open_append(self._train_vref_filename()))
                val_vref_file = stack.enter_context(self._open_append(self._val_vref_filename()))
                test_vref_file = stack.enter_context(
                    self._open_append(self._test_vref_filename(src_file.iso, trg_file.iso))
                )
                test_projects = self._get_test_projects(src_file.iso, trg_file.iso)
                test_trg_project_files = [
                    stack.enter_context(self._open_append(self._test_trg_filename(src_file.iso, trg_file.iso, project)))
                    for project in test_projects
                    if project != BASIC_DATA_PROJECT
                ]
                val_ref_count = self._get_val_ref_count(src_file.iso, trg_file.iso)
                val_trg_ref_files = [
                    stack.enter_context(self._open_append(self._val_trg_filename(index)))
                    for index in range(1, val_ref_count)
                ]
            if pair.is_dictionary:
                dict_src_file = stack.enter_context(self._open_append(self._dict_src_filename()))
                dict_trg_file = stack.enter_context(self._open_append(self._dict_trg_filename()))

            index = 0
            for src_line, trg_line in zip(input_src_file, input_trg_file):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                if len(src_line) == 0 or len(trg_line) == 0:
                    continue

                src_sentence = tags_str + src_line
                trg_sentence = trg_line

                if pair.is_test and (test_indices is None or index in test_indices):
                    test_src_file.write(encode_sp(src_spp, src_sentence) + "\n")
                    test_trg_file.write(decode_sp(encode_sp(trg_spp, trg_sentence)) + "\n")
                    if test_vref_file is not None:
                        test_vref_file.write("\n")
                    for test_trg_project_file in test_trg_project_files:
                        test_trg_project_file.write("\n")
                    test_count += 1
                elif pair.is_val and (val_indices is None or index in val_indices):
                    val_src_file.write(encode_sp(src_spp, src_sentence) + "\n")
                    val_trg_file.write(encode_sp(trg_spp, trg_sentence) + "\n")
                    if val_vref_file is not None:
                        val_vref_file.write("\n")
                    for val_trg_ref_file in val_trg_ref_files:
                        val_trg_ref_file.write("\n")
                    val_count += 1
                elif pair.is_train and (train_indices is None or index in train_indices):
                    noised_src_sentence = self._noise(pair.src_noise, src_sentence)
                    train_count += self._write_train_sentence_pair(
                        train_src_file,
                        train_trg_file,
                        train_vref_file,
                        src_spp,
                        trg_spp,
                        noised_src_sentence,
                        trg_sentence,
                        pair.is_lexical_data,
                    )
                    if self.mirror:
                        mirror_src_sentence = mirror_tags_str + trg_line
                        mirror_trg_sentence = src_line
                        mirror_src_spp = trg_spp
                        mirror_trg_spp = src_spp
                        train_count += self._write_train_sentence_pair(
                            train_src_file,
                            train_trg_file,
                            train_vref_file,
                            mirror_src_spp,
                            mirror_trg_spp,
                            mirror_src_sentence,
                            mirror_trg_sentence,
                            pair.is_lexical_data,
                        )

                if pair.is_dictionary and dict_src_file is not None and dict_trg_file is not None:
                    src_variants = [
                        encode_sp(src_spp, src_sentence, add_dummy_prefix=True),
                        encode_sp(src_spp, src_sentence, add_dummy_prefix=False),
                    ]
                    trg_variants = [
                        encode_sp(trg_spp, trg_sentence, add_dummy_prefix=True),
                        encode_sp(trg_spp, trg_sentence, add_dummy_prefix=False),
                    ]
                    dict_src_file.write("\t".join(src_variants) + "\n")
                    dict_trg_file.write("\t".join(trg_variants) + "\n")
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
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        src_sentence: str,
        trg_sentence: str,
        is_lexical: bool,
    ) -> int:
        src_variants = [encode_sp(src_spp, src_sentence, add_dummy_prefix=True)]
        trg_variants = [encode_sp(trg_spp, trg_sentence, add_dummy_prefix=True)]
        if is_lexical:
            src_variants.append(encode_sp(src_spp, src_sentence, add_dummy_prefix=False))
            trg_variants.append(encode_sp(trg_spp, trg_sentence, add_dummy_prefix=False))
        for src_variant, trg_variant in zip(src_variants, trg_variants):
            src_file.write(src_variant + "\n")
            trg_file.write(trg_variant + "\n")
            if vref_file is not None:
                vref_file.write("\n")
        return len(src_variants)

    def _get_tags_str(self, tags: List[str], trg_iso: str) -> str:
        tags_str = ""
        if len(tags) > 0:
            tags_str += " ".join(f"<{t}>" for t in tags) + " "
        if self.write_trg_tag:
            tags_str += f"<2{trg_iso}> "
        return tags_str

    def _noise(self, src_noise: List[NoiseMethod], src_sentence: str) -> str:
        if len(src_noise) == 0:
            return src_sentence
        tokens = src_sentence.split()
        tag: List[str] = []
        if self.write_trg_tag:
            tag = tokens[:1]
            tokens = tokens[1:]
        for noise_method in src_noise:
            tokens = noise_method(tokens)
        if self.write_trg_tag:
            tokens = tag + tokens
        return " ".join(tokens)

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

    def _train_src_filename(self) -> str:
        return "train.src.txt"

    def _train_trg_filename(self) -> str:
        return "train.trg.txt"

    def _train_vref_filename(self) -> str:
        return "train.vref.txt"

    def _val_src_filename(self) -> str:
        return "val.src.txt"

    def _val_vref_filename(self) -> str:
        return "val.vref.txt"

    def _val_trg_filename(self, index: int = 0) -> str:
        return f"val.trg.txt.{index}" if self.root["eval"]["multi_ref_eval"] else "val.trg.txt"

    def _test_src_filename(self, src_iso: str, trg_iso: str) -> str:
        return f"test.{src_iso}.{trg_iso}.src.txt" if self._multiple_test_iso_pairs else "test.src.txt"

    def _test_vref_filename(self, src_iso: str, trg_iso: str) -> str:
        return f"test.{src_iso}.{trg_iso}.vref.txt" if self._multiple_test_iso_pairs else "test.vref.txt"

    def _test_trg_filename(self, src_iso: str, trg_iso: str, project: str = BASIC_DATA_PROJECT) -> str:
        prefix = f"test.{src_iso}.{trg_iso}" if self._multiple_test_iso_pairs else "test"
        has_multiple_test_projects = self._iso_pairs[(src_iso, trg_iso)].has_multiple_test_projects
        suffix = f".{project}" if has_multiple_test_projects else ""
        return f"{prefix}.trg.detok{suffix}.txt"

    def _dict_src_filename(self) -> str:
        return "dict.src.txt"

    def _dict_trg_filename(self) -> str:
        return "dict.trg.txt"

    def _build_vocabs(self) -> None:
        if not self.data["tokenize"]:
            return

        if self.share_vocab:
            LOGGER.info("Building shared vocabulary...")
            vocab_size: Optional[int] = self.data.get("vocab_size")
            if vocab_size is None:
                vocab_size = self.data.get("src_vocab_size")
                if vocab_size is None:
                    vocab_size = self.data["trg_vocab_size"]
                elif self.data.get("trg_vocab_size", vocab_size) != vocab_size:
                    raise RuntimeError(
                        "The source and target vocab sizes cannot be different when creating a shared vocab."
                    )
            assert vocab_size is not None

            casing: Optional[str] = self.data.get("casing")
            if casing is None:
                casing = self.data.get("src_casing")
                if casing is None:
                    casing = self.data["trg_casing"]
                elif self.data.get("trg_casing", casing) != casing:
                    raise RuntimeError("The source and target casing cannot be different when creating a shared vocab.")
            assert casing is not None

            model_prefix = self.exp_dir / "sp"
            vocab_path = self.exp_dir / "onmt.vocab"
            share_vocab_file_paths: Set[Path] = self.src_file_paths | self.trg_file_paths
            character_coverage = self.data.get("character_coverage", 1.0)
            build_vocab(
                share_vocab_file_paths, vocab_size, casing, character_coverage, model_prefix, vocab_path, self._tags
            )

            self._update_vocab(vocab_path, vocab_path)
        else:
            src_vocab_file_paths: Set[Path] = set(self.src_file_paths)
            if self.mirror:
                src_vocab_file_paths.update(self.trg_file_paths)
            self._create_unshared_vocab(self.src_isos, src_vocab_file_paths, "source")

            trg_vocab_file_paths: Set[Path] = set(self.trg_file_paths)
            if self.mirror:
                trg_vocab_file_paths.update(self.src_file_paths)
            self._create_unshared_vocab(self.trg_isos, trg_vocab_file_paths, "target")

            self._update_vocab(self.exp_dir / "src-onmt.vocab", self.exp_dir / "trg-onmt.vocab")

    def _update_vocab(self, src_vocab_path: Path, trg_vocab_path: Path) -> None:
        if self.parent_config is None:
            return

        model_dir = SIL_NLP_ENV.get_source_experiment_path(self.parent_config.model_dir)
        parent_model_to_use = (
            CheckpointType.BEST
            if self.data["parent_use_best"]
            else CheckpointType.AVERAGE
            if self.data["parent_use_average"]
            else CheckpointType.LAST
        )
        checkpoint_path, step = get_checkpoint_path(model_dir, parent_model_to_use)
        if checkpoint_path is not None:
            SIL_NLP_ENV.copy_experiment_from_bucket(checkpoint_path.parent)
            checkpoint_path = SIL_NLP_ENV.get_temp_experiment_path(checkpoint_path)
        parent_runner = create_runner(self.parent_config)
        parent_runner.update_vocab(
            str(self.exp_dir / "parent"),
            str(src_vocab_path),
            str(trg_vocab_path),
            None if checkpoint_path is None else str(checkpoint_path),
            step,
        )

    def _create_unshared_vocab(self, isos: Set[str], vocab_file_paths: Set[Path], side: str) -> None:
        prefix = "src" if side == "source" else "trg"
        model_prefix = self.exp_dir / f"{prefix}-sp"
        vocab_path = self.exp_dir / f"{prefix}-onmt.vocab"
        if self.parent_config is not None:
            parent_isos = self.parent_config.src_isos if side == "source" else self.parent_config.trg_isos
            if isos == parent_isos:
                if self.parent_config.share_vocab:
                    parent_sp_prefix_path = self.parent_config.exp_dir / "sp"
                    parent_vocab_path = self.parent_config.exp_dir / "onmt.vocab"
                else:
                    parent_sp_prefix_path = self.parent_config.exp_dir / f"{prefix}-sp"
                    parent_vocab_path = self.parent_config.exp_dir / f"{prefix}-onmt.vocab"

                parent_vocab: Optional[Vocab] = None
                child_tokens: Optional[Set[str]] = None
                parent_use_vocab: bool = self.data["parent_use_vocab"]
                if not parent_use_vocab:
                    parent_spp = sp.SentencePieceProcessor()
                    parent_spp.Load(str(parent_sp_prefix_path.with_suffix(".model")))

                    parent_vocab = Vocab()
                    parent_vocab.load(str(parent_vocab_path))

                    child_tokens = set()
                    for vocab_file_path in vocab_file_paths:
                        for line in encode_sp_lines(parent_spp, load_corpus(vocab_file_path)):
                            child_tokens.update(line.split())
                    parent_use_vocab = child_tokens.issubset(parent_vocab.words)

                # all tokens in the child corpora are in the parent vocab, so we can just use the parent vocab
                # or, the user wants to reuse the parent vocab for this child experiment
                if parent_use_vocab:
                    sp_vocab_path = self.exp_dir / f"{prefix}-sp.vocab"
                    onmt_vocab_path = self.exp_dir / f"{prefix}-onmt.vocab"
                    shutil.copy2(parent_sp_prefix_path.with_suffix(".model"), self.exp_dir / f"{prefix}-sp.model")
                    shutil.copy2(parent_sp_prefix_path.with_suffix(".vocab"), sp_vocab_path)
                    convert_vocab(sp_vocab_path, onmt_vocab_path)
                    return
                elif child_tokens is not None and parent_vocab is not None:
                    onmt_delta_vocab_path = self.exp_dir / f"{prefix}-onmt-delta.vocab"
                    vocab_delta = child_tokens.difference(parent_vocab.words)
                    with onmt_delta_vocab_path.open("w", encoding="utf-8", newline="\n") as f:
                        [f.write(f"{token}\n") for token in vocab_delta]

        LOGGER.info(f"Building {side} vocabulary...")
        vocab_size: int = self.data.get(f"{prefix}_vocab_size", self.data.get("vocab_size"))
        casing: str = self.data.get(f"{prefix}_casing", self.data.get("casing"))
        character_coverage: float = self.data.get(f"{prefix}_character_coverage", self.data.get("character_coverage"))
        tags = self._tags if side == "source" else set()
        build_vocab(vocab_file_paths, vocab_size, casing, character_coverage, model_prefix, vocab_path, tags)

    def _create_train_alignments(self, train_count: int) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            aligner = FastAlignMachineAligner(temp_dir)
            aligner.lowercase = True

            src_align_path = self.exp_dir / "train.src.txt"
            trg_align_path = self.exp_dir / "train.trg.txt"
            train_size: Union[int, float] = self.data["guided_alignment_train_size"]
            split_indices = split_corpus(train_count, train_size)
            if split_indices is not None:
                # reduce size of alignment training data
                src_train_path = temp_dir / "train.src.align.txt"
                trg_train_path = temp_dir / "train.trg.align.txt"

                with src_align_path.open("r", encoding="utf-8-sig") as src_in_file, trg_align_path.open(
                    "r", encoding="utf-8-sig"
                ) as trg_in_file, src_train_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as src_out_file, trg_train_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as trg_out_file:
                    i = 0
                    for src_sentence, trg_sentence in zip(src_in_file, trg_in_file):
                        if i in split_indices:
                            src_out_file.write(src_sentence)
                            trg_out_file.write(trg_sentence)
                        i += 1
            else:
                src_train_path = src_align_path
                trg_train_path = trg_align_path

            aligner.train(src_train_path, trg_train_path)
            aligner.force_align(src_align_path, trg_align_path, self.exp_dir / "train.alignments.txt")


class SILWordNoiser(WordNoiser):
    def __init__(
        self,
        noises: Optional[List[Noise]] = None,
        subword_token: str = "￭",
        is_spacer: Optional[bool] = None,
        has_lang_tag: bool = False,
    ) -> None:
        super().__init__(noises=noises, subword_token=subword_token, is_spacer=is_spacer)
        self.has_lang_tag = has_lang_tag

    def _call(
        self, tokens: tf.Tensor, sequence_length: Optional[Union[int, tf.Tensor]], keep_shape: bool
    ) -> Tuple[tf.Tensor, Union[int, tf.Tensor]]:
        rank = tokens.shape.rank
        if rank == 1:
            input_length = tf.shape(tokens)[0]
            if sequence_length is not None:
                tokens = tokens[:sequence_length]
            else:
                tokens = tokens[: tf.math.count_nonzero(tokens)]
            words = tokens_to_words(tokens, subword_token=self.subword_token, is_spacer=self.is_spacer)
            words = cast(tf.Tensor, words.to_tensor())
            tag: Optional[str] = None
            if self.has_lang_tag:
                tag = words[:1]
                words = words[1:]
            for noise in self.noises:
                words = noise(words)
            if tag is not None:
                words = tf.concat([tag, words], axis=0)
            outputs = tf.RaggedTensor.from_tensor(words, padding="").flat_values
            output_length = tf.shape(outputs)[0]
            if keep_shape:
                outputs = tf.pad(outputs, [[0, input_length - output_length]])
            return outputs, output_length
        else:
            return super()._call(tokens, sequence_length=sequence_length, keep_shape=keep_shape)


def create_model(config: Config) -> Model:
    model_name = config.model
    if model_name.startswith("Transformer"):
        model_name = "SIL" + model_name
    model = get_model_from_catalog(model_name)
    if isinstance(model, SILTransformer):
        dropout = config.params["transformer_dropout"]
        attention_dropout = config.params["transformer_attention_dropout"]
        ffn_dropout = config.params["transformer_ffn_dropout"]
        if dropout != 0.1 or attention_dropout != 0.1 or ffn_dropout != 0.1:
            model.set_dropout(dropout=dropout, attention_dropout=attention_dropout, ffn_dropout=ffn_dropout)

    word_dropout: float = config.params["word_dropout"]
    if word_dropout > 0:
        source_noiser = SILWordNoiser(subword_token="▁", has_lang_tag=config.write_trg_tag)
        source_noiser.add(WordDropout(word_dropout))
        cast(TextInputter, model.features_inputter).set_noise(source_noiser, probability=1.0)

        target_noiser = SILWordNoiser(subword_token="▁")
        target_noiser.add(WordDropout(word_dropout))
        cast(TextInputter, model.labels_inputter).set_noise(target_noiser, probability=1.0)
    return model


def set_tf_log_level(log_level: int = logging.INFO) -> None:
    tf.get_logger().setLevel(log_level)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL[log_level])


def create_runner(config: Config, mixed_precision: bool = False, memory_growth: bool = False) -> SILRunner:
    set_tf_log_level()

    if memory_growth:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, enable=True)

    model = create_model(config)

    return SILRunner(model, config.root, auto_config=True, mixed_precision=mixed_precision)


def load_config(exp_name: str) -> Config:
    exp_dir = get_mt_exp_dir(exp_name)
    config_path = exp_dir / "config.yml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return Config(exp_dir, config)


def copy_config_value(src: dict, trg: dict, key: str) -> None:
    if key in src:
        trg[key] = src[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Creates a NMT experiment config file")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--src", nargs="*", metavar="corpus", default=[], help="Source corpora")
    parser.add_argument("--trg", nargs="*", metavar="corpus", default=[], help="Target corpora")
    parser.add_argument("--vocab-size", type=int, help="Shared vocabulary size")
    parser.add_argument("--src-vocab-size", type=int, help="Source vocabulary size")
    parser.add_argument("--trg-vocab-size", type=int, help="Target vocabulary size")
    parser.add_argument("--parent", type=str, help="Parent experiment name")
    parser.add_argument("--mirror", default=False, action="store_true", help="Mirror train and validation data sets")
    parser.add_argument("--force", default=False, action="store_true", help="Overwrite existing config file")
    parser.add_argument("--seed", type=int, help="Randomization seed")
    parser.add_argument("--model", type=str, help="The neural network model")
    args = parser.parse_args()

    get_git_revision_hash()

    if len(args.src) != len(args.trg):
        raise RuntimeError("The number of source and target corpora must be the same.")

    exp_dir = get_mt_exp_dir(args.experiment)
    config_path = exp_dir / "config.yml"
    if config_path.is_file() and not args.force:
        LOGGER.warning(
            f'The experiment config file {config_path} already exists. Use "--force" if you want to overwrite the existing config.'
        )
        return

    exp_dir.mkdir(exist_ok=True, parents=True)

    config = _DEFAULT_NEW_CONFIG.copy()
    if args.model is not None:
        config["model"] = args.model
    data_config: dict = config["data"]
    corpus_pairs: List[dict] = []
    for src_corpus, trg_corpus in zip(args.src, args.trg):
        corpus_pairs.append({"src": src_corpus, "trg": trg_corpus})
    data_config["corpus_pairs"] = corpus_pairs
    if args.parent is not None:
        data_config["parent"] = args.parent
        SIL_NLP_ENV.copy_experiment_from_bucket(args.parent, extensions="config.yml")
        parent_config = load_config(args.parent)
        for key in [
            "share_vocab",
            "vocab_size",
            "src_vocab_size",
            "trg_vocab_size",
            "casing",
            "src_casing",
            "trg_casing",
        ]:
            if key in parent_config.data:
                data_config[key] = parent_config.data[key]
    if args.vocab_size is not None:
        data_config["vocab_size"] = args.vocab_size
    elif args.src_vocab_size is not None or args.trg_vocab_size is not None:
        data_config["share_vocab"] = False
        if args.src_vocab_size is not None:
            data_config["src_vocab_size"] = args.src_vocab_size
        elif "vocab_size" in data_config:
            data_config["src_vocab_size"] = data_config["vocab_size"]
            del data_config["vocab_size"]
        if args.trg_vocab_size is not None:
            data_config["trg_vocab_size"] = args.trg_vocab_size
        elif "vocab_size" in data_config:
            data_config["trg_vocab_size"] = data_config["vocab_size"]
            del data_config["vocab_size"]
    if data_config["share_vocab"]:
        if "vocab_size" not in data_config:
            data_config["vocab_size"] = 24000
        if "casing" not in data_config:
            data_config["casing"] = "lower"
    else:
        if "src_vocab_size" not in data_config:
            data_config["src_vocab_size"] = 8000
        if "trg_vocab_size" not in data_config:
            data_config["trg_vocab_size"] = 8000
        if "src_casing" not in data_config:
            data_config["src_casing"] = data_config.get("casing", "lower")
        if "trg_casing" not in data_config:
            data_config["trg_casing"] = data_config.get("casing", "lower")
        if "casing" in data_config:
            del data_config["casing"]
    if args.seed is not None:
        data_config["seed"] = args.seed
    if args.mirror:
        data_config["mirror"] = True
    with config_path.open("w", encoding="utf-8") as file:
        yaml.dump(config, file)
    LOGGER.info(f"Config file created: {config_path}")


if __name__ == "__main__":
    main()
