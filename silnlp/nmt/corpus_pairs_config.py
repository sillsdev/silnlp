import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, TextIO, Tuple, Type, Union

import pandas as pd
import sentencepiece as sp

from ..common.corpus import (
    Term,
    get_scripture_parallel_corpus,
    get_terms,
    get_terms_corpus,
    get_terms_data_frame,
    get_terms_glosses_path,
    get_terms_list,
    get_terms_renderings_path,
    parse_scripture_path,
    split_corpus,
)
from ..common.environment import SNE
from ..common.utils import (
    DeleteRandomToken,
    NoiseMethod,
    RandomTokenPermutation,
    ReplaceRandomToken,
    is_set,
    merge_dict,
)
from .config import Config, DataFileType
from .utils import decode_sp, encode_sp

LOGGER = logging.getLogger(__name__)


def parse_iso(file_path: Path) -> str:
    file_name = file_path.name
    index = file_name.index("-")
    return file_name[:index]


class CorpusPair:
    def __init__(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        type: DataFileType,
        src_noise: List[NoiseMethod] = [],
        src_tags: List[str] = [],
        size: Union[float, int] = 1.0,
        test_size: Optional[Union[float, int]] = None,
        val_size: Optional[Union[float, int]] = None,
    ) -> None:
        self.src_file_path = src_file_path
        self.src_iso = parse_iso(src_file_path)
        self.trg_file_path = trg_file_path
        self.trg_iso = parse_iso(trg_file_path)
        self.type = type
        self.src_noise = src_noise
        self.src_tags = src_tags
        self.size = size
        self.test_size = test_size
        self.val_size = val_size

    @property
    def is_train(self):
        return is_set(self.type, DataFileType.TRAIN)

    @property
    def is_test(self):
        return is_set(self.type, DataFileType.TEST)

    @property
    def is_val(self):
        return is_set(self.type, DataFileType.VAL)

    @property
    def is_dictionary(self):
        return is_set(self.type, DataFileType.DICT)

    @property
    def is_scripture(self):
        return self.src_file_path.parent == SNE._MT_SCRIPTURE_DIR

    @property
    def is_terms(self):
        return self.src_file_path.parent == SNE._MT_TERMS_DIR


def create_noise_methods(params: List[dict]) -> List[NoiseMethod]:
    if params is None:
        return None
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
    corpus_path = SNE._MT_CORPORA_DIR / f"{corpus}.txt"
    if corpus_path.is_file():
        return corpus_path
    corpus_path = SNE._MT_SCRIPTURE_DIR / f"{corpus}.txt"
    if not corpus_path.is_file():
        LOGGER.warning(f"Could not find file '{corpus}' in either {SNE._MT_CORPORA_DIR} or {SNE._MT_SCRIPTURE_DIR}")
    return corpus_path


def get_terms_paths(data_file_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    iso, project = parse_scripture_path(data_file_path)
    terms_renderings_path = get_terms_renderings_path(iso, project)
    terms_glosses_path: Optional[Path] = None
    if terms_renderings_path is not None:
        list_name = get_terms_list(terms_renderings_path)
        terms_glosses_path = get_terms_glosses_path(list_name)
        if not terms_glosses_path.is_file():
            terms_glosses_path = None
    return (terms_renderings_path, terms_glosses_path)


def parse_corpus_pairs(corpus_pairs: List[dict], terms_config: dict) -> List[CorpusPair]:
    pairs: List[CorpusPair] = []
    terms_type = DataFileType.NONE
    if terms_config["train"]:
        terms_type |= DataFileType.TRAIN
    if terms_config["dictionary"]:
        terms_type |= DataFileType.DICT
    for pair in corpus_pairs:
        type_strs: Optional[Union[str, List[str]]] = pair.get("type")
        if type_strs is None:
            type_strs = ["train", "test", "val"]
        elif isinstance(type_strs, str):
            type_strs = type_strs.split(",")
        type = DataFileType.NONE
        for type_str in type_strs:
            type_str = type_str.strip().lower()
            if type_str == "train":
                type |= DataFileType.TRAIN
            elif type_str == "test":
                type |= DataFileType.TEST
            elif type_str == "val":
                type |= DataFileType.VAL
            elif type_str == "dict" or type_str == "dictionary":
                type |= DataFileType.DICT
        src: str = pair["src"]
        src_file_path = get_corpus_path(src)
        trg: str = pair["trg"]
        trg_file_path = get_corpus_path(trg)
        src_tag_str = pair.get("src_tags")
        if src_tag_str is None:
            src_tags: List[str] = []
        else:
            src_tags = src_tag_str.split(",")
        src_noise = create_noise_methods(pair.get("src_noise", []))
        size: Union[float, int] = pair.get("size", 1.0)
        test_size: Optional[Union[float, int]] = pair.get("test_size")
        if test_size is None and is_set(type, DataFileType.TRAIN | DataFileType.TEST):
            test_size = 250
        val_size: Optional[Union[float, int]] = pair.get("val_size")
        if val_size is None and is_set(type, DataFileType.TRAIN | DataFileType.VAL):
            val_size = 250

        corpus_pair = CorpusPair(src_file_path, trg_file_path, type, src_noise, src_tags, size, test_size, val_size)
        pairs.append(corpus_pair)

        if corpus_pair.is_scripture and corpus_pair.is_train and terms_type != DataFileType.NONE:
            src_renderings_path, src_glosses_path = get_terms_paths(corpus_pair.src_file_path)
            trg_renderings_path, trg_glosses_path = get_terms_paths(corpus_pair.trg_file_path)
            if src_renderings_path is not None and trg_renderings_path is not None:
                pairs.append(CorpusPair(src_renderings_path, trg_renderings_path, terms_type))
            if terms_config["include_glosses"]:
                if src_renderings_path is not None and src_glosses_path is not None and corpus_pair.trg_iso == "en":
                    pairs.append(CorpusPair(src_renderings_path, src_glosses_path, terms_type))
                if trg_renderings_path is not None and trg_glosses_path is not None and corpus_pair.src_iso == "en":
                    pairs.append(CorpusPair(trg_glosses_path, trg_renderings_path, terms_type))

    return pairs


def get_parallel_corpus_size(src_file_path: Path, trg_file_path: Path) -> int:
    count = 0
    with open(src_file_path, "r", encoding="utf-8") as src_file, open(trg_file_path, "r", encoding="utf-8") as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if len(src_line) > 0 and len(trg_line) > 0:
                count += 1
    return count


class CorpusPairsConfig(Config):
    def __init__(self, exp_dir: Path, config: dict) -> None:
        config = merge_dict(
            {
                "data": {
                    "terms": {
                        "train": True,
                        "dictionary": False,
                        "categories": "PN",
                        "include_glosses": True,
                    },
                }
            },
            config,
        )

        data_config: dict = config["data"]
        self.corpus_pairs = parse_corpus_pairs(data_config["corpus_pairs"], data_config["terms"])

        if any(p.is_dictionary for p in self.corpus_pairs):
            data_config["source_dictionary"] = str(exp_dir / "dict.src.txt")
            data_config["target_dictionary"] = str(exp_dir / "dict.trg.txt")

        src_isos: Set[str] = set()
        trg_isos: Set[str] = set()
        src_file_paths: Set[Path] = set()
        trg_file_paths: Set[Path] = set()
        src_tags: Set[str] = set()
        for pair in self.corpus_pairs:
            src_isos.add(pair.src_iso)
            trg_isos.add(pair.trg_iso)
            src_file_paths.add(pair.src_file_path)
            trg_file_paths.add(pair.trg_file_path)
            for src_tag in pair.src_tags:
                src_tags.add(src_tag)

        super().__init__(exp_dir, config, src_isos, trg_isos, src_file_paths, trg_file_paths, src_tags)

    def _build_corpora(
        self, src_spp: Optional[sp.SentencePieceProcessor], trg_spp: Optional[sp.SentencePieceProcessor], stats: bool
    ) -> int:
        test_iso_pairs_count = len(set((p.src_iso, p.trg_iso) for p in self.corpus_pairs if p.is_test))
        for old_file_path in self.exp_dir.glob("test.*.txt"):
            old_file_path.unlink()
        with open(self.exp_dir / "train.src.txt", "w", encoding="utf-8", newline="\n") as train_src_file, open(
            self.exp_dir / "train.trg.txt", "w", encoding="utf-8", newline="\n"
        ) as train_trg_file, open(
            self.exp_dir / "val.src.txt", "w", encoding="utf-8", newline="\n"
        ) as val_src_file, open(
            self.exp_dir / "val.trg.txt", "w", encoding="utf-8", newline="\n"
        ) as val_trg_file:
            dict_src_file: Optional[TextIO] = None
            dict_trg_file: Optional[TextIO] = None
            if any(p.is_dictionary for p in self.corpus_pairs):
                dict_src_file = open(self.exp_dir / "dict.src.txt", "w", encoding="utf-8", newline="\n")
                dict_trg_file = open(self.exp_dir / "dict.trg.txt", "w", encoding="utf-8", newline="\n")
            train_count = 0
            for pair in self.corpus_pairs:
                if pair.is_scripture:
                    train_count += self._write_scripture_data_sets(
                        src_spp,
                        trg_spp,
                        train_src_file,
                        train_trg_file,
                        val_src_file,
                        val_trg_file,
                        dict_src_file,
                        dict_trg_file,
                        test_iso_pairs_count,
                        pair,
                    )
                elif pair.is_terms:
                    train_count += self._write_terms_data_sets(
                        src_spp,
                        trg_spp,
                        train_src_file,
                        train_trg_file,
                        val_src_file,
                        val_trg_file,
                        dict_src_file,
                        dict_trg_file,
                        test_iso_pairs_count,
                        pair,
                    )
                else:
                    train_count += self._write_standard_data_sets(
                        src_spp,
                        trg_spp,
                        train_src_file,
                        train_trg_file,
                        val_src_file,
                        val_trg_file,
                        dict_src_file,
                        dict_trg_file,
                        test_iso_pairs_count,
                        pair,
                    )
            return train_count

    def _write_scripture_data_sets(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        train_src_file: TextIO,
        train_trg_file: TextIO,
        val_src_file: TextIO,
        val_trg_file: TextIO,
        dict_src_file: Optional[TextIO],
        dict_trg_file: Optional[TextIO],
        test_iso_pairs_count: int,
        pair: CorpusPair,
    ) -> int:
        corpus = get_scripture_parallel_corpus(pair.src_file_path, pair.trg_file_path)
        return self._write_data_sets(
            src_spp,
            trg_spp,
            train_src_file,
            train_trg_file,
            val_src_file,
            val_trg_file,
            dict_src_file,
            dict_trg_file,
            test_iso_pairs_count,
            pair,
            len(corpus),
            corpus["source"],
            corpus["target"],
        )

    def _write_standard_data_sets(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        train_src_file: TextIO,
        train_trg_file: TextIO,
        val_src_file: TextIO,
        val_trg_file: TextIO,
        dict_src_file: Optional[TextIO],
        dict_trg_file: Optional[TextIO],
        test_iso_pairs_count: int,
        pair: CorpusPair,
    ) -> int:
        corpus_size = get_parallel_corpus_size(pair.src_file_path, pair.trg_file_path)
        with open(pair.src_file_path, "r", encoding="utf-8") as input_src_file, open(
            pair.trg_file_path, "r", encoding="utf-8"
        ) as input_trg_file:
            return self._write_data_sets(
                src_spp,
                trg_spp,
                train_src_file,
                train_trg_file,
                val_src_file,
                val_trg_file,
                dict_src_file,
                dict_trg_file,
                test_iso_pairs_count,
                pair,
                corpus_size,
                input_src_file,
                input_trg_file,
            )

    def _write_terms_data_sets(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        train_src_file: TextIO,
        train_trg_file: TextIO,
        val_src_file: TextIO,
        val_trg_file: TextIO,
        dict_src_file: Optional[TextIO],
        dict_trg_file: Optional[TextIO],
        test_iso_pairs_count: int,
        pair: CorpusPair,
    ) -> int:
        term_cats: Optional[Union[str, List[str]]] = self.data["terms"]["categories"]
        if isinstance(term_cats, str):
            term_cats = [cat.strip() for cat in term_cats.split(",")]
        if term_cats is not None and len(term_cats) == 0:
            return 0

        term_cats_set: Optional[Set[str]] = None if term_cats is None else set(term_cats)
        src_terms: Optional[Dict[str, Term]] = None
        if pair.src_file_path.stem.endswith("-renderings"):
            src_terms = get_terms(pair.src_file_path)
        trg_terms: Optional[Dict[str, Term]] = None
        if pair.trg_file_path.stem.endswith("-renderings"):
            trg_terms = get_terms(pair.trg_file_path)

        corpus: pd.DataFrame
        if src_terms is not None and trg_terms is not None:
            corpus = get_terms_corpus(src_terms, trg_terms, term_cats_set)
        elif src_terms is not None:
            corpus = get_terms_data_frame(src_terms, term_cats_set)
            corpus = corpus.rename(columns={"rendering": "source", "gloss": "target"})
        elif trg_terms is not None:
            corpus = get_terms_data_frame(trg_terms, term_cats_set)
            corpus = corpus.rename(columns={"rendering": "target", "gloss": "source"})
        else:
            return 0

        return self._write_data_sets(
            src_spp,
            trg_spp,
            train_src_file,
            train_trg_file,
            val_src_file,
            val_trg_file,
            dict_src_file,
            dict_trg_file,
            test_iso_pairs_count,
            pair,
            len(corpus),
            corpus["source"],
            corpus["target"],
        )

    def _write_data_sets(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        train_src_file: TextIO,
        train_trg_file: TextIO,
        val_src_file: TextIO,
        val_trg_file: TextIO,
        dict_src_file: Optional[TextIO],
        dict_trg_file: Optional[TextIO],
        test_iso_pairs_count: int,
        pair: CorpusPair,
        corpus_size: int,
        src_corpus: Iterable[str],
        trg_corpus: Iterable[str],
    ) -> int:
        LOGGER.info(f"Writing {pair.src_file_path.stem} -> {pair.trg_file_path.stem}...")
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
        test_prefix = "test" if test_iso_pairs_count == 1 else f"test.{pair.src_iso}.{pair.trg_iso}"
        src_prefix = self._get_src_tags(pair.trg_iso, pair.src_tags)
        mirror_prefix = self._get_src_tags(pair.src_iso, pair.src_tags)

        with open(self.exp_dir / f"{test_prefix}.src.txt", "a", encoding="utf-8", newline="\n") as test_src_file, open(
            self.exp_dir / f"{test_prefix}.trg.detok.txt", "a", encoding="utf-8", newline="\n"
        ) as test_trg_file:
            index = 0
            for src_line, trg_line in zip(src_corpus, trg_corpus):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                if len(src_line) == 0 or len(trg_line) == 0:
                    continue

                src_sentence = src_prefix + src_line
                trg_sentence = trg_line

                mirror_src_sentence = mirror_prefix + trg_line
                mirror_trg_sentence = src_line
                mirror_src_spp = trg_spp
                mirror_trg_spp = src_spp

                if pair.is_test and (test_indices is None or index in test_indices):
                    test_src_file.write(encode_sp(src_spp, src_sentence) + "\n")
                    test_trg_file.write(decode_sp(encode_sp(trg_spp, trg_sentence)) + "\n")
                    test_count += 1
                elif pair.is_val and (val_indices is None or index in val_indices):
                    val_src_file.write(encode_sp(src_spp, src_sentence) + "\n")
                    val_trg_file.write(encode_sp(trg_spp, trg_sentence) + "\n")
                    val_count += 1
                    if self.mirror:
                        val_src_file.write(encode_sp(mirror_src_spp, mirror_src_sentence) + "\n")
                        val_trg_file.write(encode_sp(mirror_trg_spp, mirror_trg_sentence) + "\n")
                        val_count += 1
                elif pair.is_train and (train_indices is None or index in train_indices):
                    noised_src_sentence = self._noise(pair.src_noise, src_sentence)
                    train_count += self._write_train_sentence_pair(
                        train_src_file,
                        train_trg_file,
                        src_spp,
                        trg_spp,
                        noised_src_sentence,
                        trg_sentence,
                        pair.is_dictionary or pair.is_terms,
                    )
                    if self.mirror:
                        train_count += self._write_train_sentence_pair(
                            train_src_file,
                            train_trg_file,
                            mirror_src_spp,
                            mirror_trg_spp,
                            mirror_src_sentence,
                            mirror_trg_sentence,
                            pair.is_dictionary or pair.is_terms,
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
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        src_sentence: str,
        trg_sentence: str,
        is_word: bool,
    ) -> int:
        src_variants = [encode_sp(src_spp, src_sentence, add_dummy_prefix=True)]
        trg_variants = [encode_sp(trg_spp, trg_sentence, add_dummy_prefix=True)]
        if is_word:
            src_variants.append(encode_sp(src_spp, src_sentence, add_dummy_prefix=False))
            trg_variants.append(encode_sp(trg_spp, trg_sentence, add_dummy_prefix=False))
        for src_variant, trg_variant in zip(src_variants, trg_variants):
            src_file.write(src_variant + "\n")
            trg_file.write(trg_variant + "\n")
        return len(src_variants)

    def _get_src_tags(self, trg_iso: str, src_tags: List[str]) -> str:
        tags = ""
        if self.write_trg_tag:
            tags = f"<2{trg_iso}> "
        if len(src_tags) > 0:
            tags = " ".join(map(lambda x: "<2" + str(x) + ">", src_tags)) + " " + tags
        return tags

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
