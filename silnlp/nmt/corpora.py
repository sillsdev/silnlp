from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from machine.scripture import get_chapters

from ..common.corpus import get_mt_corpus_path, get_terms_glosses_path, get_terms_list, get_terms_renderings_path
from ..common.environment import SIL_NLP_ENV
from ..common.utils import NoiseMethod, create_noise_methods, is_set
from .augment import AugmentMethod, create_augment_methods


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


BASIC_DATA_PROJECT = "BASIC"


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
        glosses_path = get_terms_glosses_path(list_name, iso=terms_file.iso)
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
