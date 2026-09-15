from pathlib import Path
from typing import Iterable, List, Optional, Union

import pytest

from silnlp.common.environment import SilNlpEnv
from silnlp.common.utils import Side
from silnlp.nmt.corpora import CorpusPair, DataFile, DataFileMapping, DataFileType
from silnlp.nmt.tokenizer import Tokenizer


@pytest.fixture
def environment(tmp_path: Path) -> SilNlpEnv:
    silnlp_env = SilNlpEnv(
        data_dir=tmp_path,
        mt_dir=tmp_path / "MT",
        mt_experiments_dir=tmp_path / "MT" / "experiments",
        mt_terms_dir=tmp_path / "MT" / "terms",
        mt_scripture_dir=tmp_path / "MT" / "scripture",
    )
    for directory in (silnlp_env.mt_scripture_dir, silnlp_env.mt_corpora_dir, silnlp_env.mt_terms_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return silnlp_env


@pytest.fixture
def corpora(environment: SilNlpEnv) -> "CorpusBuilder":
    return CorpusBuilder(environment)


class CorpusBuilder:
    """Builds the corpus pairs of an experiment, with files that exist on disk."""

    def __init__(self, environment: SilNlpEnv) -> None:
        self._environment = environment

    def scripture_file(self, iso: str, project: str) -> DataFile:
        return self._data_file(self._environment.mt_scripture_dir / f"{iso}-{project}.txt")

    def basic_file(self, iso: str, name: str, lines: Iterable[str] = ()) -> DataFile:
        return self._data_file(self._environment.mt_corpora_dir / f"{iso}-{name}.txt", lines)

    def terms_file(self, iso: str, project: str, list_type: str = "Custom") -> DataFile:
        return self._data_file(self._environment.mt_terms_dir / f"{iso}-{project}-{list_type}-renderings.txt")

    def glosses_file(self, iso: str, list_type: str = "Custom") -> Path:
        # A list type the repository does not ship glosses for, so the resolver uses this directory.
        path = self._environment.mt_terms_dir / f"{iso.lower()}-{list_type}-glosses.txt"
        path.touch()
        return path

    def pair(
        self,
        src_files: Iterable[DataFile],
        trg_files: Iterable[DataFile],
        type: DataFileType = DataFileType.TRAIN | DataFileType.TEST | DataFileType.VAL,
        tags: Iterable[str] = (),
        size: Union[float, int] = 1.0,
        test_size: Optional[Union[float, int]] = None,
        val_size: Optional[Union[float, int]] = None,
        src_terms_files: Iterable[DataFile] = (),
        trg_terms_files: Iterable[DataFile] = (),
        mapping: DataFileMapping = DataFileMapping.ONE_TO_ONE,
        is_lexical_data: bool = False,
    ) -> CorpusPair:
        return CorpusPair(
            src_files=list(src_files),
            trg_files=list(trg_files),
            type=type,
            src_noise=[],
            tags=list(tags),
            size=size,
            test_size=test_size,
            val_size=val_size,
            disjoint_test=True,
            disjoint_val=True,
            score_threshold=0.0,
            corpus_books={},
            test_books={},
            use_test_set_from="",
            src_terms_files=list(src_terms_files),
            trg_terms_files=list(trg_terms_files),
            is_lexical_data=is_lexical_data,
            mapping=mapping,
        )

    def _data_file(self, path: Path, lines: Iterable[str] = ()) -> DataFile:
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        return DataFile(path, environment=self._environment)


def isos(data_files: Iterable[DataFile]) -> List[str]:
    return [data_file.iso for data_file in data_files]


class MarkingTokenizer(Tokenizer):
    """Marks what it was asked to do, so the writers' calls are visible in the written files."""

    def __init__(self) -> None:
        self.src_lang = ""
        self.trg_lang = ""

    def set_src_lang(self, src_lang: str) -> None:
        self.src_lang = src_lang

    def set_trg_lang(self, trg_lang: str) -> None:
        self.trg_lang = trg_lang

    def tokenize(self, side, line, add_dummy_prefix=True, sample_subwords=False, add_special_tokens=True) -> str:
        lang = self.src_lang if side is Side.SOURCE else self.trg_lang
        return f"{'_' if add_dummy_prefix else ''}{lang}|{line}"

    def normalize(self, side: Side, line: str) -> str:
        lang = self.src_lang if side is Side.SOURCE else self.trg_lang
        return f"norm({lang}|{line})"

    def detokenize(self, line: str) -> str:
        return line
