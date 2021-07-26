import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Type, Union

import yaml

from ..common.canon import ALL_BOOK_IDS, book_id_to_number, is_ot_nt
from ..common.environment import SIL_NLP_ENV
from ..common.flatcat_stemmer import FlatCatStemmer
from ..common.null_stemmer import NullStemmer
from ..common.snowball_stemmer import SnowballStemmer
from ..common.stemmer import Stemmer
from ..common.utils import merge_dict
from ..common.wordnet_stemmer import WordNetStemmer
from .aligner import Aligner
from .clear_aligner import ClearAligner
from .fast_align import FastAlign
from .giza_aligner import HmmGizaAligner, Ibm1GizaAligner, Ibm2GizaAligner, Ibm3GizaAligner, Ibm4GizaAligner
from .machine_aligner import (
    FastAlignMachineAligner,
    HmmMachineAligner,
    Ibm1MachineAligner,
    Ibm2MachineAligner,
    ParatextMachineAligner,
)

ALIGNERS: Dict[str, Tuple[Type[Aligner], str]] = {
    "fast_align": (FastAlignMachineAligner, "FastAlign"),
    "ibm1": (Ibm1MachineAligner, "IBM-1"),
    "ibm2": (Ibm2MachineAligner, "IBM-2"),
    "hmm": (HmmMachineAligner, "HMM"),
    "pt": (ParatextMachineAligner, "PT"),
    "giza_ibm1": (Ibm1GizaAligner, "Giza-IBM-1"),
    "giza_ibm2": (Ibm2GizaAligner, "Giza-IBM-2"),
    "giza_hmm": (HmmGizaAligner, "Giza-HMM"),
    "giza_ibm3": (Ibm3GizaAligner, "Giza-IBM-3"),
    "giza_ibm4": (Ibm4GizaAligner, "Giza-IBM-4"),
    "clear2_fa": (ClearAligner, "Clear-2-FA"),
    "clear2_hmm": (ClearAligner, "Clear-2-HMM"),
    "clear2_ibm1": (ClearAligner, "Clear-2-IBM-1"),
    "clear2_ibm2": (ClearAligner, "Clear-2-IBM-2"),
    "clear2_ibm4": (ClearAligner, "Clear-2-IBM-4"),
    "clear3_fa": (ClearAligner, "Clear-3-FA"),
    "clear3_hmm": (ClearAligner, "Clear-3-HMM"),
    "clab_fast_align": (FastAlign, "clab-FastAlign"),
}


STEMMERS: Dict[str, Type[Stemmer]] = {
    "snowball": SnowballStemmer,
    "wordnet": WordNetStemmer,
    "flatcat": FlatCatStemmer,
    "none": NullStemmer,
}


def get_aligner(id: str, exp_dir: Path) -> Aligner:
    aligner = ALIGNERS.get(id)
    if aligner is None:
        raise RuntimeError("An invalid aligner Id was specified.")
    aligner_cls: Type = aligner[0]
    return aligner_cls(exp_dir / (id + os.path.sep))


def get_aligner_name(id: str) -> str:
    aligner = ALIGNERS.get(id)
    if aligner is None:
        raise RuntimeError("An invalid aligner Id was specified.")
    return aligner[1]


def get_stemmer(stemmer_config: Union[dict, str]) -> Stemmer:
    if isinstance(stemmer_config, str):
        id = stemmer_config
        kwargs = {}
    else:
        id = stemmer_config["name"]
        kwargs = stemmer_config.copy()
        del kwargs["name"]
    stemmer_cls = STEMMERS.get(id)
    if stemmer_cls is None:
        raise RuntimeError("An invalid stemmer Id was specified.")

    return stemmer_cls(**kwargs)


def load_config(exp_dir: Path) -> dict:
    config: dict = {}
    while exp_dir != SIL_NLP_ENV.align_experiments_dir:
        config_path = exp_dir / "config.yml"
        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as file:
                loaded_config = yaml.safe_load(file)
                config = merge_dict(loaded_config, config)
        exp_dir = exp_dir.parent
    return merge_dict(
        {
            "seed": 111,
            "src_stemmer": "none",
            "trg_stemmer": "none",
            "use_src_lemma": False,
            "by_book": False,
            "src_casing": "lower",
            "trg_casing": "lower",
            "src_normalize": True,
            "trg_normalize": True,
        },
        config,
    )


def get_all_book_paths(testament_dir: Path) -> Iterable[Tuple[str, Path]]:
    for book in ALL_BOOK_IDS:
        book_num = book_id_to_number(book)
        if not is_ot_nt(book_num):
            continue
        book_exp_dir = testament_dir / (str(book_num).zfill(3) + "-" + book)
        yield book, book_exp_dir
