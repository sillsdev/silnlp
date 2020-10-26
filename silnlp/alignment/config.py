import os
from typing import Dict, Type, Union
import yaml

from nlp.alignment.aligner import Aligner
from nlp.alignment.clear_aligner import ClearAligner
from nlp.alignment.fast_align import FastAlign
from nlp.alignment.ibm4_aligner import Ibm4Aligner
from nlp.alignment.machine_aligner import HmmAligner, Ibm1Aligner, Ibm2Aligner, ParatextAligner, SmtAligner
from nlp.common.flatcat_stemmer import FlatCatStemmer
from nlp.common.null_stemmer import NullStemmer
from nlp.common.snowball_stemmer import SnowballStemmer
from nlp.common.stemmer import Stemmer
from nlp.common.utils import get_align_root_dir, merge_dict
from nlp.common.wordnet_stemmer import WordNetStemmer

ALIGNERS: Dict[str, Type[Aligner]] = {
    "fast_align": FastAlign,
    "ibm1": Ibm1Aligner,
    "ibm2": Ibm2Aligner,
    "ibm4": Ibm4Aligner,
    "hmm": HmmAligner,
    "smt": SmtAligner,
    "pt": ParatextAligner,
    "clear": ClearAligner,
}


STEMMERS: Dict[str, Type[Stemmer]] = {
    "snowball": SnowballStemmer,
    "wordnet": WordNetStemmer,
    "flatcat": FlatCatStemmer,
    "none": NullStemmer,
}


def get_aligner(id: str, root_dir: str) -> Aligner:
    aligner_cls = ALIGNERS.get(id)
    if aligner_cls is None:
        raise RuntimeError("An invalid aligner Id was specified.")
    return aligner_cls(os.path.join(root_dir, id + os.path.sep))


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


def load_config(exp_name: str) -> dict:
    root_dir = get_align_root_dir(exp_name)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {"seed": 111, "src_stemmer": "none", "trg_stemmer": "none", "use_src_lemma": False}

    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)
