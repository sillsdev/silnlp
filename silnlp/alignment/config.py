import os
from typing import Dict, Type
import yaml

from nlp.alignment.aligner import Aligner
from nlp.alignment.clear_aligner import ClearAligner
from nlp.alignment.fast_align import FastAlign
from nlp.alignment.ibm4_aligner import Ibm4Aligner
from nlp.alignment.machine_aligner import HmmAligner, Ibm1Aligner, Ibm2Aligner, ParatextAligner, SmtAligner
from nlp.common.utils import get_align_root_dir, merge_dict


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


def get_aligner(id: str, root_dir: str) -> Aligner:
    aligner_cls = ALIGNERS.get(id)
    if aligner_cls is None:
        raise RuntimeError("An invalid method was specified.")
    return aligner_cls(os.path.join(root_dir, id + os.path.sep))


def load_config(exp_name: str) -> dict:
    root_dir = get_align_root_dir(exp_name)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {}

    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)
