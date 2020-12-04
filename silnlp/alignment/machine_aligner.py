import os
import subprocess
from typing import Optional

from ..common.utils import get_repo_dir
from .aligner import Aligner


def train_alignment_model(
    model_type: str,
    smt_model_type: Optional[str],
    plugin_file_path: Optional[str],
    model_dir: str,
    src_file_path: str,
    trg_file_path: str,
) -> None:
    args = [
        "dotnet",
        "machine",
        "train",
        "alignment-model",
        model_dir,
        src_file_path,
        trg_file_path,
        "-mt",
        model_type,
    ]
    if smt_model_type is not None:
        args.append("-smt")
        args.append(smt_model_type)
    if plugin_file_path is not None:
        args.append("-mp")
        args.append(plugin_file_path)
    subprocess.run(args, cwd=get_repo_dir())


def align_parallel_corpus(
    model_type: str,
    smt_model_type: Optional[str],
    plugin_file_path: Optional[str],
    model_dir: str,
    src_file_path: str,
    trg_file_path: str,
    output_file_path: str,
) -> None:
    args = [
        "dotnet",
        "machine",
        "align",
        model_dir,
        src_file_path,
        trg_file_path,
        output_file_path,
        "-mt",
        model_type,
    ]
    if smt_model_type is not None:
        args.append("-smt")
        args.append(smt_model_type)
    if plugin_file_path is not None:
        args.append("-mp")
        args.append(plugin_file_path)
    subprocess.run(args, cwd=get_repo_dir())


class MachineAligner(Aligner):
    def __init__(
        self,
        id: str,
        model_type: str,
        model_dir: str,
        smt_model_type: Optional[str] = None,
        plugin_file_path: Optional[str] = None,
    ) -> None:
        super().__init__(id, model_dir)
        self.model_type = model_type
        self.smt_model_type = smt_model_type
        self._plugin_file_path = plugin_file_path

    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        train_alignment_model(
            self.model_type, self.smt_model_type, self._plugin_file_path, self.model_dir, src_file_path, trg_file_path
        )
        align_parallel_corpus(
            self.model_type,
            self.smt_model_type,
            self._plugin_file_path,
            self.model_dir,
            src_file_path,
            trg_file_path,
            out_file_path,
        )


class Ibm1Aligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("ibm1", "ibm1", model_dir)


class Ibm2Aligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("ibm2", "ibm2", model_dir)


class HmmAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("hmm", "hmm", model_dir)


class ParatextAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("pt", "betainv", model_dir, plugin_file_path=os.getenv("BETA_INV_PLUGIN_PATH"))


class SmtAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("smt", "smt", model_dir, smt_model_type="hmm")
