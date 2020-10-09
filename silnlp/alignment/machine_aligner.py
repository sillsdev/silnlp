import os
import subprocess
from typing import Optional

from nlp.alignment.aligner import Aligner


def train_alignment_model(
    model_type: str, plugin_file_path: Optional[str], model_dir: str, src_file_path: str, trg_file_path: str
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
    if plugin_file_path is not None:
        args.append("-mp")
        args.append(plugin_file_path)
    subprocess.run(args)


def align_parallel_corpus(
    model_type: str,
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
    if plugin_file_path is not None:
        args.append("-mp")
        args.append(plugin_file_path)
    subprocess.run(args)


class MachineAligner(Aligner):
    def __init__(self, name: str, model_type: str, model_dir: str, plugin_file_path: Optional[str] = None) -> None:
        super().__init__(name, model_dir)
        self.model_type = model_type
        self._plugin_file_path = plugin_file_path

    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        train_alignment_model(self.model_type, self._plugin_file_path, self.model_dir, src_file_path, trg_file_path)
        align_parallel_corpus(
            self.model_type, self._plugin_file_path, self.model_dir, src_file_path, trg_file_path, out_file_path
        )


class Ibm1Aligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("IBM-1", "ibm1", model_dir)


class Ibm2Aligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("IBM-2", "ibm2", model_dir)


class HmmAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("HMM", "hmm", model_dir)


class ParatextAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("PT", "betainv", model_dir, os.getenv("BETA_INV_PLUGIN_PATH"))


class SmtAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("SMT", "smt", model_dir)
