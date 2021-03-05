import os
import subprocess
from typing import Any, Dict, Optional

from ..common.utils import get_repo_dir
from .aligner import Aligner
from .lexicon import Lexicon


class MachineAligner(Aligner):
    def __init__(
        self,
        id: str,
        model_type: str,
        model_dir: str,
        smt_model_type: Optional[str] = None,
        plugin_file_path: Optional[str] = None,
        has_inverse_model: bool = True,
        params: Dict[str, Any] = {},
    ) -> None:
        super().__init__(id, model_dir)
        self.model_type = model_type
        self.smt_model_type = smt_model_type
        self._plugin_file_path = plugin_file_path
        self._has_inverse_model = has_inverse_model
        self._params = params

    @property
    def has_inverse_model(self) -> bool:
        return self._has_inverse_model

    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        direct_lex_path = os.path.join(self.model_dir, "lexicon.direct.txt")
        if os.path.isfile(direct_lex_path):
            os.remove(direct_lex_path)
        inverse_lex_path = os.path.join(self.model_dir, "lexicon.inverse.txt")
        if os.path.isfile(inverse_lex_path):
            os.remove(inverse_lex_path)
        self._train_alignment_model(src_file_path, trg_file_path)
        self._align_parallel_corpus(src_file_path, trg_file_path, out_file_path)

    def extract_lexicon(self, out_file_path: str) -> None:
        lexicon = self.get_direct_lexicon()
        if self._has_inverse_model:
            inverse_lexicon = self.get_inverse_lexicon()
            print("Symmetrizing lexicons...", end="", flush=True)
            lexicon = Lexicon.symmetrize(lexicon, inverse_lexicon)
            print(" done.")
        lexicon.write(out_file_path)

    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        direct_lex_path = os.path.join(self.model_dir, "lexicon.direct.txt")
        self._extract_lexicon("direct", direct_lex_path)
        return Lexicon.load(direct_lex_path, include_special_tokens)

    def get_inverse_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        if not self._has_inverse_model:
            raise RuntimeError("The aligner does not have an inverse model.")
        inverse_lex_path = os.path.join(self.model_dir, "lexicon.inverse.txt")
        self._extract_lexicon("inverse", inverse_lex_path)
        return Lexicon.load(inverse_lex_path, include_special_tokens)

    def _train_alignment_model(self, src_file_path: str, trg_file_path: str) -> None:
        args = [
            "dotnet",
            "machine",
            "train",
            "alignment-model",
            self.model_dir,
            src_file_path,
            trg_file_path,
            "-mt",
            self.model_type,
            "-l",
        ]
        if self.smt_model_type is not None:
            args.append("-smt")
            args.append(self.smt_model_type)
        if self._plugin_file_path is not None:
            args.append("-mp")
            args.append(self._plugin_file_path)
        if len(self._params) > 0:
            args.append("-tp")
            for key, value in self._params.items():
                args.append(f"{key}={value}")
        subprocess.run(args, cwd=get_repo_dir())

    def _align_parallel_corpus(self, src_file_path: str, trg_file_path: str, output_file_path: str) -> None:
        args = [
            "dotnet",
            "machine",
            "align",
            self.model_dir,
            src_file_path,
            trg_file_path,
            output_file_path,
            "-mt",
            self.model_type,
            "-sh",
            "grow-diag-final-and",
            "-l",
        ]
        if self.smt_model_type is not None:
            args.append("-smt")
            args.append(self.smt_model_type)
        if self._plugin_file_path is not None:
            args.append("-mp")
            args.append(self._plugin_file_path)
        subprocess.run(args, cwd=get_repo_dir())

    def _extract_lexicon(self, direction: str, out_file_path: str, threshold: float = 0.01) -> None:
        args = [
            "dotnet",
            "machine",
            "extract-lexicon",
            self.model_dir,
            out_file_path,
            "-mt",
            self.model_type,
            "-p",
            "-ss",
            "-t",
            str(threshold),
            "-d",
            direction,
        ]
        if self.smt_model_type is not None:
            args.append("-smt")
            args.append(self.smt_model_type)
        if self._plugin_file_path is not None:
            args.append("-mp")
            args.append(self._plugin_file_path)
        subprocess.run(args, cwd=get_repo_dir())


class Ibm1Aligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("ibm1", "ibm1", model_dir)


class Ibm2Aligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("ibm2", "ibm2", model_dir)


class HmmAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("hmm", "hmm", model_dir)


class FastAlign(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("fast_align", "fast_align", model_dir)


class ParatextAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__(
            "pt", "betainv", model_dir, plugin_file_path=os.getenv("BETA_INV_PLUGIN_PATH"), has_inverse_model=False
        )

    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        direct_lex_path = os.path.join(self.model_dir, "lexicon.direct.txt")
        self._extract_lexicon("direct", direct_lex_path, threshold=0)
        return Lexicon.load(direct_lex_path, include_special_tokens)


class SmtAligner(MachineAligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("smt", "smt", model_dir, smt_model_type="hmm")
