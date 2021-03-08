import glob
import os
from os.path import join
import platform
import subprocess
from typing import List, Tuple

from ..common.corpus import load_corpus, write_corpus
from ..common.utils import get_repo_dir
from .aligner import Aligner
from .lexicon import Lexicon


class Ibm4Aligner(Aligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("ibm4", model_dir)

    def train(self, src_file_path: str, trg_file_path: str) -> None:
        os.makedirs(self.model_dir, exist_ok=True)

        self._execute_mkcls(src_file_path)
        self._execute_mkcls(trg_file_path)

        src_trg_snt_file_path, trg_src_snt_file_path = self._execute_plain2snt(src_file_path, trg_file_path)

        self._execute_snt2cooc(src_trg_snt_file_path)
        self._execute_snt2cooc(trg_src_snt_file_path)

        src_trg_prefix, _ = os.path.splitext(src_trg_snt_file_path)
        src_trg_output_prefix = src_trg_prefix + "_invswm"
        self._execute_mgiza(src_trg_snt_file_path, src_trg_output_prefix)
        src_trg_alignments_file_path = src_trg_output_prefix + ".A3.final.all"
        self._merge_alignment_parts(src_trg_output_prefix, src_trg_alignments_file_path)

        trg_src_output_prefix = src_trg_prefix + "_swm"
        self._execute_mgiza(trg_src_snt_file_path, trg_src_output_prefix)
        trg_src_alignments_file_path = trg_src_output_prefix + ".A3.final.all"
        self._merge_alignment_parts(trg_src_output_prefix, trg_src_alignments_file_path)

    def align(self, out_file_path: str, sym_heuristic: str = "grow-diag-final-and") -> None:
        src_trg_alignments_file_path = os.path.join(self.model_dir, "src_trg_invswm.A3.final.all")
        trg_src_alignments_file_path = os.path.join(self.model_dir, "src_trg_swm.A3.final.all")
        self._symmetrize(src_trg_alignments_file_path, trg_src_alignments_file_path, out_file_path, sym_heuristic)

    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        src_vocab = self._load_vocab("src")
        trg_vocab = self._load_vocab("trg")
        return self._load_lexicon(src_vocab, trg_vocab, "invswm", include_special_tokens=include_special_tokens)

    def get_inverse_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        src_vocab = self._load_vocab("src")
        trg_vocab = self._load_vocab("trg")
        return self._load_lexicon(trg_vocab, src_vocab, "swm", include_special_tokens=include_special_tokens)

    def extract_lexicon(self, out_file_path: str) -> None:
        direct_lexicon = self.get_direct_lexicon()
        inverse_lexicon = self.get_inverse_lexicon()
        print("Symmetrizing lexicons...", end="", flush=True)
        lexicon = Lexicon.symmetrize(direct_lexicon, inverse_lexicon)
        print(" done.")
        lexicon.write(out_file_path)

    def _execute_mkcls(self, input_file_path: str) -> None:
        mkcls_path = os.path.join(os.getenv("MGIZA_PATH", "."), "mkcls")
        if platform.system() == "Windows":
            mkcls_path += ".exe"
        if not os.path.isfile(mkcls_path):
            raise RuntimeError("mkcls is not installed.")

        input_prefix, _ = os.path.splitext(os.path.basename(input_file_path))
        output_file_path = os.path.join(self.model_dir, f"{input_prefix}.vcb.classes")

        subprocess.run([mkcls_path, "-n10", f"-p{input_file_path}", f"-V{output_file_path}"])

    def _execute_plain2snt(self, src_file_path: str, trg_file_path: str) -> Tuple[str, str]:
        plain2snt_path = os.path.join(os.getenv("MGIZA_PATH", "."), "plain2snt")
        if platform.system() == "Windows":
            plain2snt_path += ".exe"
        if not os.path.isfile(plain2snt_path):
            raise RuntimeError("plain2snt is not installed.")

        src_prefix, _ = os.path.splitext(os.path.basename(src_file_path))
        trg_prefix, _ = os.path.splitext(os.path.basename(trg_file_path))

        src_trg_snt_file_path = os.path.join(self.model_dir, f"{src_prefix}_{trg_prefix}.snt")
        trg_src_snt_file_path = os.path.join(self.model_dir, f"{trg_prefix}_{src_prefix}.snt")

        subprocess.run(
            [
                plain2snt_path,
                src_file_path,
                trg_file_path,
                "-vcb1",
                os.path.join(self.model_dir, f"{src_prefix}.vcb"),
                "-vcb2",
                os.path.join(self.model_dir, f"{trg_prefix}.vcb"),
                "-snt1",
                src_trg_snt_file_path,
                "-snt2",
                trg_src_snt_file_path,
            ]
        )
        return src_trg_snt_file_path, trg_src_snt_file_path

    def _execute_snt2cooc(self, snt_file_path: str) -> None:
        snt2cooc_path = os.path.join(os.getenv("MGIZA_PATH", "."), "snt2cooc")
        if platform.system() == "Windows":
            snt2cooc_path += ".exe"
        if not os.path.isfile(snt2cooc_path):
            raise RuntimeError("snt2cooc is not installed.")

        snt_dir = os.path.dirname(snt_file_path)
        prefix, _ = os.path.splitext(os.path.basename(snt_file_path))
        prefix1, prefix2 = prefix.split("_")

        subprocess.run(
            [
                snt2cooc_path,
                os.path.join(self.model_dir, f"{prefix}.cooc"),
                os.path.join(snt_dir, f"{prefix1}.vcb"),
                os.path.join(snt_dir, f"{prefix2}.vcb"),
                snt_file_path,
            ]
        )

    def _execute_mgiza(self, snt_file_path: str, output_path: str) -> None:
        mgiza_path = os.path.join(os.getenv("MGIZA_PATH", "."), "mgiza")
        if platform.system() == "Windows":
            mgiza_path += ".exe"
        if not os.path.isfile(mgiza_path):
            raise RuntimeError("mgiza is not installed.")

        snt_dir = os.path.dirname(snt_file_path)
        prefix, _ = os.path.splitext(os.path.basename(snt_file_path))
        prefix1, prefix2 = prefix.split("_")

        subprocess.run(
            [
                mgiza_path,
                "-C",
                snt_file_path,
                "-CoocurrenceFile",
                os.path.join(snt_dir, f"{prefix}.cooc"),
                "-S",
                os.path.join(snt_dir, f"{prefix1}.vcb"),
                "-T",
                os.path.join(snt_dir, f"{prefix2}.vcb"),
                "-o",
                output_path,
            ],
            stderr=subprocess.DEVNULL,
        )

    def _merge_alignment_parts(self, model_prefix: str, output_file_path: str) -> None:
        alignments: List[Tuple[int, str]] = []
        for input_file_path in glob.glob(model_prefix + ".A3.final.part*"):
            with open(input_file_path, "r", encoding="utf-8") as in_file:
                line_index = 0
                segment_index = 0
                cur_alignment: str = ""
                for line in in_file:
                    cur_alignment += line
                    alignment_line_index = line_index % 3
                    if alignment_line_index == 0:
                        start = line.index("(")
                        end = line.index(")")
                        segment_index = int(line[start + 1 : end])
                    elif alignment_line_index == 2:
                        alignments.append((segment_index, cur_alignment.strip()))
                        cur_alignment = ""
                    line_index += 1

        write_corpus(output_file_path, map(lambda a: str(a[1]), sorted(alignments, key=lambda a: a[0])))

    def _symmetrize(
        self, direct_align_path: str, inverse_align_path: str, output_path: str, sym_heuristic: str
    ) -> None:
        args = [
            "dotnet",
            "machine",
            "symmetrize",
            direct_align_path,
            inverse_align_path,
            output_path,
            "-sh",
            sym_heuristic,
        ]
        subprocess.run(args, cwd=get_repo_dir())

    def _load_vocab(self, side: str) -> List[str]:
        vocab_path = os.path.join(self.model_dir, f"{side}.vcb")
        vocab: List[str] = ["NULL", "UNK"]
        for line in load_corpus(vocab_path):
            index_str, word, _ = line.split()
            assert int(index_str) == len(vocab)
            vocab.append(word)
        return vocab

    def _load_lexicon(
        self, src_vocab: List[str], trg_vocab: List[str], align_model: str, include_special_tokens: bool
    ) -> Lexicon:
        lexicon = Lexicon()
        model_path = os.path.join(self.model_dir, f"src_trg_{align_model}.t3.final")
        for line in load_corpus(model_path):
            src_index_str, trg_index_str, prob_str = line.split()
            src_index = int(src_index_str)
            trg_index = int(trg_index_str)
            if include_special_tokens or (src_index > 1 and trg_index > 1):
                src_word = src_vocab[src_index]
                trg_word = trg_vocab[trg_index]
                lexicon[src_word, trg_word] = float(prob_str)
        return lexicon

