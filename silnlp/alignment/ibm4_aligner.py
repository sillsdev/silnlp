import glob
import os
import platform
import subprocess
from typing import List, Tuple

from nltk.translate import Alignment

from nlp.alignment.aligner import Aligner
from nlp.alignment.fast_align import execute_atools
from nlp.common.corpus import write_corpus


def execute_plain2snt(src_file_path: str, trg_file_path: str, output_dir: str) -> Tuple[str, str]:
    plain2snt_path = os.path.join(os.getenv("MGIZA_PATH", "."), "plain2snt")
    if platform.system() == "Windows":
        plain2snt_path += ".exe"
    if not os.path.isfile(plain2snt_path):
        raise RuntimeError("plain2snt is not installed.")

    src_prefix, _ = os.path.splitext(os.path.basename(src_file_path))
    trg_prefix, _ = os.path.splitext(os.path.basename(trg_file_path))

    src_trg_snt_file_path = os.path.join(output_dir, f"{src_prefix}_{trg_prefix}.snt")
    trg_src_snt_file_path = os.path.join(output_dir, f"{trg_prefix}_{src_prefix}.snt")

    subprocess.run(
        [
            plain2snt_path,
            src_file_path,
            trg_file_path,
            "-vcb1",
            os.path.join(output_dir, f"{src_prefix}.vcb"),
            "-vcb2",
            os.path.join(output_dir, f"{trg_prefix}.vcb"),
            "-snt1",
            src_trg_snt_file_path,
            "-snt2",
            trg_src_snt_file_path,
        ]
    )
    return src_trg_snt_file_path, trg_src_snt_file_path


def execute_mkcls(input_file_path: str, output_dir: str) -> None:
    mkcls_path = os.path.join(os.getenv("MGIZA_PATH", "."), "mkcls")
    if platform.system() == "Windows":
        mkcls_path += ".exe"
    if not os.path.isfile(mkcls_path):
        raise RuntimeError("mkcls is not installed.")

    input_prefix, _ = os.path.splitext(os.path.basename(input_file_path))
    output_file_path = os.path.join(output_dir, f"{input_prefix}.vcb.classes")

    subprocess.run([mkcls_path, "-n10", f"-p{input_file_path}", f"-V{output_file_path}"])


def execute_snt2cooc(snt_file_path: str, output_dir: str) -> None:
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
            os.path.join(output_dir, f"{prefix}.cooc"),
            os.path.join(snt_dir, f"{prefix1}.vcb"),
            os.path.join(snt_dir, f"{prefix2}.vcb"),
            snt_file_path,
        ]
    )


def execute_mgiza(snt_file_path: str, output_path: str) -> None:
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
        ]
    )


def giza_to_pharaoh(model_prefix: str, output_file_path: str) -> None:
    alignments: List[Tuple[int, Alignment]] = []
    for input_file_path in glob.glob(model_prefix + ".A3.final.part*"):
        with open(input_file_path, "r", encoding="utf-8") as in_file:
            line_index = 0
            segment_index = 0
            for line in in_file:
                line = line.strip()
                if line.startswith("#"):
                    start = line.index("(")
                    end = line.index(")")
                    segment_index = int(line[start + 1 : end])
                    line_index = 0
                elif line_index == 2:
                    start = line.find("({")
                    end = line.find("})")
                    src_index = -1
                    pairs: List[Tuple[int, int]] = []
                    while start != -1 and end != -1:
                        if src_index > -1:
                            trg_indices_str = line[start + 3 : end - 1].strip()
                            trg_indices = trg_indices_str.split()
                            for trg_index in trg_indices:
                                pairs.append((src_index, int(trg_index) - 1))
                        start = line.find("({", start + 2)
                        end = line.find("})", end + 2)
                        src_index += 1
                    alignments.append((segment_index, Alignment(pairs)))
                line_index += 1

    write_corpus(output_file_path, map(lambda a: str(a[1]), sorted(alignments, key=lambda a: a[0])))


class Ibm4Aligner(Aligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("ibm4", model_dir)

    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        os.makedirs(self.model_dir, exist_ok=True)

        execute_mkcls(src_file_path, self.model_dir)
        execute_mkcls(trg_file_path, self.model_dir)

        src_trg_snt_file_path, trg_src_snt_file_path = execute_plain2snt(src_file_path, trg_file_path, self.model_dir)

        execute_snt2cooc(src_trg_snt_file_path, self.model_dir)
        execute_snt2cooc(trg_src_snt_file_path, self.model_dir)

        src_trg_prefix, _ = os.path.splitext(src_trg_snt_file_path)
        src_trg_output_prefix = src_trg_prefix + "_invswm"
        execute_mgiza(src_trg_snt_file_path, src_trg_output_prefix)
        src_trg_alignments_file_path = src_trg_output_prefix + ".pharaoh.txt"
        giza_to_pharaoh(src_trg_output_prefix, src_trg_alignments_file_path)

        trg_src_output_prefix = src_trg_prefix + "_swm"
        execute_mgiza(trg_src_snt_file_path, trg_src_output_prefix)
        trg_src_alignments_file_path = trg_src_output_prefix + ".pharaoh.txt"
        giza_to_pharaoh(trg_src_output_prefix, trg_src_alignments_file_path)

        execute_atools(src_trg_alignments_file_path, trg_src_alignments_file_path, out_file_path)
