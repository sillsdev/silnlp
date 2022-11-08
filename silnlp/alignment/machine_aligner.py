import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, List

from machine.corpora import TextFileTextCorpus
from machine.tokenization import WhitespaceTokenizer
from machine.translation import (
    SymmetrizationHeuristic,
    SymmetrizedWordAlignmentModel,
    SymmetrizedWordAlignmentModelTrainer,
)
from machine.translation.thot import (
    ThotFastAlignWordAlignmentModel,
    ThotHmmWordAlignmentModel,
    ThotIbm1WordAlignmentModel,
    ThotIbm2WordAlignmentModel,
    ThotIbm3WordAlignmentModel,
    ThotIbm4WordAlignmentModel,
    ThotSymmetrizedWordAlignmentModel,
    ThotWordAlignmentModel,
    ThotWordAlignmentModelTrainer,
    ThotWordAlignmentModelType,
    ThotWordAlignmentParameters,
)
from machine.utils import Phase, PhasedProgressReporter, ProgressStatus
from tqdm import tqdm

from ..common.environment import get_env_path
from .aligner import Aligner
from .lexicon import Lexicon

LOGGER = logging.getLogger(__name__)

_BATCH_SIZE = 1024


class MachineAligner(Aligner):
    def __init__(
        self,
        id: str,
        model_type: ThotWordAlignmentModelType,
        model_dir: Path,
        threshold: float = 0.01,
    ) -> None:
        super().__init__(id, model_dir)
        self.model_type = model_type
        self._threshold = threshold
        self.lowercase = False

    def train(self, src_file_path: Path, trg_file_path: Path) -> None:
        direct_lex_path = self.model_dir / "lexicon.direct.txt"
        if direct_lex_path.is_file():
            direct_lex_path.unlink()
        inverse_lex_path = self.model_dir / "lexicon.inverse.txt"
        if inverse_lex_path.is_file():
            inverse_lex_path.unlink()
        self.model_dir.mkdir(exist_ok=True)
        if self.model_type is ThotWordAlignmentModelType.IBM4:
            self._execute_mkcls(src_file_path, "src")
            self._execute_mkcls(trg_file_path, "trg")
        self._train_alignment_model(src_file_path, trg_file_path)

    def align(
        self, out_file_path: Path, sym_heuristic: str = "grow-diag-final-and", export_probabilities: bool = False
    ) -> None:
        self._align_parallel_corpus(
            self.model_dir / "src_trg_invswm.src",
            self.model_dir / "src_trg_invswm.trg",
            out_file_path,
            sym_heuristic,
            export_probabilities=export_probabilities,
        )

    def force_align(
        self, src_file_path: Path, trg_file_path: Path, out_file_path: Path, sym_heuristic: str = "grow-diag-final-and"
    ) -> None:
        self._align_parallel_corpus(src_file_path, trg_file_path, out_file_path, sym_heuristic)

    def extract_lexicon(self, out_file_path: Path) -> None:
        lexicon = self.get_direct_lexicon()
        inverse_lexicon = self.get_inverse_lexicon()
        lexicon = Lexicon.symmetrize(lexicon, inverse_lexicon)
        lexicon.write(out_file_path)

    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        direct_lex_path = self.model_dir / "lexicon.direct.txt"
        self._extract_lexicon(direct_lex_path, direct=True)
        return Lexicon.load(direct_lex_path, include_special_tokens)

    def get_inverse_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        inverse_lex_path = self.model_dir / "lexicon.inverse.txt"
        self._extract_lexicon(inverse_lex_path, direct=False)
        return Lexicon.load(inverse_lex_path, include_special_tokens)

    def _train_alignment_model(self, src_file_path: Path, trg_file_path: Path) -> None:
        src_corpus = TextFileTextCorpus(src_file_path)
        trg_corpus = TextFileTextCorpus(trg_file_path)
        parallel_corpus = src_corpus.align_rows(trg_corpus).tokenize(WhitespaceTokenizer())
        if self.lowercase:
            parallel_corpus = parallel_corpus.lowercase()

        direct_params = ThotWordAlignmentParameters()
        direct_params.ibm1_iteration_count = 5
        direct_params.ibm2_iteration_count = 5 if self.model_type is ThotWordAlignmentModelType.IBM2 else 0
        direct_params.hmm_iteration_count = 5
        direct_params.ibm3_iteration_count = 5
        direct_params.ibm4_iteration_count = 5

        inverse_params = ThotWordAlignmentParameters()
        inverse_params.ibm1_iteration_count = 5
        inverse_params.ibm2_iteration_count = 5 if self.model_type is ThotWordAlignmentModelType.IBM2 else 0
        inverse_params.hmm_iteration_count = 5
        inverse_params.ibm3_iteration_count = 5
        inverse_params.ibm4_iteration_count = 5
        if self.model_type is ThotWordAlignmentModelType.IBM4:
            direct_params.source_word_classes = self._load_word_classes("src")
            direct_params.target_word_classes = self._load_word_classes("trg")
            inverse_params.source_word_classes = direct_params.target_word_classes
            inverse_params.target_word_classes = direct_params.source_word_classes

        direct_trainer = ThotWordAlignmentModelTrainer(
            self.model_type, parallel_corpus, self.model_dir / "src_trg_invswm", parameters=direct_params
        )

        inverse_trainer = ThotWordAlignmentModelTrainer(
            self.model_type, parallel_corpus.invert(), self.model_dir / "src_trg_swm", parameters=inverse_params
        )

        trainer = SymmetrizedWordAlignmentModelTrainer(direct_trainer, inverse_trainer)

        LOGGER.info("Training model")
        with tqdm(total=1.0, bar_format="{percentage:3.0f}%|{bar:40}|{desc}", leave=False) as pbar:

            def progress(status: ProgressStatus) -> None:
                pbar.update(status.percent_completed - pbar.n)
                pbar.set_description_str(status.message)

            reporter = PhasedProgressReporter(progress, [Phase("Training model", 0.96), Phase("Saving model", 0.04)])
            with reporter.start_next_phase() as phase_progress:
                trainer.train(phase_progress)
            with reporter.start_next_phase():
                trainer.save()

    def _align_parallel_corpus(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        output_file_path: Path,
        sym_heuristic: str,
        export_probabilities: bool = False,
    ) -> None:
        src_corpus = TextFileTextCorpus(src_file_path)
        trg_corpus = TextFileTextCorpus(trg_file_path)
        parallel_corpus = src_corpus.align_rows(trg_corpus).tokenize(WhitespaceTokenizer())
        LOGGER.info("Loading model")
        model = self._create_symmetrized_model(sym_heuristic)
        if self.lowercase:
            parallel_corpus = parallel_corpus.lowercase()

        count = parallel_corpus.count()

        LOGGER.info("Aligning corpus")
        with open(output_file_path, "w", encoding="utf-8", newline="\n") as out_file, parallel_corpus.batch(
            _BATCH_SIZE
        ) as batches, tqdm(total=count, bar_format="{l_bar}{bar:40}{r_bar}", leave=False) as pbar:
            for row_batch in batches:
                for (source_segment, target_segment), alignment in zip(row_batch, model.align_batch(row_batch)):
                    if export_probabilities:
                        word_pairs = alignment.to_aligned_word_pairs()
                        alignened_word_pairs = model.compute_aligned_word_pair_scores(
                            source_segment, target_segment, word_pairs
                        )
                        out_file.write(" ".join(str(wp) for wp in alignened_word_pairs) + "\n")
                    else:
                        out_file.write(str(alignment) + "\n")
                pbar.update(len(row_batch))

    def _extract_lexicon(self, out_file_path: Path, direct: bool) -> None:
        model = self._create_model(direct)
        with open(out_file_path, "w", encoding="utf-8", newline="\n") as out_file:
            src_words = list(model.source_words)
            trg_words = list(model.target_words)
            for src_word_index in range(len(src_words)):
                src_word = src_words[src_word_index]
                trg_word_probs = model.get_translations(src_word_index, self._threshold)
                for trg_word_index, prob in sorted(trg_word_probs, key=lambda wp: wp[1], reverse=True):
                    prob = round(prob, 8)
                    if prob > 0:
                        try:
                            trg_word = trg_words[trg_word_index]
                            out_file.write(f"{src_word}\t{trg_word}\t{prob}\n")
                        except IndexError:
                            print(
                                f"Index error! Source Word/Index: {src_word}/{src_word_index},"
                                f"Target Index: {trg_word_index}, Probability: {prob}"
                            )

    def _create_symmetrized_model(self, sym_heuristic: str) -> SymmetrizedWordAlignmentModel:
        model = ThotSymmetrizedWordAlignmentModel(self._create_model(direct=True), self._create_model(direct=False))
        model.heuristic = SymmetrizationHeuristic[sym_heuristic.upper().replace("-", "_")]
        return model

    def _create_model(self, direct: bool) -> ThotWordAlignmentModel:
        model_path = self.model_dir / ("src_trg_invswm" if direct else "src_trg_swm")
        if self.model_type is ThotWordAlignmentModelType.IBM1:
            return ThotIbm1WordAlignmentModel(model_path)
        elif self.model_type is ThotWordAlignmentModelType.IBM2:
            return ThotIbm2WordAlignmentModel(model_path)
        elif self.model_type is ThotWordAlignmentModelType.HMM:
            return ThotHmmWordAlignmentModel(model_path)
        elif self.model_type is ThotWordAlignmentModelType.IBM3:
            return ThotIbm3WordAlignmentModel(model_path)
        elif self.model_type is ThotWordAlignmentModelType.IBM4:
            return ThotIbm4WordAlignmentModel(model_path)
        elif self.model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            return ThotFastAlignWordAlignmentModel(model_path)
        else:
            raise ValueError("An invalid model type was specified.")

    def _execute_mkcls(self, input_file_path: Path, side: str) -> None:
        mkcls_path = Path(get_env_path("MGIZA_PATH"), "mkcls")
        if platform.system() == "Windows":
            mkcls_path = mkcls_path.with_suffix(".exe")
        if not mkcls_path.is_file():
            raise RuntimeError("mkcls is not installed.")

        output_file_path = self.model_dir / f"src_trg.{side}.classes"

        args: List[str] = [str(mkcls_path), "-n10", f"-p{input_file_path}", f"-V{output_file_path}"]
        subprocess.run(args)

    def _load_word_classes(self, side: str) -> Dict[str, str]:
        word_classes: Dict[str, str] = {}
        with open(self.model_dir / f"src_trg.{side}.classes", "r", encoding="utf-8-sig") as file:
            for line in file:
                line = line.strip()
                word, word_class = line.split("\t", maxsplit=2)
                word_classes[word] = word_class
        return word_classes


class Ibm1MachineAligner(MachineAligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("ibm1", ThotWordAlignmentModelType.IBM1, model_dir)


class Ibm2MachineAligner(MachineAligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("ibm2", ThotWordAlignmentModelType.IBM2, model_dir)


class HmmMachineAligner(MachineAligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("hmm", ThotWordAlignmentModelType.HMM, model_dir)


class FastAlignMachineAligner(MachineAligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("fast_align", ThotWordAlignmentModelType.FAST_ALIGN, model_dir)


class Ibm3MachineAligner(MachineAligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("ibm3", ThotWordAlignmentModelType.IBM3, model_dir)


class Ibm4MachineAligner(MachineAligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("ibm4", ThotWordAlignmentModelType.IBM4, model_dir)
