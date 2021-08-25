import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from machine.corpora import (
    LOWERCASE,
    NO_OP,
    ParallelTextCorpus,
    ParallelTextSegment,
    TextFileTextCorpus,
    TokenProcessor,
)
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
    ThotSymmetrizedWordAlignmentModel,
    ThotWordAlignmentModel,
    ThotWordAlignmentModelTrainer,
    ThotWordAlignmentModelType,
)
from machine.utils import Phase, PhasedProgressReporter, ProgressStatus
from tqdm import tqdm

from .aligner import Aligner
from .lexicon import Lexicon

LOGGER = logging.getLogger(__name__)

_BATCH_SIZE = 1000


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
        self._train_alignment_model(src_file_path, trg_file_path)

    def align(self, out_file_path: Path, sym_heuristic: str = "grow-diag-final-and") -> None:
        self._align_parallel_corpus(
            self.model_dir / "src_trg_invswm.src", self.model_dir / "src_trg_invswm.trg", out_file_path, sym_heuristic
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
        tokenizer = WhitespaceTokenizer()
        src_corpus = TextFileTextCorpus(tokenizer, src_file_path)
        trg_corpus = TextFileTextCorpus(tokenizer, trg_file_path)
        parallel_corpus = ParallelTextCorpus(src_corpus, trg_corpus)
        preprocessor = NO_OP
        if self.lowercase:
            preprocessor = LOWERCASE

        direct_trainer = ThotWordAlignmentModelTrainer(
            self.model_type, self.model_dir / "src_trg_invswm", preprocessor, preprocessor, parallel_corpus
        )

        inverse_trainer = ThotWordAlignmentModelTrainer(
            self.model_type, self.model_dir / "src_trg_swm", preprocessor, preprocessor, parallel_corpus.invert()
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
        self, src_file_path: Path, trg_file_path: Path, output_file_path: Path, sym_heuristic: str
    ) -> None:
        tokenizer = WhitespaceTokenizer()
        src_corpus = TextFileTextCorpus(tokenizer, src_file_path)
        trg_corpus = TextFileTextCorpus(tokenizer, trg_file_path)
        parallel_corpus = ParallelTextCorpus(src_corpus, trg_corpus)
        LOGGER.info("Loading model")
        model = self._create_symmetrized_model(sym_heuristic)
        preprocessor = NO_OP
        if self.lowercase:
            preprocessor = LOWERCASE

        count = parallel_corpus.get_count()

        LOGGER.info("Aligning corpus")
        with open(
            output_file_path, "w", encoding="utf-8", newline="\n"
        ) as out_file, parallel_corpus.segments as segments, tqdm(
            total=count, bar_format="{l_bar}{bar:40}{r_bar}", leave=False
        ) as pbar:
            for src_segments, trg_segments in _batch(segments, preprocessor):
                for alignment in model.get_best_alignments(src_segments, trg_segments):
                    out_file.write(str(alignment) + "\n")
                pbar.update(len(src_segments))

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
                        trg_word = trg_words[trg_word_index]
                        out_file.write(f"{src_word}\t{trg_word}\t{prob}\n")

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
        elif self.model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            return ThotFastAlignWordAlignmentModel(model_path)
        else:
            raise ValueError("An invalid model type was specified.")


def _batch(
    segments: Iterable[ParallelTextSegment], preprocessor: TokenProcessor
) -> Iterable[Tuple[List[Sequence[str]], List[Sequence[str]]]]:
    src_segments: List[Sequence[str]] = []
    trg_segments: List[Sequence[str]] = []
    for segment in segments:
        src_segments.append(preprocessor.process(segment.source_segment))
        trg_segments.append(preprocessor.process(segment.target_segment))
        if len(src_segments) == _BATCH_SIZE:
            yield src_segments, trg_segments
            src_segments.clear()
            trg_segments.clear()
    if len(src_segments) > 0:
        yield src_segments, trg_segments


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
