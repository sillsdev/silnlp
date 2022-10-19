import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from machine.scripture import book_number_to_id, get_books

from ..common.environment import SIL_NLP_ENV
from ..common.paratext import book_file_name_digits
from ..common.tf_utils import enable_eager_execution, enable_memory_growth
from ..common.translator import Translator
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import CheckpointType, Config, NMTModel
from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__name__)


class NMTTranslator(Translator):
    def __init__(self, tokenizer: Tokenizer, model: NMTModel, checkpoint: Union[CheckpointType, str, int]):
        self._tokenizer = tokenizer
        self._model = model
        self._checkpoint = checkpoint

    def translate(
        self,
        sentences: Iterable[Union[str, List[str]]],
        src_iso: Optional[str] = None,
        trg_iso: Optional[str] = None,
    ) -> Iterable[str]:
        return self._tokenizer.detokenize_all(self._model.translate(sentences, src_iso, trg_iso, self._checkpoint))


@dataclass
class TranslationTask:
    name: str
    checkpoint: str = "last"
    clearml_queue: Optional[str] = None

    def __post_init__(self):
        if self.checkpoint is None:
            self.checkpoint = "last"

    def init_translation_task(self, experiment_suffix: str) -> Tuple[Translator, Config, str]:
        clearml = SILClearML(
            self.name,
            self.clearml_queue,
            project_suffix="_infer",
            experiment_suffix=experiment_suffix,
        )
        self.name = clearml.name

        SIL_NLP_ENV.copy_experiment_from_bucket(
            self.name, extensions=(".vocab", ".model", ".yml", "dict.src.txt", "dict.trg.txt", "dict.vref.txt")
        )

        clearml.config.set_seed()

        tokenizer = clearml.config.create_tokenizer()
        model = clearml.config.create_model()
        translator = NMTTranslator(tokenizer, model, self.checkpoint)
        step = model.get_checkpoint_step(self.checkpoint)
        step_str = "avg" if step == -1 else str(step)
        return translator, clearml.config, step_str

    def translate_books(
        self,
        books: str,
        src_project: Optional[str] = None,
        trg_iso: Optional[str] = None,
    ):
        translator, config, step_str = self.init_translation_task(experiment_suffix=f"_{self.checkpoint}_{books}")
        book_nums = get_books(books)

        if src_project is None:
            if len(config.src_projects) != 1:
                raise RuntimeError("A source project must be specified.")
            src_project = next(iter(config.src_projects))

        output_dir = config.exp_dir / "infer" / step_str
        output_dir.mkdir(exist_ok=True, parents=True)

        displayed_error_already = False
        for book_num in book_nums:
            book = book_number_to_id(book_num)
            output_path = output_dir / f"{book_file_name_digits(book_num)}{book}.SFM"
            try:
                LOGGER.info(f"Translating {book} ...")
                translator.translate_book(src_project, book, output_path, trg_iso=trg_iso)
            except Exception as e:
                if not displayed_error_already:
                    LOGGER.error(f"Was not able to translate {book}.  Error: {e.args[0]}")
                    displayed_error_already = True
                else:
                    LOGGER.error(f"Was not able to translate {book}.")
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def translate_text_files(
        self, src_prefix: str, trg_prefix: str, start_seq: int, end_seq: int, trg_iso: Optional[str] = None
    ):
        translator, _, _ = self.init_translation_task(experiment_suffix=f"_{self.checkpoint}_{src_prefix}")
        if trg_prefix is None:
            raise RuntimeError("A target file prefix must be specified.")
        if start_seq is None or end_seq is None:
            raise RuntimeError("Start and end sequence numbers must be specified.")
        cwd = Path.cwd()
        for i in range(start_seq, end_seq + 1):
            file_num = f"{i:04d}"
            src_file_path = cwd / f"{src_prefix}{file_num}.txt"
            trg_file_path = cwd / f"{trg_prefix}{file_num}.txt"
            if src_file_path.is_file() and not trg_file_path.is_file():
                start = time.time()
                translator.translate_text_file(src_file_path, trg_file_path, trg_iso=trg_iso)
                end = time.time()
                print(f"Translated {src_file_path.name} to {trg_file_path.name} in {((end-start)/60):.2f} minutes")
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def translate_text_file(self, src_filename: str, trg_iso: Optional[str] = None, trg_filename: Optional[str] = None):
        translator, config, step_str = self.init_translation_task(
            experiment_suffix=f"_{self.checkpoint}_{os.path.basename(src_filename)}"
        )

        if Path(src_filename).exists():
            src_file_path = Path(src_filename)
        elif (SIL_NLP_ENV.data_dir / src_filename).exists():
            src_file_path = SIL_NLP_ENV.data_dir / src_filename
        else:
            raise FileNotFoundError("Cannot find: " + src_filename)

        if trg_filename is not None:
            if Path(trg_filename).parent.exists():
                trg_file_path = Path(trg_filename)
            elif (SIL_NLP_ENV.data_dir / trg_filename).parent.exists():
                trg_file_path = SIL_NLP_ENV.data_dir / trg_filename
            else:
                raise FileNotFoundError("Cannot find parent folder of: " + trg_filename)
        else:
            default_output_dir = config.exp_dir / "infer" / step_str
            trg_file_path = default_output_dir / src_file_path.name
            trg_file_path.parent.mkdir(exist_ok=True, parents=True)
        translator.translate_text_file(src_file_path, trg_file_path, trg_iso=trg_iso)
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use (last, best, avg, or checkpoint #)")
    parser.add_argument("--src", default=None, type=str, help="Source file")
    parser.add_argument("--trg", default=None, type=str, help="Target file")
    parser.add_argument("--src-prefix", default=None, type=str, help="Source file prefix (e.g., de-news2019-)")
    parser.add_argument("--trg-prefix", default=None, type=str, help="Target file prefix (e.g., en-news2019-)")
    parser.add_argument("--start-seq", default=None, type=int, help="Starting file sequence #")
    parser.add_argument("--end-seq", default=None, type=int, help="Ending file sequence #")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to translate; e.g., 'NT', 'OT', 'GEN,EXO'"
    )
    parser.add_argument("--trg-iso", default=None, type=str, help="The target language (iso code) to translate into")
    parser.add_argument("--output-usfm", default=None, type=str, help="The output USFM file path")
    parser.add_argument(
        "--eager-execution",
        default=False,
        action="store_true",
        help="Enable TensorFlow eager execution.",
    )
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run it locally and register it with ClearML.",
    )
    args = parser.parse_args()

    get_git_revision_hash()

    if args.eager_execution:
        enable_eager_execution()

    if args.memory_growth:
        enable_memory_growth()

    translator = TranslationTask(
        name=args.experiment,
        checkpoint=args.checkpoint,
        clearml_queue=args.clearml_queue,
    )

    if len(args.books) > 0:
        translator.translate_books(args.books, src_project=args.src_project, trg_iso=args.trg_iso)
    elif args.src_prefix is not None:
        translator.translate_text_files(args.src_prefix, args.trg_prefix, args.start_seq, args.end_seq, args.trg_iso)
    elif args.src is not None:
        translator.translate_text_file(args.src, args.trg_iso, args.trg)
    else:
        raise RuntimeError("A Scripture book, file, or file prefix must be specified.")


if __name__ == "__main__":
    main()
