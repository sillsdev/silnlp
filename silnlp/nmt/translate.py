import argparse
import logging
import os
import time
from dataclasses import dataclass
from inspect import getmembers
from pathlib import Path
from pprint import pprint
from types import FunctionType
from typing import Iterable, Optional, Tuple, Union

from machine.scripture import VerseRef, book_number_to_id, get_books

from ..common.environment import SIL_NLP_ENV
from ..common.paratext import book_file_name_digits, get_project_dir
from ..common.tf_utils import enable_eager_execution, enable_memory_growth
from ..common.translator import Translator
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import CheckpointType, Config, NMTModel

LOGGER = logging.getLogger(__package__ + ".translate")


class NMTTranslator(Translator):
    def __init__(self, model: NMTModel, checkpoint: Union[CheckpointType, str, int]) -> None:
        self._model = model
        self._checkpoint = checkpoint

    def translate(
        self, sentences: Iterable[str], src_iso: str, trg_iso: str, vrefs: Optional[Iterable[VerseRef]] = None
    ) -> Iterable[str]:
        return self._model.translate(sentences, src_iso, trg_iso, vrefs, self._checkpoint)


@dataclass
class TranslationTask:
    name: str
    checkpoint: str = "last"
    clearml_queue: Optional[str] = None

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            self.checkpoint = "last"

    def translate_books(
        self,
        books: str,
        src_project: Optional[str],
        trg_iso: Optional[str],
    ):
        translator, config, step_str = self._init_translation_task(experiment_suffix=f"_{self.checkpoint}_{books}")
        book_nums = get_books(books)

        if src_project is None:
            if len(config.src_projects) != 1:
                raise RuntimeError("A source project must be specified.")
            src_project = next(iter(config.src_projects))

        src_project_dir = get_project_dir(src_project)
        if not src_project_dir.is_dir():
            LOGGER.error(f"Source project {src_project} not found in projects folder {src_project_dir}")
            return

        if trg_iso is None:
            trg_iso = config.default_trg_iso

        output_dir = config.exp_dir / "infer" / step_str
        output_dir.mkdir(exist_ok=True, parents=True)

        displayed_error_already = False
        for book_num in book_nums:
            book = book_number_to_id(book_num)
            output_path = output_dir / f"{book_file_name_digits(book_num)}{book}.SFM"
            try:
                LOGGER.info(f"Translating {book} ...")
                translator.translate_book(src_project, book, output_path, trg_iso)
            except Exception as e:
                if not displayed_error_already:
                    LOGGER.error(f"Was not able to translate {book}.  Error: {e.args[0]}")
                    displayed_error_already = True
                else:
                    LOGGER.error(f"Was not able to translate {book}.")
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def translate_text_files(
        self,
        src_prefix: str,
        trg_prefix: str,
        start_seq: int,
        end_seq: int,
        src_iso: Optional[str],
        trg_iso: Optional[str],
    ) -> None:
        translator, config, _ = self._init_translation_task(experiment_suffix=f"_{self.checkpoint}_{src_prefix}")
        if trg_prefix is None:
            raise RuntimeError("A target file prefix must be specified.")
        if start_seq is None or end_seq is None:
            raise RuntimeError("Start and end sequence numbers must be specified.")

        if src_iso is None:
            src_iso = config.default_src_iso
        if trg_iso is None:
            trg_iso = config.default_trg_iso

        cwd = Path.cwd()
        for i in range(start_seq, end_seq + 1):
            file_num = f"{i:04d}"
            src_file_path = cwd / f"{src_prefix}{file_num}.txt"
            trg_file_path = cwd / f"{trg_prefix}{file_num}.txt"
            if src_file_path.is_file() and not trg_file_path.is_file():
                start = time.time()
                translator.translate_text(src_file_path, trg_file_path, src_iso, trg_iso)
                end = time.time()
                print(f"Translated {src_file_path.name} to {trg_file_path.name} in {((end-start)/60):.2f} minutes")
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def translate_files(self, src: str, trg: Optional[str], src_iso: Optional[str], trg_iso: Optional[str]) -> None:
        translator, config, step_str = self._init_translation_task(
            experiment_suffix=f"_{self.checkpoint}_{os.path.basename(src)}"
        )

        if src_iso is None:
            src_iso = config.default_src_iso
        if trg_iso is None:
            trg_iso = config.default_trg_iso

        src_path = Path(src)
        if not src_path.exists() and not src_path.is_absolute():
            src_path = SIL_NLP_ENV.data_dir / src
        if not src_path.exists():
            raise FileNotFoundError("Cannot find source: " + src)

        if trg is not None:
            trg_path = Path(trg)
            if not trg_path.exists() and not trg_path.is_absolute():
                trg_path = SIL_NLP_ENV.data_dir / trg
            if not trg_path.exists() and ((src_path.is_file() and not trg_path.parent.is_dir()) or src_path.is_dir()):
                raise FileNotFoundError("Cannot find target: " + trg)

        else:
            trg_path = config.exp_dir / "infer" / step_str
            trg_path.mkdir(exist_ok=True, parents=True)

        if src_path.is_file():
            src_file_paths = [src_path]
        else:
            src_file_paths = list(p for p in src_path.rglob("*.*") if p.is_file())

        for src_file_path in src_file_paths:
            if trg_path.is_dir():
                if src_path.is_file():
                    trg_file_path = trg_path / src_file_path.name
                    src_name = src_file_path.name
                else:
                    relative_path = src_file_path.relative_to(src_path)
                    trg_file_path = trg_path / relative_path
                    trg_file_path.parent.mkdir(exist_ok=True, parents=True)
                    src_name = str(relative_path)
            else:
                trg_file_path = trg_path
                src_name = src_file_path.name

            ext = src_file_path.suffix.lower()
            LOGGER.info(f"Translating {src_name}")
            if ext == ".txt":
                translator.translate_text(src_file_path, trg_file_path, src_iso, trg_iso)
            elif ext == ".docx":
                translator.translate_docx(src_file_path, trg_file_path, src_iso, trg_iso)
            elif ext == ".usfm" or ext == ".sfm":
                translator.translate_usfm(src_file_path, trg_file_path, src_iso, trg_iso)
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def _init_translation_task(self, experiment_suffix: str) -> Tuple[Translator, Config, str]:
        clearml = SILClearML(
            self.name,
            self.clearml_queue,
            project_suffix="_infer",
            experiment_suffix=experiment_suffix,
        )
        self.name = clearml.name

        SIL_NLP_ENV.copy_experiment_from_bucket(
            self.name, patterns=("*.vocab", "*.model", "*.yml", "dict.*.txt", "*.json", "checkpoint", "ckpt*.index")
        )

        clearml.config.set_seed()

        model = clearml.config.create_model()
        translator = NMTTranslator(model, self.checkpoint)
        if clearml.config.model_dir.exists():
            checkpoint_path, step = model.get_checkpoint_path(self.checkpoint)
            SIL_NLP_ENV.copy_experiment_from_bucket(
                self.name, patterns=SIL_NLP_ENV.get_source_experiment_path(checkpoint_path) + "/*.*"
            )
            step_str = "avg" if step == -1 else str(step)
        else:
            step_str = "last"
        return translator, clearml.config, step_str


def api(obj):
    return [name for name in dir(obj) if name[0] != "_"]


def attrs(obj):
    disallowed_properties = {
        name for name, value in getmembers(type(obj)) if isinstance(value, (property, FunctionType))
    }
    return {name: getattr(obj, name) for name in api(obj) if name not in disallowed_properties and hasattr(obj, name)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable TensorFlow memory growth")
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
    parser.add_argument("--src-iso", default=None, type=str, help="The source language (iso code) to translate from")
    parser.add_argument("--trg-iso", default=None, type=str, help="The target language (iso code) to translate to")
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

    pprint(attrs(SIL_NLP_ENV))
    print("Press Ctrl+C to exit. Will continue automatically in 30 seconds.")
    time.sleep(30)
    
    translator = TranslationTask(
        name=args.experiment,
        checkpoint=args.checkpoint,
        clearml_queue=args.clearml_queue,
    )

    if len(args.books) > 0:
        translator.translate_books(args.books, args.src_project, args.trg_iso)
    elif args.src_prefix is not None:
        translator.translate_text_files(
            args.src_prefix, args.trg_prefix, args.start_seq, args.end_seq, args.src_iso, args.trg_iso
        )
    elif args.src is not None:
        translator.translate_files(args.src, args.trg, args.src_iso, args.trg_iso)
    else:
        raise RuntimeError("A Scripture book, file, or file prefix must be specified.")


if __name__ == "__main__":
    main()
