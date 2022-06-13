import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import tensorflow as tf
from machine.scripture import book_number_to_id, get_books

from ..common.clearml_connection import SILClearML
from ..common.environment import SIL_NLP_ENV
from ..common.paratext import book_file_name_digits
from ..common.translator import Translator
from ..common.utils import get_git_revision_hash
from .config import Config, create_runner, get_checkpoint_path
from .utils import decode_sp_lines, enable_memory_growth, encode_sp

LOGGER = logging.getLogger(__name__)


class NMTTranslator(Translator):
    def __init__(self, config: Config, checkpoint_path: Optional[Path]):
        self._multiple_trg_isos = len(config.trg_isos) > 1
        self._default_trg_iso = config.default_trg_iso
        self.checkpoint_path = checkpoint_path
        self._runner = create_runner(config)
        self._src_spp = config.create_src_sp_processor()

    def translate(
        self,
        sentences: Iterable[Union[str, List[str]]],
        src_iso: Optional[str] = None,
        trg_iso: Optional[str] = None,
    ) -> Iterable[str]:
        features_list: List[List[str]] = [[]]
        for sentence in sentences:
            if isinstance(sentence, str):
                features_list[0].append(encode_sp(self._src_spp, self._insert_lang_tag(sentence, trg_iso)))
            else:
                features_list[0].append(encode_sp(self._src_spp, self._insert_lang_tag(sentence[0], trg_iso)))
                for i in range(1, len(sentence)):
                    if i == len(features_list):
                        features_list.append([])
                    features_list[i].append(sentence[i])
        translations = self._runner.infer_list(
            features_list, checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path is not None else None
        )
        return decode_sp_lines(t[0] for t in translations)

    def _insert_lang_tag(self, text: str, trg_iso: Optional[str]) -> str:
        if self._multiple_trg_isos:
            if trg_iso is None:
                trg_iso = self._default_trg_iso
            return f"<2{trg_iso}> {text}"
        return text


@dataclass
class TranslationTask:
    name: str
    checkpoint: str = "last"
    clearml_queue: Optional[str] = None
    clearml: Optional[SILClearML] = None
    name: Optional[str] = None
    config: Optional[Config] = None

    def __post_init__(self):
        if self.checkpoint is None:
            self.checkpoint = "last"
        self.translator = None

    def init_translation_task(self, experiment_suffix: str):
        self.clearml = SILClearML(
            self.name,
            self.clearml_queue,
            project_suffix="_infer",
            experiment_suffix=experiment_suffix,
        )
        self.name: str = self.clearml.name
        self.config: Config = self.clearml.config

        SIL_NLP_ENV.copy_experiment_from_bucket(
            self.name, extensions=(".vocab", ".model", ".yml", "dict.src.txt", "dict.trg.txt")
        )

        self.config.set_seed()

        checkpoint_path, step = get_checkpoint_path(self.config.model_dir, self.checkpoint)
        self.translator = NMTTranslator(config=self.config, checkpoint_path=checkpoint_path)
        self._step_str = "avg" if step == -1 else str(step)

    def translate_books(
        self,
        books: str,
        src_project: Optional[str] = None,
        trg_iso: Optional[str] = None,
    ):
        self.init_translation_task(experiment_suffix=f"_{self.checkpoint}_{books}")
        book_nums = get_books(books)

        if src_project is None:
            if len(self.config.src_projects) != 1:
                raise RuntimeError("A source project must be specified.")
            src_project = next(iter(self.config.src_projects))

        output_dir = self.config.exp_dir / "infer" / self._step_str
        output_dir.mkdir(exist_ok=True, parents=True)

        displayed_error_already = False
        for book_num in book_nums:
            book = book_number_to_id(book_num)
            output_path = output_dir / f"{book_file_name_digits(book_num)}{book}.SFM"
            try:
                LOGGER.info(f"Translating {book} ...")
                self.translator.translate_book(src_project, book, output_path, trg_iso=trg_iso)
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
        self.init_translation_task(experiment_suffix=f"_{self.checkpoint}_{src_prefix}")
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
                self.translator.translate_text_file(src_file_path, trg_file_path, trg_iso=trg_iso)
                end = time.time()
                print(f"Translated {src_file_path.name} to {trg_file_path.name} in {((end-start)/60):.2f} minutes")
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def translate_text_file(self, src_file_path, trg_iso=None, trg_file_path=None):
        self.init_translation_task(experiment_suffix=f"_{self.checkpoint}_{os.path.basename(src_file_path)}")
        if(Path(src_file_path).exists()):
            src_file_path = Path(src_file_path)
        elif (SIL_NLP_ENV.data_dir / src_file_path).exists():
            src_file_path = SIL_NLP_ENV.data_dir / src_file_path
        else:
            raise FileNotFoundError("Cannot find: " + src_file_path)

        default_output_dir = self.config.exp_dir / "infer" / self._step_str
        trg_file_path = default_output_dir / src_file_path.name if trg_file_path is None else Path(trg_file_path)
        trg_file_path.parent.mkdir(exist_ok=True, parents=True)
        self.translator.translate_text_file(src_file_path, trg_file_path, trg_iso=trg_iso)
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
        "--clearml-queue", default=None, type=str, help="Process the infer on ClearML on the specified queue."
    )
    args = parser.parse_args()

    get_git_revision_hash()

    if args.eager_execution:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

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
