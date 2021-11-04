import argparse
import time
from pathlib import Path
from typing import Iterable, Optional, Dict
from dataclasses import dataclass

import tensorflow as tf
from machine.scripture import book_id_to_number

from ..common.clearml import SILClearML
from ..common.environment import SIL_NLP_ENV
from ..common.paratext import book_file_name_digits
from ..common.translator import Translator
from ..common.utils import get_git_revision_hash
from .config import Config, create_runner, load_config, get_checkpoint_path
from .utils import decode_sp_lines, encode_sp, get_best_model_dir, get_last_checkpoint


@dataclass
class NMTTranslator(Translator):
    name: str
    checkpoint: str = "last"
    memory_growth: bool = False
    clearml_queue: str = None
    experiment_suffix: str = ""

    def __post_init__(self):
        self.clearml = SILClearML(
            self.name, self.clearml_queue, project_suffix="_infer", experiment_suffix=self.experiment_suffix
        )
        self.name = self.clearml.get_remote_name()
        self.config: Config = self.clearml.load_config()
        self.config.set_seed()

        if self.checkpoint is None:
            self.checkpoint = "last"
        self.checkpoint_path, step = get_checkpoint_path(self.config.model_dir, self.checkpoint)

        self._runner = create_runner(self.config, memory_growth=self.memory_growth)
        self._src_spp = self.config.create_src_sp_processor()
        self._step_str = "avg" if step == -1 else str(step)

        self._multiple_trg_langs = len(self.config.trg_isos) > 1
        self._default_trg_iso = self.config.default_trg_iso

    def translate(
        self, sentences: Iterable[str], src_iso: Optional[str] = None, trg_iso: Optional[str] = None
    ) -> Iterable[str]:
        features_list = [encode_sp(self._src_spp, self._insert_lang_tag(s, trg_iso)) for s in sentences]
        translations = self._runner.infer_list(features_list, checkpoint_path=str(self.checkpoint_path))
        return decode_sp_lines(t[0] for t in translations)

    def _insert_lang_tag(self, text: str, trg_iso: Optional[str]) -> str:
        if self._multiple_trg_langs:
            if trg_iso is None:
                trg_iso = self._default_trg_iso
            return f"<2{trg_iso}> {text}"
        return text

    def translate_book_by_step(self, book, src_project=None, output_usfm=None, trg_lang=None):
        if src_project is None:
            if len(self.config.src_projects) != 1:
                raise RuntimeError("A source project must be specified.")
            src_project = next(iter(self.config.src_projects))

        default_output_dir = self.config.exp_dir / "infer" / self._step_str
        output_path: Optional[Path] = None if output_usfm is None else Path(output_usfm)
        if output_path is None:
            book_num = book_id_to_number(book)
            output_path = default_output_dir / f"{book_file_name_digits(book_num)}{book}.SFM"
        elif output_path.name == output_path:
            output_path = default_output_dir / output_path

        output_path.parent.mkdir(exist_ok=True, parents=True)
        self.translate_book(src_project, book, output_path, trg_iso=trg_lang)
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def translate_parts(self, src_prefix, trg_prefix, start_seq, end_seq, trg_lang=None):
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
                self.translate_text_file(src_file_path, trg_file_path, trg_iso=trg_lang)
                end = time.time()
                print(f"Translated {src_file_path.name} to {trg_file_path.name} in {((end-start)/60):.2f} minutes")
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def translate_src(self, src_file_path, trg_lang=None, trg_file_path=None):
        src_file_path = Path(src_file_path)
        default_output_dir = self.config.exp_dir / "infer" / self._step_str
        trg_file_path = default_output_dir / src_file_path.name if trg_file_path is None else Path(trg_file_path)
        trg_file_path.parent.mkdir(exist_ok=True, parents=True)
        self.translate_text_file(src_file_path, trg_file_path, trg_iso=trg_lang)
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
    parser.add_argument("--book", default=None, type=str, help="The book to translate")
    parser.add_argument("--trg-lang", default=None, type=str, help="The target language to translate into")
    parser.add_argument("--output-usfm", default=None, type=str, help="The output USFM file path")
    parser.add_argument(
        "--eager-execution",
        default=False,
        action="store_true",
        help="Enable TensorFlow eager execution.",
    )
    parser.add_argument(
        "--clearml_queue", default=None, type=str, help="Process the infer on ClearML on the specified queue."
    )
    args = parser.parse_args()

    get_git_revision_hash()

    if args.eager_execution:
        tf.config.run_functions_eagerly(True)

    translator = NMTTranslator(
        name=args.experiment,
        checkpoint=args.checkpoint,
        memory_growth=args.memory_growth,
        clearml_queue=args.clearml_queue,
    )

    if args.book is not None:
        translator.translate_book_by_step(
            args.book, src_project=args.src_project, output_usfm=args.output_usfm, trg_lang=args.trg_lang
        )
    elif args.src_prefix is not None:
        translator.translate_parts(args.src_prefix, args.trg_prefix, args.start_seq, args.end_seq, args.trg_lang)
    elif args.src is not None:
        translator.translate_src(args.src, args.trg_lang, args.trg)
    else:
        raise RuntimeError("A Scripture book, file, or file prefix must be specified.")


if __name__ == "__main__":
    main()
