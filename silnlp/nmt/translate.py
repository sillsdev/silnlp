import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from machine.scripture import VerseRef, book_number_to_id, get_chapters

from ..common.environment import SIL_NLP_ENV
from ..common.paratext import book_file_name_digits, get_project_dir
from ..common.translator import TranslationGroup, Translator
from ..common.utils import get_git_revision_hash, show_attrs
from .clearml_connection import SILClearML
from .config import CheckpointType, Config, NMTModel

LOGGER = logging.getLogger(__package__ + ".translate")


class NMTTranslator(Translator):
    def __init__(self, model: NMTModel, checkpoint: Union[CheckpointType, str, int]) -> None:
        self._model = model
        self._checkpoint = checkpoint

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        vrefs: Optional[Iterable[VerseRef]] = None,
    ) -> Iterable[TranslationGroup]:
        return self._model.translate(
            sentences, src_iso, trg_iso, produce_multiple_translations, vrefs, self._checkpoint
        )


@dataclass
class TranslationTask:
    name: str
    checkpoint: Union[str, int] = "last"
    clearml_queue: Optional[str] = None
    commit: Optional[str] = None

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            self.checkpoint = "last"

    def translate_books(
        self,
        books: str,
        src_project: Optional[str],
        trg_project: Optional[str],
        trg_iso: Optional[str],
        produce_multiple_translations: bool = False,
        include_paragraph_markers: bool = False,
        include_style_markers: bool = False,
        include_embeds: bool = False,
    ):
        book_nums = get_chapters(books)
        translator, config, step_str = self._init_translation_task(
            experiment_suffix=f"_{self.checkpoint}_{[book_number_to_id(book) for book in book_nums.keys()]}"
        )

        if src_project is None:
            if len(config.src_projects) != 1:
                raise RuntimeError("A source project must be specified.")
            src_project = next(iter(config.src_projects))

        src_project_dir = get_project_dir(src_project)
        if not src_project_dir.is_dir():
            raise FileNotFoundError(f"Source project {src_project} not found in projects folder {src_project_dir}")

        if any(len(book_nums[book]) > 0 for book in book_nums) and trg_project is not None:

            trg_project_dir = get_project_dir(trg_project)
            if not trg_project_dir.is_dir():
                raise FileNotFoundError(f"Target project {trg_project} not found in projects folder {trg_project_dir}")
        else:
            trg_project = None

        if trg_iso is None:
            trg_iso = config.default_test_trg_iso
            if trg_iso == "" and len(config.trg_isos) > 0:
                trg_iso = next(iter(config.trg_isos))
        if trg_iso == "":
            LOGGER.warning("No language code was set for the target language")

        output_dir = config.exp_dir / "infer" / step_str / src_project
        if not config.model_dir.exists():
            output_dir = config.exp_dir / "infer" / "base" / src_project
        if trg_project is not None:
            output_dir = output_dir / trg_project
        output_dir.mkdir(exist_ok=True, parents=True)

        experiment_ckpt_str = f"{self.name}:{self.checkpoint}"
        if not config.model_dir.exists():
            experiment_ckpt_str = f"{self.name}:base"

        translation_failed = []
        for book_num, chapters in book_nums.items():
            book = book_number_to_id(book_num)
            try:
                LOGGER.info(f"Translating {book} ...")
                output_path = output_dir / f"{book_file_name_digits(book_num)}{book}.SFM"
                translator.translate_book(
                    src_project,
                    book,
                    output_path,
                    trg_iso,
                    produce_multiple_translations,
                    chapters,
                    trg_project,
                    include_paragraph_markers,
                    include_style_markers,
                    include_embeds,
                    experiment_ckpt_str,
                )
            except Exception as e:
                translation_failed.append(book)
                LOGGER.exception(f"Was not able to translate {book}.")

        if len(translation_failed) > 0:
            raise RuntimeError(f"Some books failed to translate: {' '.join(translation_failed)}")

    def translate_text_files(
        self,
        src_prefix: str,
        trg_prefix: str,
        start_seq: int,
        end_seq: int,
        src_iso: Optional[str],
        trg_iso: Optional[str],
        produce_multiple_translations: bool = False,
    ) -> None:
        translator, config, _ = self._init_translation_task(
            experiment_suffix=f"_{self.checkpoint}_{src_prefix}", patterns=[src_prefix, trg_prefix]
        )
        if trg_prefix is None:
            raise RuntimeError("A target file prefix must be specified.")
        if start_seq is None or end_seq is None:
            raise RuntimeError("Start and end sequence numbers must be specified.")

        if src_iso is None:
            src_iso = config.default_test_src_iso
            if src_iso == "" and len(config.src_iso) > 0:
                src_iso = next(iter(config.src_iso))
        if src_iso == "":
            LOGGER.warning("No language code was set for the source language")
        if trg_iso is None:
            trg_iso = config.default_test_trg_iso
            if trg_iso == "" and len(config.trg_isos) > 0:
                trg_iso = next(iter(config.trg_isos))
        if trg_iso == "":
            LOGGER.warning("No language code was set for the target language")

        for i in range(start_seq, end_seq + 1):
            file_num = f"{i:04d}"
            src_file = f"{src_prefix}{file_num}.txt"
            src_file_path = Path(SIL_NLP_ENV.mt_experiments_dir / self.name / src_file)
            if not src_file_path.exists():
                raise FileNotFoundError("Cannot find source: " + src_file)

            trg_file = f"{trg_prefix}{file_num}.txt"
            trg_file_path = Path(SIL_NLP_ENV.mt_experiments_dir / self.name / trg_file)

            if src_file_path.is_file() and not trg_file_path.is_file():
                start = time.time()
                translator.translate_text(src_file_path, trg_file_path, src_iso, trg_iso, produce_multiple_translations)
                end = time.time()
                print(f"Translated {src_file_path.name} to {trg_file_path.name} in {((end-start)/60):.2f} minutes")

    def translate_files(
        self,
        src: str,
        trg: Optional[str],
        src_iso: Optional[str],
        trg_iso: Optional[str],
        produce_multiple_translations: bool = False,
        include_paragraph_markers: bool = False,
        include_style_markers: bool = False,
        include_embeds: bool = False,
    ) -> None:
        translator, config, step_str = self._init_translation_task(
            experiment_suffix=f"_{self.checkpoint}_{os.path.basename(src)}",
            patterns=[src, trg] if trg is not None else [src],
        )

        if src_iso is None:
            src_iso = config.default_test_src_iso
            if src_iso == "" and len(config.src_iso) > 0:
                src_iso = next(iter(config.src_iso))
        if src_iso == "":
            LOGGER.warning("No language code was set for the source language")
        if trg_iso is None:
            trg_iso = config.default_test_trg_iso
            if trg_iso == "" and len(config.trg_isos) > 0:
                trg_iso = next(iter(config.trg_isos))
        if trg_iso == "":
            LOGGER.warning("No language code was set for the target language")

        src_path = Path(SIL_NLP_ENV.mt_experiments_dir / self.name / src)
        if not src_path.exists():
            raise FileNotFoundError(f"Cannot find source: {src} in {self.name}")

        if trg is not None:
            trg_path = Path(SIL_NLP_ENV.mt_experiments_dir / self.name / trg)
        else:
            trg_path = config.exp_dir / "infer" / step_str
            if not config.model_dir.exists():
                trg_path = config.exp_dir / "infer" / "base"
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
                trg_path.parent.mkdir(parents=True, exist_ok=True)
                trg_file_path = trg_path
                src_name = src_file_path.name

            ext = src_file_path.suffix.lower()
            LOGGER.info(f"Translating {src_name}")
            if ext == ".txt":
                translator.translate_text(src_file_path, trg_file_path, src_iso, trg_iso, produce_multiple_translations)
            elif ext == ".docx":
                translator.translate_docx(src_file_path, trg_file_path, src_iso, trg_iso, produce_multiple_translations)
            elif ext == ".usfm" or ext == ".sfm":
                experiment_ckpt_str = f"{self.name}:{self.checkpoint}"
                if not config.model_dir.exists():
                    experiment_ckpt_str = f"{self.name}:base"
                translator.translate_usfm(
                    src_file_path,
                    trg_file_path,
                    src_iso,
                    trg_iso,
                    produce_multiple_translations,
                    include_paragraph_markers=include_paragraph_markers,
                    include_style_markers=include_style_markers,
                    include_embeds=include_embeds,
                    experiment_ckpt_str=experiment_ckpt_str,
                )

    def _init_translation_task(self, experiment_suffix: str) -> Tuple[Translator, Config, str]:
        clearml = SILClearML(
            self.name,
            self.clearml_queue,
            project_suffix="_infer",
            experiment_suffix=experiment_suffix,
            commit=self.commit,
        )
        self.name = clearml.name

        clearml.config.set_seed()

        model = clearml.config.create_model()
        translator = NMTTranslator(model, self.checkpoint)
        if clearml.config.model_dir.exists():
            checkpoint_path, step = model.get_checkpoint_path(self.checkpoint)
            step_str = "avg" if step == -1 else str(step)
        else:
            step_str = "last"
        return translator, clearml.config, step_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use (last, best, avg, or checkpoint #)")
    parser.add_argument(
        "--src",
        default=None,
        type=str,
        help="Source file name, must be in the experiment directory",
    )
    parser.add_argument(
        "--trg",
        default=None,
        type=str,
        help="Target file name, must relative to the experiment directory",
    )
    parser.add_argument(
        "--src-prefix",
        default=None,
        type=str,
        help="Source file prefix (e.g., de-news2019-), must be in the experiment directory",
    )
    parser.add_argument(
        "--trg-prefix",
        default=None,
        type=str,
        help="Target file prefix (e.g., en-news2019-), must be relative to the experiment directory",
    )
    parser.add_argument("--start-seq", default=None, type=int, help="Starting file sequence #")
    parser.add_argument("--end-seq", default=None, type=int, help="Ending file sequence #")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument(
        "--trg-project",
        default=None,
        type=str,
        help="The target project to use as the output for chapters that aren't translated",
    )
    parser.add_argument(
        "--books",
        metavar="books",
        nargs="+",
        default=[],
        help="The books to translate; e.g., 'NT', 'OT', 'GEN,EXO', can also select chapters; e.g., 'MAT-REV;-LUK10-30', 'MAT1,2,3,5-11'",
    )
    parser.add_argument("--src-iso", default=None, type=str, help="The source language (iso code) to translate from")
    parser.add_argument("--trg-iso", default=None, type=str, help="The target language (iso code) to translate to")
    parser.add_argument(
        "--multiple-translations",
        default=False,
        action="store_true",
        help='Produce multiple translations of each verse. These will be saved in separate files with suffixes like ".1.txt", ".2.txt", etc.',
    )
    parser.add_argument(
        "--include-paragraph-markers",
        default=False,
        action="store_true",
        help="For files in USFM format, attempt to place paragraph markers in translated verses based on the source project's markers",
    )
    parser.add_argument(
        "--include-style-markers",
        default=False,
        action="store_true",
        help="For files in USFM format, attempt to place style markers in translated verses based on the source project's markers",
    )
    parser.add_argument(
        "--include-embeds",
        default=False,
        action="store_true",
        help="For files in USFM format, carry over embeds from the source project to the output without translating them",
    )
    parser.add_argument(
        "--include-inline-elements",
        default=False,
        action="store_true",
        help="Deprecated argument, equivalent to --include-embeds",
    )
    parser.add_argument(
        "--preserve-usfm-markers",
        default=False,
        action="store_true",
        help="Deprecated argument, equivalent to --include-paragraph-markers AND --include-style-markers",
    )
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Show information about the environment variables and arguments.",
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="The silnlp git commit id with which to run a remote job"
    )

    args = parser.parse_args()

    get_git_revision_hash()

    translator = TranslationTask(
        name=args.experiment, checkpoint=args.checkpoint, clearml_queue=args.clearml_queue, commit=args.commit
    )

    # For backwards compatibility
    if args.preserve_usfm_markers:
        args.include_paragraph_markers = True
        args.include_style_markers = True
    if args.include_inline_elements:
        args.include_embeds = True

    if len(args.books) > 0:
        if args.debug:
            show_attrs(cli_args=args, actions=[f"Will attempt to translate books {args.books} into {args.trg_iso}"])
            exit()
        translator.translate_books(
            ";".join(args.books),
            args.src_project,
            args.trg_project,
            args.trg_iso,
            args.multiple_translations,
            args.include_paragraph_markers,
            args.include_style_markers,
            args.include_embeds,
        )
    elif args.src_prefix is not None:
        if args.debug:
            show_attrs(
                cli_args=args,
                actions=[f"Will attempt to translate matching files from {args.src_iso} into {args.trg_iso}."],
            )
            exit()
        translator.translate_text_files(
            args.src_prefix,
            args.trg_prefix,
            args.start_seq,
            args.end_seq,
            args.src_iso,
            args.trg_iso,
            args.multiple_translations,
        )
    elif args.src is not None:
        if args.debug:
            show_attrs(
                cli_args=args,
                actions=[f"Will attempt to translate {args.src} from {args.src_iso} into {args.trg_iso}."],
            )
            exit()
        translator.translate_files(
            args.src,
            args.trg,
            args.src_iso,
            args.trg_iso,
            args.multiple_translations,
            args.include_paragraph_markers,
            args.include_style_markers,
            args.include_embeds,
        )
    else:
        raise RuntimeError("A Scripture book, file, or file prefix must be specified.")


if __name__ == "__main__":
    main()
