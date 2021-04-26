import argparse
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

logging.basicConfig()

from ..common.canon import book_id_to_number
from ..common.paratext import book_file_name_digits
from ..common.translator import Translator
from ..common.utils import get_git_revision_hash
from .config import Config, create_runner, load_config
from .langs_config import LangsConfig
from .utils import decode_sp_lines, encode_sp, get_best_model_dir, get_last_checkpoint


class NMTTranslator(Translator):
    def __init__(self, config: Config, checkpoint_path: Path, memory_growth: bool) -> None:
        self._runner = create_runner(config, memory_growth=memory_growth)
        self._src_spp = config.create_src_sp_processor()
        self._checkpoint_path = checkpoint_path
        self._multiple_trg_langs = len(config.trg_isos) > 1
        self._default_trg_iso = config.default_trg_iso

    def translate(
        self, sentences: Iterable[str], src_iso: Optional[str] = None, trg_iso: Optional[str] = None
    ) -> Iterable[str]:
        features_list = [encode_sp(self._src_spp, self._insert_lang_tag(s, trg_iso)) for s in sentences]
        translations = self._runner.infer_list(features_list, checkpoint_path=str(self._checkpoint_path))
        return decode_sp_lines(t[0] for t in translations)

    def _insert_lang_tag(self, text: str, trg_iso: Optional[str]) -> str:
        if self._multiple_trg_langs:
            if trg_iso is None:
                trg_iso = self._default_trg_iso
            return f"<2{trg_iso}> {text}"
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use (last, best, avg, or checkpoint #)")
    parser.add_argument("--src-prefix", default=None, type=str, help="Source file prefix (e.g., de-news2019-)")
    parser.add_argument("--trg-prefix", default=None, type=str, help="Target file prefix (e.g., en-news2019-)")
    parser.add_argument("--start-seq", default=None, type=int, help="Starting file sequence #")
    parser.add_argument("--end-seq", default=None, type=int, help="Ending file sequence #")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument("--book", default=None, type=str, help="The book to translate")
    parser.add_argument("--trg-lang", default=None, type=str, help="The target language to translate into")
    parser.add_argument("--output-usfm", default=None, type=str, help="The output USFM file path")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    config = load_config(exp_name)

    config.set_seed()

    checkpoint: Optional[str] = args.checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint.lower()
    if checkpoint is None or checkpoint == "last":
        checkpoint_path, step = get_last_checkpoint(config.model_dir)
    elif checkpoint == "best":
        best_model_path, step = get_best_model_dir(config.model_dir)
        checkpoint_path = best_model_path / "ckpt"
    elif checkpoint == "avg":
        checkpoint_path, _ = get_last_checkpoint(config.model_dir / "avg")
        step = -1
    else:
        checkpoint_path = config.model_dir / f"ckpt-{checkpoint}"
        step = int(checkpoint)

    translator = NMTTranslator(config, checkpoint_path, args.memory_growth)
    trg_iso: Optional[str] = args.trg_lang
    book: Optional[str] = args.book
    if book is not None:
        src_project: Optional[str] = args.src_project
        if src_project is None:
            if not isinstance(config, LangsConfig) or len(config.src_projects) > 1:
                raise RuntimeError("A source project must be specified.")
            src_project = next(iter(config.src_projects))

        step_str = "avg" if step == -1 else str(step)
        default_output_dir = config.exp_dir / "infer" / step_str
        output_path: Optional[Path] = None if args.output_usfm is None else Path(args.output_usfm)
        if output_path is None:
            book_num = book_id_to_number(book)
            output_path = default_output_dir / f"{book_file_name_digits(book_num)}{book}.SFM"
        elif output_path.name == output_path:
            output_path = default_output_dir / output_path

        output_path.parent.mkdir(exist_ok=True, parents=True)
        translator.translate_book(src_project, book, output_path, trg_iso=trg_iso)
    elif args.src_prefix is not None:
        if args.trg_prefix is None:
            raise RuntimeError("A target file prefix must be specified.")
        if args.start_seq is None or args.end_seq is None:
            raise RuntimeError("Start and end sequence numbers must be specified.")

        checkpoint_name = "averaged checkpoint" if step == -1 else f"checkpoint {step}"
        print(f"Translating using {checkpoint_name}...")
        cwd = Path.cwd()
        for i in range(args.start_seq, args.end_seq + 1):
            file_num = f"{i:04d}"
            src_file_path = cwd / f"{args.src_prefix}{file_num}.txt"
            trg_file_path = cwd / f"{args.trg_prefix}{file_num}.txt"
            if src_file_path.is_file() and not trg_file_path.is_file():
                start = time.time()
                translator.translate_text_file(src_file_path, trg_file_path, trg_iso=trg_iso)
                end = time.time()
                print(f"Translated {src_file_path.name} to {trg_file_path.name} in {((end-start)/60):.2f} minutes")
    else:
        raise RuntimeError("A Scripture book or source file prefix must be specified.")


if __name__ == "__main__":
    main()
