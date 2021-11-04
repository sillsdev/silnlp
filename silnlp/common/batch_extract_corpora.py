import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from zipfile import ZipFile

from tqdm.contrib.concurrent import process_map

logging.disable(logging.CRITICAL)
from ..common.corpus import count_lines, write_corpus
from ..common.environment import SIL_NLP_ENV
from .paratext import extract_project


def extract_bundle(args: Tuple[Path, Path, bytes, int]) -> Tuple[str, Optional[str]]:
    bundle_path, output_path, password, expected_verse_count = args
    with TemporaryDirectory() as temp_dir:
        project = bundle_path.stem
        project_dir = Path(temp_dir) / project
        with ZipFile(bundle_path, "r") as bundle_file:
            bundle_file.extractall(project_dir, pwd=password)

        try:
            corpus_path, verse_count = extract_project(project_dir, output_path)
            if verse_count != expected_verse_count:
                corpus_path.unlink()
                return project, f"The number of verses is {verse_count}, but should be {expected_verse_count}."
            return project, None
        except Exception as err:
            return project, str(err)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts text corpora from a folder of Paratext project bundles")
    parser.add_argument("--input", type=str, help="The input bundle folder.")
    parser.add_argument("--output", type=str, help="The output corpus folder.")
    parser.add_argument("--password", type=str, help="The bundle password.")
    parser.add_argument("--error-log", type=str, default="", help="The error log file.")
    args = parser.parse_args()

    input = Path(args.input)
    output = Path(args.output)
    password = bytes(args.password, "utf-8")

    expected_verse_count = count_lines(SIL_NLP_ENV.assets_dir / "vref.txt")
    work = [(p, output, password, expected_verse_count) for p in input.glob("*.p8z")]
    print(f"Extracting {len(work)} projects...")
    errors: List[Tuple[str, str]] = []
    for project, msg in process_map(extract_bundle, work, unit="project", max_workers=cpu_count(), chunksize=1):
        if msg is not None:
            errors.append((project, msg))
    print(f"{len(work) - len(errors)} projects successfully extracted.")

    if args.error_log is not None:
        write_corpus(Path(args.error_log), (f"{project}\t{msg}" for project, msg in errors))


if __name__ == "__main__":
    main()
