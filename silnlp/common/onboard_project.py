import argparse
import getpass
import logging
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Tuple

import wildebeest.wb_analysis as wb_ana
import yaml

from machine.corpora import FileParatextProjectSettingsParser

from silnlp.common.clean_projects import process_single_project_for_cleaning

from ..nmt.config_utils import create_config
from .collect_verse_counts import collect_verse_counts
from .environment import SIL_NLP_ENV
from .extract_corpora import extract_corpora
from .iso_info import NLLB_TAG_FROM_ISO
from .iso_info import data as iso_data

LOGGER = logging.getLogger(__package__ + ".onboard_project")


def get_paratext_project_dir(project: str) -> Path:
    return SIL_NLP_ENV.pt_projects_dir / project


def create_paratext_project_folder_if_not_exists(project_name: str) -> Path:
    pt_project_path = get_paratext_project_dir(project_name)
    if pt_project_path.exists():
        LOGGER.info(f"Paratext project folder '{pt_project_path}' already exists.")
    else:
        LOGGER.info(f"Creating new Paratext project folder: {pt_project_path}")
        pt_project_path.mkdir()
    return pt_project_path


def _copy_file_to_paratext_project(source_file: Path, target_file: Path, overwrite=False) -> None:
    if target_file.exists() and not overwrite:
        LOGGER.info(f"File '{target_file}' already exists. Skipping.")
    else:
        target_file.write_bytes(source_file.read_bytes())


def _copy_directory_to_paratext_project(source_dir: Path, target_dir: Path, overwrite=False) -> None:
    if not target_dir.exists():
        target_dir.mkdir()
    for sub_item in source_dir.iterdir():
        target_item = target_dir / sub_item.name
        if sub_item.is_dir():
            _copy_directory_to_paratext_project(sub_item, target_item, overwrite)
        else:
            _copy_file_to_paratext_project(sub_item, target_item, overwrite)


def copy_paratext_project_folder(source_dir: Path, project_name: str, overwrite=False) -> None:
    pt_project_path = get_paratext_project_dir(project_name)

    if not any(source_dir.iterdir()):
        LOGGER.warning(f"Source directory '{source_dir}' is empty.")
        return

    for source_item in source_dir.iterdir():
        target_item = pt_project_path / source_item.name
        if source_item.is_dir():
            _copy_directory_to_paratext_project(source_item, target_item, overwrite=overwrite)
        else:
            _copy_file_to_paratext_project(source_item, target_item, overwrite=overwrite)


def collect_verse_counts_wrapper(project_name: str, verse_counts_config: dict, overwrite=False) -> None:

    output_folder = Path(
        verse_counts_config.get(
            "output_folder", SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name / "verse_counts"
        )
    )
    if output_folder.exists() and not overwrite:
        LOGGER.info(f"Verse counts output folder '{output_folder}' already exists. Skipping verse counts collection.")
        return
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    input_folder = verse_counts_config.get("input_folder", SIL_NLP_ENV.mt_scripture_dir)

    file_patterns = verse_counts_config.get("files", f"*{project_name}*.txt")

    input_folder_path = Path(input_folder)
    if not input_folder_path.exists():
        LOGGER.error(f"Input folder '{input_folder_path}' does not exist. Skipping verse counts collection.")
        return

    matched_files = list(input_folder_path.glob(file_patterns))
    if not matched_files:
        LOGGER.error(
            f"No files matching pattern '{file_patterns}' found in '{input_folder_path}'. Skipping verse counts collection."
        )
        return

    LOGGER.info(f"Collecting verse counts for project '{project_name}'")
    collect_verse_counts(
        input_folder=input_folder_path,
        output_folder=output_folder,
        file_patterns=file_patterns,
        deutero=verse_counts_config.get("deutero", False),
        recount=verse_counts_config.get("recount", False),
    )


def extract_corpora_wrapper(project_name: str, extract_config: dict, overwrite=False) -> None:
    extract_paths = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*{project_name}.txt"))
    extract_path = extract_paths[0] if extract_paths else None
    if extract_path is not None and not overwrite:
        LOGGER.info(f"Extracted corpus '{extract_path}' already exists. Skipping corpus extraction.")
        return
    LOGGER.info(f"Extracting corpora for project '{project_name}'")
    extract_corpora(
        projects={project_name},
        books_to_include=extract_config.get("include", []),
        books_to_exclude=extract_config.get("exclude", []),
        include_markers=extract_config.get("markers", False),
        extract_lemmas=extract_config.get("lemmas", False),
        extract_project_vrefs=extract_config.get("project-vrefs", False),
        extract_surface_forms=extract_config.get("surface-forms", False),
        parent_project=extract_config.get("parent_project", None),
        versification_error_output_path=SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name / "versification_errors.txt"
    )


def wildebeest_analysis_wrapper(project_name: str, wildebeest_config: dict, overwrite=False) -> None:
    wildebeest_output_dir = SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name / "wildebeest"
    if wildebeest_output_dir.exists() and not overwrite:
        LOGGER.info(
            f"Wildebeest output directory '{wildebeest_output_dir}' already exists. Skipping Wildebeest analysis."
        )
        return
    if not wildebeest_output_dir.exists():
        wildebeest_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        extract_path = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*{project_name}.txt"))[0]
    except IndexError:
        LOGGER.error(f"No extracted corpus found for project '{project_name}'. Skipping Wildebeest analysis.")
        return
    LOGGER.info(f"Running Wildebeest analysis on {extract_path}.")
    old_argv = sys.argv
    try:
        sys.argv = [
            "wb_ana",
            "-i",
            str(extract_path),
            "-j",
            f"{wildebeest_output_dir}/{project_name}_wildebeest_report.json",
            "-o",
            f"{wildebeest_output_dir}/{project_name}_wildebeest_report.txt",
            "-x",
            str(wildebeest_config.get("max_examples", 500)),
            "-n",
            str(wildebeest_config.get("max_cases", 500)),
            "-r",
            str(wildebeest_config.get("ref_id_file", "silnlp/assets/vref.txt")),
        ]
        wb_ana.main()
    finally:
        sys.argv = old_argv


def calculate_tokenization_stats(project_name: str, stats_config: dict = None, overwrite=False) -> None:
    stats_dir = SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name / "stats"

    if stats_dir.exists() and not overwrite:
        LOGGER.info(f"Stats directory '{stats_dir}' already exists. Skipping stats calculation.")
        return
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    try:
        extract_path = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*{project_name}*.txt"))[0]
    except IndexError:
        LOGGER.error(
            f"No extracted corpus found for project '{project_name}'. Skipping tokenization stats calculation."
        )
        return
    extract_file = extract_path.stem

    iso_code = extract_file.split("-")[0]

    iso_dict = {**{k: v for k, v in iso_data}, **{v: k for k, v in iso_data}}

    iso_code = iso_dict.get(iso_code, iso_code)
    nllb_tag = NLLB_TAG_FROM_ISO.get(iso_code, "eng_Latn")

    if stats_config is None:
        stats_config = {
            "data": {
                "corpus_pairs": [
                    {
                        "src": extract_file,
                        "trg": extract_file,
                        "type": "train",
                        "lang_codes": {iso_code: nllb_tag},
                    }
                ],
            },
        }

    LOGGER.info(f"Calculating tokenization stats for project '{project_name}'")
    config = create_config(exp_dir=stats_dir, config=stats_config)

    config.set_seed()
    config.preprocess(stats=True, force_align=True)


def get_config(config_path: str) -> dict:
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
        with config_file.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    else:
        return {}


def check_for_project_errors(copy_from: Path | None, project: str) -> None:
    if copy_from:
        if not copy_from.exists():
            raise FileNotFoundError(f"The specified --copy-from path '{copy_from}' does not exist.")
        project_path = copy_from / project
        settings_file = project_path / "Settings.xml"
        if not project_path.exists():
            raise FileNotFoundError(
                f"The specified project folder '{project_path}' does not exist in the --copy-from path."
            )
        if not settings_file.exists():
            raise FileNotFoundError(
                f"The Settings.xml file was not found in the project folder '{project_path}'. Please ensure this is a valid Paratext project folder."
            )
        
        settings = FileParatextProjectSettingsParser(project_path).parse()

        if settings.translation_type != "Standard":
            LOGGER.warning(
                f"{project} is a non-Standard project. Type is '{settings.translation_type}'."
            )

        book_part = re.sub(r"\d", "[0-9]", settings.file_name_form)
        book_part = re.sub(r"([A-Z])", r"[A-Z]", book_part)
        pattern = f"{settings.file_name_prefix}.*{book_part}.*{settings.file_name_suffix}"
        if not any([re.match(pattern, file.name) for file in project_path.iterdir()]):
            raise ValueError(
                f"{project_path} does not contain any files using the naming convention, '{pattern}', found in the Settings.xml file."
            )


def setup_local_project(
    project: str, copy_from: Path | None, zip_password: str, datestamp: bool
) -> Tuple[str, Path | None, Path | None]:
    if project.endswith(".zip") or project.endswith(".p8z"):
        with zipfile.ZipFile(copy_from / project, "r") as zip_ref:
            project = Path(project).stem
            needs_password = any(zinfo.flag_bits & 0x1 for zinfo in zip_ref.infolist())
            if needs_password:
                if zip_password:
                    pwd = zip_password
                if not pwd:
                    pwd = getpass.getpass(prompt=f"Enter password for {project}: ")
                zip_ref.extractall(copy_from / project, pwd=pwd.encode())
            else:
                zip_ref.extractall(copy_from / project)

    check_for_project_errors(copy_from, project)

    project_name = project
    local_project_path = copy_from / project if copy_from else None

    if "-" in project_name:
        LOGGER.info(f"Project name '{project_name}' contains hyphens. Replacing hyphens with underscores.")
        project_name = project_name.replace("-", "_")
        LOGGER.info(f"New project name: '{project_name}'")
    if datestamp:
        now = datetime.now()
        datestamp = now.strftime("%Y_%m_%d")
        project_name = f"{project_name}_{datestamp}"
        LOGGER.info(f"Datestamping project. New project name: {project_name}")

        # Copy local project folder contents to a folder with the new name
    if local_project_path and local_project_path.exists() and local_project_path.name != project_name:
        new_local_project_path = local_project_path.parent / project_name
        if not new_local_project_path.exists():
            new_local_project_path.mkdir(parents=True, exist_ok=True)
        _copy_directory_to_paratext_project(local_project_path, new_local_project_path, overwrite=True)
        local_project_path = new_local_project_path

    return project_name, local_project_path, copy_from


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performs several steps to onboard a new project before training a model.",
    )

    parser.add_argument(
        "projects",
        help="Paratext project name. The project will be stored on the bucket at Paratext/projects/<project>.",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--copy-from",
        help="Path to a downloaded Paratext project folder. The local project will be copied to the bucket. If provided without a value, uses the user's Downloads directory.",
        nargs="?",
        const=Path.home() / "Downloads",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--config",
        help="Path to a configuration file in YAML format. This is used to configure the onboarding process.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--overwrite", help="Overwrite any existing files and folders", default=False, action="store_true"
    )

    parser.add_argument(
        "--extract-corpora",
        default=False,
        action="store_true",
        help="Extract text corpora from the Paratext project.",
    )

    parser.add_argument(
        "--collect-verse-counts",
        default=False,
        action="store_true",
        help="Collect various counts from the extracted Paratext project.",
    )
    parser.add_argument(
        "--no-clean",
        default=False,
        action="store_true",
        help="Skips cleaning the Paratext project folder.",
    )
    parser.add_argument(
        "--datestamp",
        default=False,
        action="store_true",
        help="Add a datestamp to the project folder name when creating a new Paratext project folder.",
    )
    parser.add_argument(
        "--wildebeest", default=False, action="store_true", help="Run Wildebeest analysis on the extracted corpora."
    )
    parser.add_argument("--stats", default=False, action="store_true", help="Compute tokenization statistics")

    args = parser.parse_args()
    config = get_config(args.config) if args.config else {}

    for project in args.projects:
        pwd = config.get("zip_password", None)
        project_name, local_project_path, copy_from = setup_local_project(project, args.copy_from, pwd, args.datestamp)

        if not args.no_clean:
            LOGGER.info(f"Cleaning Paratext project: {project_name}.")
            process_single_project_for_cleaning(
                local_project_path,
            )

        if copy_from:
            LOGGER.info(
                f"Copying project: {project_name} from {copy_from} to {SIL_NLP_ENV.pt_projects_dir}/{project_name}"
            )
            source_path = Path(copy_from)
            if source_path.name != project_name:
                source_path = Path(source_path / project_name)
            paratext_project_dir: Path = create_paratext_project_folder_if_not_exists(project_name)
            copy_paratext_project_folder(source_path, paratext_project_dir, overwrite=args.overwrite)
            if project_name != project:
                shutil.rmtree(source_path)

        if args.extract_corpora:
            extract_config: dict = config.get("extract_corpora", {})
            extract_corpora_wrapper(project_name, extract_config, args.overwrite)

        if args.collect_verse_counts:
            if not args.extract_corpora:
                LOGGER.warning(
                    "--extract_corpora was not included. Collecting verse counts requires the corpus to be extracted first."
                )
            LOGGER.info(f"Collecting verse counts from {project_name}.")
            collect_verse_counts_wrapper(project_name, config.get("verse_counts", {}), args.overwrite)

        if args.wildebeest:
            if not args.extract_corpora:
                LOGGER.warning(
                    "--extract_corpora was not included. Wildebeest requires the corpus to be extracted first."
                )
            wildebeest_config: dict = config.get("wildebeest", {})
            wildebeest_analysis_wrapper(project_name, wildebeest_config, args.overwrite)

        if args.stats:
            if not args.extract_corpora:
                LOGGER.warning("--extract_corpora was not included. Stats requires the corpus to be extracted first.")
            calculate_tokenization_stats(project_name, config.get("stats", None), args.overwrite)


if __name__ == "__main__":
    main()
