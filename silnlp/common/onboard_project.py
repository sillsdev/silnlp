import argparse
import getpass
import logging
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import wildebeest.wb_analysis as wb_ana
import yaml

from scripts.clean_projects import clean_projects

from .collect_verse_counts import collect_verse_counts
from .environment import SIL_NLP_ENV
from .extract_corpora import extract_corpora

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


def collect_verse_counts_wrapper(project_name: str, verse_counts_config: dict) -> None:

    output_folder = Path(
        verse_counts_config.get("output_folder", SIL_NLP_ENV.mt_experiments_dir / "verse_counts" / project_name)
    )
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

    collect_verse_counts(
        input_folder=input_folder_path,
        output_folder=output_folder,
        file_patterns=file_patterns,
        deutero=verse_counts_config.get("deutero", False),
        recount=verse_counts_config.get("recount", False),
    )


def get_config(config_path: str) -> dict:
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
        with config_file.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    else:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performs several steps to onboard a new project before training a model.",
    )

    parser.add_argument(
        "project",
        help="Paratext project name. The project will be stored on the bucket at Paratext/projects/<project>.",
        type=str,
    )
    parser.add_argument(
        "--copy-from",
        help="Path to a downloaded Paratext project folder. The local project will be copied to the bucket.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--config",
        help="Path to a configuration file in YAML format. This is used to configure the onboarding process.",
        default=None,
        type=str,
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
        "--clean-project",
        default=False,
        action="store_true",
        help="Cleans the Paratext project folder by removing unnecessary files and folders.",
    )
    parser.add_argument(
        "--timestamp",
        default=False,
        action="store_true",
        help="Add a timestamp to the project folder name when creating a new Paratext project folder.",
    )
    parser.add_argument(
        "--wildebeest", default=False, action="store_true", help="Run Wildebeest analysis on the extracted corpora."
    )

    args = parser.parse_args()
    if not args.project:
        raise ValueError("Project name is required. Please provide a valid Paratext project name using <project>.")

    config = get_config(args.config) if args.config else {}

    if args.project.endswith(".zip"):
        with zipfile.ZipFile(args.project, "r") as zip_ref:
            # Check if any file in the zip is encrypted
            temp_dir = tempfile.TemporaryDirectory()
            needs_password = any(zinfo.flag_bits & 0x1 for zinfo in zip_ref.infolist())
            if needs_password:
                pwd = getpass.getpass(prompt=f"Enter password for zip file '{args.project}': ")
                zip_ref.extractall(temp_dir.name, pwd=pwd.encode())
            else:
                zip_ref.extractall(temp_dir.name)
        args.copy_from = temp_dir.name
        args.project = Path(args.project).stem

    project_name = args.project
    if args.timestamp:

        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d")
        project_name = f"{args.project}_{timestamp}"
        LOGGER.info(f"Timestamping project. New project name: {project_name}")

    if args.copy_from:
        LOGGER.info(f"Onboarding project: {args.project}")
        paratext_project_dir: Path = create_paratext_project_folder_if_not_exists(project_name)
        copy_paratext_project_folder(Path(args.copy_from), paratext_project_dir, overwrite=args.overwrite)

    if args.clean_project:
        LOGGER.info(f"Cleaning Paratext project folder for {project_name}.")
        clean_projects(
            argparse.Namespace(
                input=get_paratext_project_dir(project_name),
                delete_subfolders=True,
                confirm_delete=True,
                dry_run=False,
            )
        )

    if args.extract_corpora:
        extract_config = config.get("extract_corpora", {})
        extract_corpora(
            projects={project_name},
            books_to_include=extract_config.get("include", []),
            books_to_exclude=extract_config.get("exclude", []),
            include_markers=extract_config.get("markers", False),
            extract_lemmas=extract_config.get("lemmas", False),
            extract_project_vrefs=extract_config.get("project-vrefs", False),
        )

    if args.collect_verse_counts:
        if not args.extract_corpora:
            LOGGER.warning(
                "--extract_corpora was not included. Collecting verse counts requires the corpus to be extracted first."
            )
        LOGGER.info(f"Collecting verse counts from {project_name}.")
        collect_verse_counts_wrapper(project_name, config.get("verse_counts", {}))

    if args.wildebeest:
        if not args.extract_corpora:
            LOGGER.warning("--extract_corpora was not included. Wildebeest requires the corpus to be extracted first.")

        extract_file = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*{project_name}.txt"))[0]
        LOGGER.info(f"Running Wildebeest analysis on {extract_file}.")
        with (
            open(f"{project_name}_wildebeest.json", "w", encoding="utf-8") as json_f,
            open(f"{project_name}_wildebeest.txt", "w", encoding="utf-8") as txt_f,
        ):
            wb_ana.process(in_file=extract_file, json_output=json_f, pp_output=txt_f)


if __name__ == "__main__":
    main()
