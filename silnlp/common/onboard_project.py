import argparse
import logging
from pathlib import Path

import yaml

from .environment import SIL_NLP_ENV

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performs several steps to onboard a new project before training a model.",
    )

    parser.add_argument(
        "project",
        help="Paratext project name. The project will be stored on the bucket at Paratext/projects/<project>.",
        required=True,
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

    args = parser.parse_args()
    project_name = args.project

    if not args.config:
        raise ValueError("Config file is required. Please provide a valid config.yml file using --config.")

    if not args.copy_from:
        raise ValueError(
            "Copy path is required. Please provide a valid local Paratext project folder using --copy-from."
        )

    LOGGER.info(f"Onboarding project: {args.project}")
    paratext_project_dir: Path = create_paratext_project_folder_if_not_exists(project_name)

    if args.copy_from:
        copy_paratext_project_folder(Path(args.copy_from), paratext_project_dir, overwrite=args.overwrite)

    config_file = Path(args.config)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "extract_corpora" in config:
        from .extract_corpora import extract_corpora

        LOGGER.info(f"Extracting {project_name}.")
        extract_corpora(
            projects={project_name},
            books_to_include=config["extract_corpora"]["include"] if "include" in config["extract_corpora"] else [],
            books_to_exclude=config["extract_corpora"]["exclude"] if "exclude" in config["extract_corpora"] else [],
            include_markers=(config["extract_corpora"]["markers"] if "markers" in config["extract_corpora"] else False),
            extract_lemmas=config["extract_corpora"]["lemmas"] if "lemmas" in config["extract_corpora"] else False,
            extract_project_vrefs=(
                config["extract_corpora"]["project-vrefs"] if "project-vrefs" in config["extract_corpora"] else False
            ),
        )

    if "verse_counts" in config:
        from .collect_verse_counts import collect_verse_counts

        LOGGER.info(f"Collecting verse counts from {project_name}.")

        if config["verse_counts"]["output_folder"]:
            output_folder = Path(config["verse_counts"]["output_folder"])
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)
        else:
            output_folder = SIL_NLP_ENV.mt_experiments_dir / "verse_counts" / project_name
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)
        collect_verse_counts(
            input_folder=project_name,
            output_folder=output_folder,
            file_patterns=(config["verse_counts"]["files"] if "files" in config["verse_counts"] else project_name),
            deutero=config["verse_counts"]["deutero"] if "deutero" in config["verse_counts"] else False,
            recount=config["verse_counts"]["recount"] if "recount" in config["verse_counts"] else False,
        )


if __name__ == "__main__":
    main()
