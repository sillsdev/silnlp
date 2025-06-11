import argparse
import logging
from pathlib import Path

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
    )
    parser.add_argument(
        "--copy-from",
        help="Path to a downloaded Paratext project folder. The local project will be copied to the bucket.",
        default=None,
    )
    parser.add_argument(
        "--overwrite", help="Overwrite any existing files and folders", default=False, action="store_true"
    )
    parser.add_argument(
        "--extract-corpora",
        help="Extract text corpora.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--include",
        metavar="books",
        nargs="+",
        default=[],
        help="The books to include; e.g., 'NT', 'OT', 'GEN'. Only used with extract-corpora.",
    )
    parser.add_argument(
        "--exclude",
        metavar="books",
        nargs="+",
        default=[],
        help="The books to exclude; e.g., 'NT', 'OT', 'GEN'. Only used with extract-corpora.",
    )
    parser.add_argument(
        "--markers", default=False, action="store_true", help="Include USFM markers. Only used with extract-corpora."
    )
    parser.add_argument(
        "--lemmas",
        default=False,
        action="store_true",
        help="Extract lemmas if available. Only used with extract-corpora.",
    )
    parser.add_argument(
        "--project-vrefs",
        default=False,
        action="store_true",
        help="Extract project verse refs. Only used with extract-corpora.",
    )

    args = parser.parse_args()
    project_name = args.project

    LOGGER.info(f"Onboarding project: {args.project}")
    paratext_project_dir: Path = create_paratext_project_folder_if_not_exists(project_name)

    if args.copy_from:
        copy_paratext_project_folder(Path(args.copy_from), paratext_project_dir, overwrite=args.overwrite)

    if args.extract_corpora:
        from .extract_corpora import extract_corpora

        extract_corpora(
            projects={project_name},
            include=args.include,
            exclude=args.exclude,
            markers=args.markers,
            lemmas=args.lemmas,
            project_vrefs=args.project_vrefs,
        )


if __name__ == "__main__":
    main()
