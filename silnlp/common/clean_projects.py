#!/usr/bin/env python3

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from machine.corpora import FileParatextProjectSettingsParser, ParatextProjectSettings
from tqdm import tqdm

# --- Global Constants ---
PROJECTS_FOLDER_DEFAULT = "M:/Paratext/projects" 
logger = logging.getLogger(__name__)
SETTINGS_FILENAME = "Settings.xml"

# --- Configuration for Deletion/Keep Rules ---

FILES_TO_DELETE_BY_NAME_CI = {
    "allclustercorrections.txt",
    "keys.asc",
}

FILES_TO_DELETE_BY_PATTERN = [
    "signature.*",
    "readme",
]

FILENAME_SUBSTRINGS_TO_DELETE_CI = ["error", "hyphenatedwords", "note"]

EXTENSIONS_TO_DELETE_CI = {
    ".bak",
    ".css",
    ".csv",
    ".dbl",
    ".doc",
    ".docx",
    ".font",
    ".git",
    ".hg",
    ".hgignore",
    ".html",
    ".id",
    ".ini",
    ".json",
    ".kb2",
    ".lds",
    ".map",
    ".md",
    ".old",
    ".p8z",
    ".rar",
    ".ssf",
    ".tag",
    ".tec",
    ".tsv",
    ".ttf",
    ".wdl",
    ".xls",
    ".xlsx",
    ".xml",
    ".yaml",
    ".zip",
}

FILES_TO_KEEP_BY_NAME_CI = {
    "settings.xml",
    "autocorrect.txt",
    "copr.htm",
    "custom.sty",
    "custom.vrs",
    "frtbak.sty",
    "wordanalyses.xml",
    "bookNames.xml",
    "canons.xml",
    "lexicon.xml",
    "TermRenderings.xml",

}

EXTENSIONS_TO_KEEP_CI = {
    ".cct",
    ".dic",
    ".ldml",
    ".lds",
}

# All subfolders should be deleted
SUBFOLDERS_TO_PRESERVE_BY_NAME_CI = {}

# --- Helper Functions ---


def has_settings_file(project_folder: Path) -> bool:
    return (project_folder / SETTINGS_FILENAME).is_file() or (
        project_folder / SETTINGS_FILENAME.lower()
    ).is_file()


class ProjectCleaner:
    def __init__(self, project_path: Path, args):
        self.project_path = project_path
        self.args = args
        self.scripture_file_extension = ".SFM"
        self.project_settings: Optional[ParatextProjectSettings] = None
        self.biblical_terms_files = set()
        self.files_to_keep = set()
        self.files_to_delete = set()
        self.folders_to_delete = set()
        self.parsing_errors = []
        self.log_prefix = f"[{self.project_path.name}] "

    def _log_info(self, message: str):
        full_message = f"{self.log_prefix}{message}"
        logger.info(full_message)
        if self.args.verbose > 0:
            print(full_message)

    def _log_action(self, action: str, item_path: Path):
        full_message = (
            f"{self.log_prefix}{action}: {item_path.relative_to(self.project_path)}"
        )
        logger.info(full_message)
        if self.args.verbose > 0:
            print(full_message)

    def _parse_settings(self):
        settings_file_path = self.project_path / SETTINGS_FILENAME
        if not settings_file_path.exists():
            settings_file_path = self.project_path / SETTINGS_FILENAME.lower()
            if not settings_file_path.exists():
                warning_msg = f"Warning: {SETTINGS_FILENAME} not found."
                if self.args.verbose:
                    self._log_info(warning_msg)
                self.parsing_errors.append(f"{SETTINGS_FILENAME} not found.")
                return

        try:
            parser = FileParatextProjectSettingsParser(str(self.project_path))
            project_settings = parser.parse()
            self.project_settings = project_settings

            full_suffix = project_settings.file_name_suffix.upper()
            self.scripture_file_extension = Path(full_suffix).suffix
            if not self.scripture_file_extension:
                self.scripture_file_extension = ""
            self._log_info(
                f"Determined scripture file extension: {self.scripture_file_extension}"
            )

            if project_settings.biblical_terms_file_name:
                terms_file_path = (
                    self.project_path / project_settings.biblical_terms_file_name
                )
                if terms_file_path.is_file():
                    self.biblical_terms_files.add(terms_file_path)
                    self._log_info(
                        f"Found BiblicalTermsListSetting file: {terms_file_path.name}"
                    )
                else:
                    warning_msg = f"Warning: BiblicalTermsListSetting file not found at expected path: {terms_file_path}"
                    if self.args.verbose:
                        self._log_info(warning_msg)
                    self.parsing_errors.append(
                        f"BiblicalTermsListSetting file not found: {terms_file_path.name}"
                    )
        except Exception as e:
            error_msg = f"Error parsing {SETTINGS_FILENAME}: {e}"
            if self.args.verbose:
                self._log_info(error_msg)
            self.parsing_errors.append(error_msg)

    def analyze_project_contents(self):
        self._parse_settings()

        all_items_in_project = list(self.project_path.glob("*"))

        # --- Pass 1: Identify files to KEEP ---
        settings_path = self.project_path / SETTINGS_FILENAME
        if settings_path.is_file():
            self.files_to_keep.add(settings_path)
        else:
            settings_path_lower = self.project_path / SETTINGS_FILENAME.lower()
            if settings_path_lower.is_file():
                self.files_to_keep.add(settings_path_lower)

        for item in self.project_path.iterdir():
            if item.is_file() and item.name.lower() in FILES_TO_KEEP_BY_NAME_CI:
                self.files_to_keep.add(item)

        for terms_file in self.biblical_terms_files:
            if terms_file.is_file():
                self.files_to_keep.add(terms_file)

        # Scripture files are identified using ParatextProjectSettings.get_book_id()
        if self.project_settings:
            for (
                item
            ) in (
                self.project_path.iterdir()
            ):  # Scripture files are typically at the project root
                if item.is_file():
                    book_id = self.project_settings.get_book_id(item.name)
                    if book_id is not None:
                        self.files_to_keep.add(item)
                        if self.args.verbose > 1:
                            self._log_info(
                                f"Kept scripture file (via get_book_id): {item.name}"
                            )
        elif self.args.verbose > 0:
            self._log_info(
                "Project settings not available; cannot use get_book_id for scripture identification."
            )

        for item in all_items_in_project:
            if item.is_file() and item.suffix.lower() in EXTENSIONS_TO_KEEP_CI:
                self.files_to_keep.add(item)

        if self.args.verbose > 1:
            self._log_info(
                f"Identified {len(self.files_to_keep)} files to keep initially."
            )

        # --- Pass 2: Identify files to DELETE ---
        for item_path in all_items_in_project:
            if not item_path.is_file() or item_path in self.files_to_keep:
                continue
            item_name_lower = item_path.name.lower()
            item_stem_lower = item_path.stem.lower()
            item_suffix_lower = item_path.suffix.lower()
            delete_file = False
            reason = ""

            if item_name_lower in FILES_TO_DELETE_BY_NAME_CI:
                delete_file = True
                reason = "specific name"
            elif any(
                item_path.match(pattern) for pattern in FILES_TO_DELETE_BY_PATTERN
            ):
                delete_file = True
                reason = "pattern match"
            elif any(
                sub_str in item_name_lower
                for sub_str in FILENAME_SUBSTRINGS_TO_DELETE_CI
            ):
                delete_file = True
                reason = "substring match"  
            elif item_suffix_lower in EXTENSIONS_TO_DELETE_CI:
                delete_file = True
                reason = f"extension ({item_suffix_lower})"
            elif item_name_lower.startswith(".") or item_name_lower.startswith("_"):
                # Files starting with . or _ are generally deleted, no special exceptions for .git or .hg files here.
                delete_file = True
                reason = "starts with . or _"
            elif not item_suffix_lower and item_path.is_file():
                delete_file = True
                reason = "no extension"
            elif "." in item_stem_lower:
                delete_file = True
                reason = "dot in filename stem"

            # If a .txt file is not in files_to_keep by this point, it's an extra .txt file.
            if item_suffix_lower == ".txt" and item_path not in self.files_to_keep:
                delete_file = True
                reason = "unidentified .txt file"

            if delete_file:
                self.files_to_delete.add(item_path)
                if self.args.verbose > 1:
                    self._log_info(
                        f"Marked for deletion ({reason}): {item_path.relative_to(self.project_path)}"
                    )

        # --- Pass 3: Identify folders to DELETE ---
        for item in self.project_path.iterdir():
            if item.is_dir():
                if item.name.lower() not in SUBFOLDERS_TO_PRESERVE_BY_NAME_CI:
                    self.folders_to_delete.add(item)
                elif self.args.verbose > 1:
                    self._log_info(f"Preserving subfolder: {item.name}")

        if self.args.verbose > 0:
            self._log_info(
                f"Identified {len(self.files_to_delete)} files and {len(self.folders_to_delete)} folders for deletion."
            )

    def execute_cleanup(self):
        if not self.files_to_delete and not self.folders_to_delete:
            if self.args.verbose > 0:
                self._log_info("No items marked for deletion.")
            return

        if self.args.dry_run:
            self._log_info("DRY RUN: No actual changes will be made.")

        for file_path in sorted(list(self.files_to_delete)):
            self._log_action(
                "DELETE_FILE (dry_run)" if self.args.dry_run else "DELETE_FILE",
                file_path,
            )
            if not self.args.dry_run and file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    self._log_info(f"Error deleting file {file_path}: {e}")

        for folder_path in sorted(list(self.folders_to_delete)):
            self._log_action(
                "DELETE_FOLDER (dry_run)" if self.args.dry_run else "DELETE_FOLDER",
                folder_path,
            )
            if not self.args.dry_run and folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    self._log_info(f"Error deleting folder {folder_path}: {e}")

        if not self.args.dry_run and self.args.verbose > 0:
            self._log_info("Cleanup execution finished.")


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Clean Paratext project folders by removing unnecessary files and folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "projects_root",
        nargs="?",
        default=PROJECTS_FOLDER_DEFAULT,
        help="The root directory containing Paratext project folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate cleaning process without actually deleting files or folders.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity. -v for project-level info, -vv for file-level decisions.",
    )
    parser.add_argument(
        "--log-file", help="Path to a file to log actions and verbose information."
    )
    args = parser.parse_args()

    # --- Configure Logging ---
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    if args.verbose == 0:
        console_handler.setLevel(logging.CRITICAL + 1)
    elif args.verbose == 1:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode="w")
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    if args.verbose > 0:
        print(f"Starting cleanup process for projects in: {args.projects_root}")
        if args.dry_run:
            print("DRY RUN mode enabled.")
    logger.info(
        f"Starting cleanup process. Projects root: {args.projects_root}. Dry run: {args.dry_run}. Verbose: {args.verbose}."
    )

    projects_root_path = Path(args.projects_root)
    if not projects_root_path.is_dir():
        print(f"Error: Projects root folder not found: {args.projects_root}")
        sys.exit(1)

    all_folders = [folder for folder in projects_root_path.iterdir() if folder.is_dir()]
    found_total_msg = f"Found {len(all_folders)} folders in {args.projects_root}."
    logger.info(found_total_msg)
    if args.verbose > 0:
        print(found_total_msg)

    project_folders = []
    non_project_folders = []
    for folder in tqdm(
        all_folders, desc="Scanning folders", unit="folder", disable=args.verbose > 0
    ):
        if has_settings_file(folder):
            project_folders.append(folder)
        else:
            non_project_folders.append(folder)

    found_msg = f"Found {len(project_folders)} project folders."
    logger.info(found_msg)
    if args.verbose > 0:
        print(found_msg)

    if non_project_folders:
        non_project_msg = (
            f"Found {len(non_project_folders)} non-project folders (will be ignored):"
        )
        logger.info(non_project_msg)
        if args.verbose > 0:
            print(non_project_msg)
        if args.verbose > 1:
            for folder in non_project_folders:
                logger.info(f"  - Ignored non-project folder: {folder.name}")
                if args.verbose > 0:
                    print(f"  - {folder.name}")

    if not project_folders:
        no_projects_msg = "No project folders found to clean."
        logger.info(no_projects_msg)
        if args.verbose > 0:
            print(no_projects_msg)
        return

    for project_path in tqdm(project_folders, desc="Cleaning projects", unit="project"):
        cleaner = ProjectCleaner(project_path, args)
        cleaner.analyze_project_contents()
        cleaner.execute_cleanup()
        if args.verbose > 0:
            print(f"--- Finished processing project: {project_path.name} ---")
        elif args.verbose == 0:
            logger.info(f"Finished processing project: {project_path.name}")

    final_msg = "\nCleanup process completed."
    logger.info(final_msg)
    if args.verbose > 0:
        print(final_msg)


if __name__ == "__main__":
    main()
