#!/usr/bin/env python3

import argparse
import concurrent.futures
import fnmatch
import logging
import shutil
import sys
import timeit
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
from machine.corpora import FileParatextProjectSettingsParser, ParatextProjectSettings # Assumes this library is installed
from tqdm import tqdm

# --- Global Constants ---
logger = logging.getLogger(__name__)
SETTINGS_FILENAME = "Settings.xml"

# --- Configuration for Deletion/Keep Rules ---
# These are matched with lower cased versions of the filename, they must be listed in lower case here.

FILES_TO_DELETE_BY_NAME = {
    "allclustercorrections.txt",
    "keys.asc",
}

FILES_TO_DELETE_BY_PATTERN = [
    "signature.*",
    "readme",
]

FILENAME_SUBSTRINGS_TO_DELETE = ["error", "hyphenatedwords", "note"]

EXTENSIONS_TO_DELETE = {
    ".bak",
    ".css",
    ".csv",
    ".cct",
    ".dbl",
    ".dic",
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

FILES_TO_KEEP_BY_NAME = {
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
    "termrenderings.xml",
}

EXTENSIONS_TO_KEEP = {
    ".ldml",
    ".lds",
}

extension_overlap = EXTENSIONS_TO_KEEP & EXTENSIONS_TO_DELETE
if extension_overlap:
    raise ValueError(
        "EXTENSIONS_TO_KEEP and EXTENSIONS_TO_DELETE must not overlap. Please check the code \
            for these extensions: {extension_overlap}"
    )

# All subfolders should be deleted
SUBFOLDERS_TO_PRESERVE_BY_NAME = {}

# --- Helper Functions ---

def has_settings_file(project_folder: Path) -> bool:
    return (project_folder / SETTINGS_FILENAME).is_file() or (project_folder / SETTINGS_FILENAME.lower()).is_file()

def create_dummy_projects(root_path: Path, num_projects: int, num_files_per_project: int):
    """
    Creates a temporary directory with dummy projects for testing.
    This simulates an I/O-heavy workload.
    """
    print(f"Creating {num_projects} dummy projects at {root_path}...")
    root_path.mkdir(exist_ok=True)
    
    for i in tqdm(range(num_projects), desc="Generating projects", unit="project"):
        project_path = root_path / f"project_{i}"
        project_path.mkdir()
        
        # Create a Settings.xml file to make it a valid project
        (project_path / SETTINGS_FILENAME).touch()
        
        # Create various dummy files to be cleaned up
        for j in range(num_files_per_project):
            # Create a mix of files to delete and keep
            if j % 5 == 0:
                # File to keep
                (project_path / f"keep_{j}.ldml").touch()
            elif j % 7 == 0:
                # File to delete by name
                (project_path / "keys.asc").touch()
            elif j % 11 == 0:
                # File to delete by pattern
                (project_path / f"signature.{j}.txt").touch()
            else:
                # Normal file to be deleted by extension or substring
                (project_path / f"delete_{j}.csv").touch()
        
        # Create a dummy folder to be deleted
        (project_path / "temp_folder").mkdir()
        (project_path / "temp_folder" / "temp_file.tmp").touch()

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
        self.log_buffer: list[str] = []  # Buffer to store log messages for this project
        self.log_prefix = f"[{self.project_path.name}] "

    def _log_info(self, message: str):
        full_message = f"{self.log_prefix}{message}"
        self.log_buffer.append(full_message)

    def _log_action(self, action: str, item_path: Path):
        full_message = f"{self.log_prefix}{action}: {item_path.relative_to(self.project_path)}"
        self.log_buffer.append(full_message)

    def _parse_settings(self):
        settings_file_path = self.project_path / SETTINGS_FILENAME
        if not settings_file_path.exists():
            settings_file_path = self.project_path / SETTINGS_FILENAME.lower()
            if not settings_file_path.exists():
                warning_msg = f"Warning: {SETTINGS_FILENAME} not found."
                if self.args.verbose > 0:
                    self._log_info(warning_msg)
                self.parsing_errors.append(f"{SETTINGS_FILENAME} not found.")
                return

        try:
            # The library 'machine.corpora' is assumed to be available
            # If not, this part will fail. For this benchmark, a mock or a simplified
            # version could be used if the library is not present.
            parser = FileParatextProjectSettingsParser(str(self.project_path))
            project_settings = parser.parse()
            self.project_settings = project_settings

        except Exception as e:
            error_msg = f"Error parsing {SETTINGS_FILENAME}: {e}"
            if self.args.verbose > 0:
                self._log_info(error_msg)
            self.parsing_errors.append(error_msg)
            
        if self.project_settings and self.project_settings.biblical_terms_file_name:
            terms_file_path = self.project_path / self.project_settings.biblical_terms_file_name
            if terms_file_path.is_file():
                self.biblical_terms_files.add(terms_file_path)

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
            if item.is_file() and item.name.lower() in FILES_TO_KEEP_BY_NAME:
                self.files_to_keep.add(item)

        for terms_file in self.biblical_terms_files:
            if terms_file.is_file():
                self.files_to_keep.add(terms_file)

        if self.project_settings:
            for item in self.project_path.iterdir():
                if item.is_file():
                    book_id = self.project_settings.get_book_id(item.name)
                    if book_id is not None:
                        self.files_to_keep.add(item)
        
        for item in all_items_in_project:
            if item.is_file() and item.suffix.lower() in EXTENSIONS_TO_KEEP:
                self.files_to_keep.add(item)

        # --- Pass 2: Identify files to DELETE ---
        for item_path in all_items_in_project:
            if not item_path.is_file() or item_path in self.files_to_keep:
                continue
            item_name_lower = item_path.name.lower()
            item_suffix_lower = item_path.suffix.lower()
            delete_file = False
            reason = ""

            if item_name_lower in FILES_TO_DELETE_BY_NAME:
                delete_file = True
            elif any(
                fnmatch.fnmatch(item_path.name.lower(), pattern.lower()) for pattern in FILES_TO_DELETE_BY_PATTERN
            ):
                delete_file = True
            elif any(sub_str in item_name_lower for sub_str in FILENAME_SUBSTRINGS_TO_DELETE):
                delete_file = True
            elif item_suffix_lower in EXTENSIONS_TO_DELETE:
                delete_file = True
            elif item_name_lower.startswith(".") or item_name_lower.startswith("_"):
                delete_file = True
            elif not item_suffix_lower and item_path.is_file():
                delete_file = True
            
            if item_suffix_lower == ".txt" and item_path not in self.files_to_keep:
                delete_file = True

            if delete_file:
                self.files_to_delete.add(item_path)

        # --- Pass 3: Identify folders to DELETE ---
        for item in self.project_path.iterdir():
            if item.is_dir():
                if item.name.lower() not in SUBFOLDERS_TO_PRESERVE_BY_NAME:
                    self.folders_to_delete.add(item)

    def execute_cleanup(self):
        if self.args.dry_run:
            return

        for file_path in self.files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    self._log_info(f"Error deleting file {file_path}: {e}")

        for folder_path in self.folders_to_delete:
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    self._log_info(f"Error deleting folder {folder_path}: {e}")

# --- Helper for concurrent project cleaning ---
def process_single_project_for_cleaning(
    project_path: Path, current_args: argparse.Namespace
) -> tuple[str, list[str], list[str]]:
    """
    Creates a ProjectCleaner instance, analyzes, and cleans a single project.
    Returns the project name, a list of log messages, and a list of parsing errors.
    """
    cleaner = ProjectCleaner(project_path, current_args)
    cleaner.analyze_project_contents()
    cleaner.execute_cleanup()
    return project_path.name, cleaner.log_buffer, cleaner.parsing_errors

# --- Benchmarking Functions ---

def single_threaded_cleanup(projects_root_path: Path, args: argparse.Namespace):
    """
    Cleans all projects in a given directory using a single thread.
    """
    all_folders = [item for item in projects_root_path.iterdir() if item.is_dir()]
    
    project_folders = []
    for folder in all_folders:
        if has_settings_file(folder):
            project_folders.append(folder)
            
    processed_project_data: List[Tuple[str, List[str], List[str], Path]] = []
    
    for project_path in tqdm(
        project_folders,
        desc="Single-threaded cleanup",
        unit="project",
        disable=args.verbose > 0,
    ):
        project_name, project_logs, project_errors = process_single_project_for_cleaning(project_path, args)
        processed_project_data.append(
            (project_name, project_logs, project_errors, project_path)
        )

def multithreaded_cleanup(projects_root_path: Path, args: argparse.Namespace):
    """
    Cleans all projects in a given directory using multiple threads.
    This is the original logic from the provided code.
    """
    all_folders = [item for item in projects_root_path.iterdir() if item.is_dir()]
    
    max_workers = 10
    
    project_folders = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {executor.submit(has_settings_file, folder): folder for folder in all_folders}
        for future in tqdm(
            concurrent.futures.as_completed(future_to_folder),
            total=len(all_folders),
            desc="Identifying project folders (threaded)",
            unit="folder",
            disable=args.verbose > 0,
        ):
            folder = future_to_folder[future]
            try:
                is_project = future.result()
                if is_project:
                    project_folders.append(folder)
            except Exception as exc:
                logger.error(f"Error checking folder {folder}: {exc}")
    
    processed_project_data: List[Tuple[str, List[str], List[str], Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_project_path_map = {
            executor.submit(process_single_project_for_cleaning, project_path, args): project_path
            for project_path in project_folders
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_project_path_map),
            total=len(project_folders),
            desc="Cleaning projects (threaded)",
            unit="project",
            disable=args.verbose > 0,
            mininterval=0.01,
        ):
            processed_project_path = future_to_project_path_map[future]
            try:
                project_name, project_logs, project_errors = future.result()
                processed_project_data.append(
                    (project_name, project_logs, project_errors, processed_project_path)
                )
            except Exception as exc:
                crit_error_msg = f"Critical error during processing of project {processed_project_path.name}: {exc}"
                logger.error(crit_error_msg)
                processed_project_data.append(
                    (processed_project_path.name, [], [f"Critical error: {exc}"], processed_project_path)
                )

# --- Main Benchmark Function ---

def run_benchmark():
    parser = argparse.ArgumentParser(
        description="Benchmark the performance of multithreaded vs. single-threaded project cleanup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-projects",
        type=int,
        default=50,
        help="Number of dummy projects to create for the test.",
    )
    parser.add_argument(
        "--num-files-per-project",
        type=int,
        default=20,
        help="Number of dummy files to create in each project.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate cleaning process without actually deleting files or folders. Does not affect the benchmark.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity. -v for project-level info, -vv for file-level decisions.",
    )
    args = parser.parse_args()

    # Create a fresh environment for the multithreaded test
    print("\n--- Starting Multithreaded Benchmark ---")
    with tempfile.TemporaryDirectory() as temp_dir_multi:
        projects_root_path_multi = Path(temp_dir_multi)
        print(f"Using temporary directory: {projects_root_path_multi}")
        create_dummy_projects(projects_root_path_multi, args.num_projects, args.num_files_per_project)
        
        multithreaded_time = timeit.timeit(
            lambda: multithreaded_cleanup(projects_root_path_multi, args),
            number=1
        )
    
    # Create a fresh environment for the single-threaded test
    print("\n--- Starting Single-threaded Benchmark ---")
    with tempfile.TemporaryDirectory() as temp_dir_single:
        projects_root_path_single = Path(temp_dir_single)
        print(f"Using temporary directory: {projects_root_path_single}")
        create_dummy_projects(projects_root_path_single, args.num_projects, args.num_files_per_project)
        
        single_threaded_time = timeit.timeit(
            lambda: single_threaded_cleanup(projects_root_path_single, args),
            number=1
        )

    print("\n--- Benchmark Results ---")
    print(f"Multithreaded Time: {multithreaded_time:.4f} seconds")
    print(f"Single-threaded Time: {single_threaded_time:.4f} seconds")

    if single_threaded_time > 0:
        speed_up = single_threaded_time / multithreaded_time
        print(f"Speed-up Factor: {speed_up:.2f}x")
        percentage_increase = ((single_threaded_time - multithreaded_time) / multithreaded_time) * 100
        print(f"Multithreading was {percentage_increase:.2f}% faster.")
    else:
        print("Single-threaded time was 0, cannot calculate speed-up.")

if __name__ == "__main__":
    run_benchmark()
