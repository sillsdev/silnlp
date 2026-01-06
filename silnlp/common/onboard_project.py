import argparse
import getpass
import logging
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import wildebeest.wb_analysis as wb_ana
import yaml

import silnlp.common.clean_projects as clean_projects

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


def calculate_tokenization_stats(project_name: str, stats_config: dict = None) -> None:
    stats_dir = Path("stats") / project_name
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    extract_path = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*{project_name}*.txt"))[0]
    extract_file = extract_path.stem

    iso_code = extract_file.split("-")[0]

    iso_dict = {**{k: v for k, v in iso_data}, **{v: k for k, v in iso_data}}

    iso_code = iso_dict.get(iso_code, iso_code)
    nllb_tag = NLLB_TAG_FROM_ISO.get(iso_code, "eng_Latn")

    if stats_config is None:
        stats_config = {
            "use_default_model_dir": False,
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

def validate_zip(zip_path, expected_folder):
    "Check if zip is valid for extraction"
    with zipfile.ZipFile(zip_path) as zf:
        names = [Path(n) for n in zf.namelist()]
        if not names: return False
        roots = {n.parts[0] if len(n.parts) > 1 else n.name.rstrip('/') for n in names}
        has_root_files = any(len(n.parts) == 1 and not n.name.endswith('/') for n in names)
        if not has_root_files and len(roots) == 1:
            return roots.pop() == expected_folder
        return True
    
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performs several steps to onboard a new project before training a model.",
    )

    parser.add_argument(
        "projects",
        help="Paratext project name. The project will be stored on the bucket at Paratext/projects/<project>.",
        nargs="*",
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
        help="Add a datestamp to the project folder name before copying to the Paratext project folder.",
    )
    parser.add_argument(
        "--wildebeest", default=False, action="store_true", help="Run Wildebeest analysis on the extracted corpora."
    )
    parser.add_argument("--stats", default=False, action="store_true", help="Compute tokenization statistics")

    args = parser.parse_args()
    if not args.projects:
        raise ValueError("Project name is required. Please provide a valid Paratext project name using <project>.")

    config = get_config(args.config) if args.config else {}

    if args.copy_from:
        copy_from_path = Path(args.copy_from).expanduser()
        errors = []
        
        for project in args.projects:
            project_folder = copy_from_path / project
            project_zip = copy_from_path / f"{project}.zip"
            
            if not project_folder.is_dir() and not project_zip.exists():
                errors.append(f"Project not found: {project} (no folder or zip in {copy_from_path})")
            elif project_zip.exists() and not project_folder.is_dir():
                if not validate_zip(project_zip, project): errors.append(f"Invalid zip: {project_zip} does not extract to single folder named {project}")
            
            normalized_name = project.replace('-', '_')
            if args.datestamp: normalized_name = f"{normalized_name}_{datetime.now().strftime('%Y%m%d')}"
            destination_folder = SIL_NLP_ENV.pt_projects_dir / normalized_name
            if destination_folder.exists(): errors.append(f"Destination already exists: {destination_folder}")
        
        if not SIL_NLP_ENV.pt_projects_dir.exists(): errors.append(f"Destination directory does not exist: {SIL_NLP_ENV.pt_projects_dir}")
        elif not SIL_NLP_ENV.pt_projects_dir.is_dir(): errors.append(f"Destination is not a directory: {SIL_NLP_ENV.pt_projects_dir}")
        
        if errors:
            for error in errors: LOGGER.error(error)
            sys.exit("Validation failed. No processing performed.")


    if args.copy_from:
        for project in args.projects:
            project_folder = copy_from_path / project
            project_zip = copy_from_path / f"{project}.zip"
            
            if project_zip.exists() and not project_folder.is_dir():
                LOGGER.info(f"Unzipping {project_zip}")
                try:
                    with zipfile.ZipFile(project_zip) as zf: zf.extractall(copy_from_path)
                except RuntimeError:
                    password = getpass.getpass(f"Password for {project_zip}: ")
                    with zipfile.ZipFile(project_zip) as zf: zf.extractall(copy_from_path, pwd=password.encode())
                
                names = [Path(n) for n in zipfile.ZipFile(project_zip).namelist()]
                roots = {n.parts[0] if len(n.parts) > 1 else n.name.rstrip('/') for n in names}
                has_root_files = any(len(n.parts) == 1 and not n.name.endswith('/') for n in names)
                if has_root_files or len(roots) != 1 or roots.pop() != project:
                    extracted_folder = copy_from_path / project
                    extracted_folder.mkdir(exist_ok=True)
                    for item in copy_from_path.iterdir():
                        if item.is_file() and item != project_zip: item.rename(extracted_folder / item.name)


    project_folders = []
    for project in args.projects:
        project_folder = copy_from_path / project
        normalized_name = project.replace('-', '_')
        if normalized_name != project:
            normalized_folder = copy_from_path / normalized_name
            LOGGER.info(f"Renaming {project_folder} to {normalized_folder}")
            project_folder = project_folder.rename(normalized_folder)
        if args.datestamp:
            date_str = datetime.now().strftime('%Y%m%d')
            datestamped_folder = copy_from_path / f"{normalized_name}_{date_str}"
            LOGGER.info(f"Adding datestamp: {project_folder} to {datestamped_folder}")
            project_folder = project_folder.rename(datestamped_folder)
        project_folders.append(project_folder)
        

    if args.clean_project:
        LOGGER.info("Cleaning Paratext project folders.")
        old_argv = sys.argv
        try:
            sys.argv = ["clean_projects"] + [str(project_folder) for project_folder in project_folders]
            clean_projects.main()
        finally:
            sys.argv = old_argv

    for project_folder in project_folders:
        project_name = project_folder.name
        destination_folder = SIL_NLP_ENV.pt_projects_dir / project_name
        LOGGER.info(f"Copying {project_name} from {args.copy_from} to {destination_folder}")
        copy_paratext_project_folder(project_folder, SIL_NLP_ENV.pt_projects_dir, overwrite=args.overwrite)

        if args.extract_corpora:
            LOGGER.info(f"Extracting corpora from {destination_folder}")
            extract_config: dict = config.get("extract_corpora", {})
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
                LOGGER.warning(
                    "--extract_corpora was not included. Wildebeest requires the corpus to be extracted first."
                )

            extract_file = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*{project_name}.txt"))[0]
            extract_file = str(extract_file)
            LOGGER.info(f"Running Wildebeest analysis on {extract_file}.")
            wildebeest_config = config.get("wildebeest", {})
            old_argv = sys.argv
            try:
                sys.argv = [
                    "wb_ana",
                    "-i",
                    extract_file,
                    "-j",
                    f"{project_name}_wildebeest.json",
                    "-o",
                    f"{project_name}_wildebeest.txt",
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

        if args.stats:
            if not args.extract_corpora:
                LOGGER.warning("--extract_corpora was not included. Stats requires the corpus to be extracted first.")
            calculate_tokenization_stats(project_name, config.get("stats", None))


if __name__ == "__main__":
    main()
