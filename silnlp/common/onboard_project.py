import argparse
import getpass
import hashlib
import logging
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import common.analyze as analyze
import wildebeest.wb_analysis as wb_ana
import yaml
from machine.corpora import FileParatextProjectSettingsParser

from silnlp.common.clean_projects import process_single_project_for_cleaning
from silnlp.nmt.clearml_connection import TAGS_LIST
from silnlp.nmt.config import Config

from ..nmt.config_utils import create_config
from .collect_verse_counts import collect_verse_counts
from .environment import SIL_NLP_ENV
from .extract_corpora import extract_corpora
from .iso_info import ALT_ISO, NLLB_TAG_FROM_ISO

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


def copy_file(source_file: Path, target_file: Path, overwrite=False) -> None:
    if target_file.exists() and not overwrite:
        LOGGER.info(f"File '{target_file}' already exists. Skipping.")
    else:
        target_file.write_bytes(source_file.read_bytes())


def copy_directory(source_dir: Path, target_dir: Path, overwrite=False) -> None:
    if not target_dir.exists():
        target_dir.mkdir()
    for sub_item in source_dir.iterdir():
        target_item = target_dir / sub_item.name
        if sub_item.is_dir():
            copy_directory(sub_item, target_item, overwrite)
        else:
            copy_file(sub_item, target_item, overwrite)


def copy_paratext_project_folder(source_dir: Path, project_name: str, overwrite=False) -> None:
    pt_project_path = get_paratext_project_dir(project_name)

    if not any(source_dir.iterdir()):
        LOGGER.warning(f"Source directory '{source_dir}' is empty.")
        return

    for source_item in source_dir.iterdir():
        target_item = pt_project_path / source_item.name
        if source_item.is_dir():
            copy_directory(source_item, target_item, overwrite=overwrite)
        else:
            copy_file(source_item, target_item, overwrite=overwrite)


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


def get_extract_path(project_name: str) -> Path | None:
    extract_paths = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*-{project_name}.txt"))
    if not extract_paths:
        return None
    return extract_paths[0]


def extract_corpora_wrapper(project_name: str, extract_config: dict, overwrite=False) -> None:
    extract_path = get_extract_path(project_name)
    if extract_path is not None and not overwrite:
        LOGGER.info(f"Extracted corpus '{extract_path}' already exists. Skipping corpus extraction.")
        return
    LOGGER.info(f"Extracting corpora for project '{project_name}'")

    versification_error_output_path = (
        SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name / "versification_errors.txt"
    )
    if not versification_error_output_path.exists():
        versification_error_output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_corpora(
        projects={project_name},
        books_to_include=extract_config.get("include", []),
        books_to_exclude=extract_config.get("exclude", []),
        include_markers=extract_config.get("markers", False),
        extract_lemmas=extract_config.get("lemmas", False),
        extract_project_vrefs=extract_config.get("project-vrefs", False),
        extract_surface_forms=extract_config.get("surface-forms", False),
        parent_project=extract_config.get("parent_project", None),
        versification_error_output_path=SIL_NLP_ENV.mt_experiments_dir
        / "OnboardingRequests"
        / project_name
        / "versification_errors.txt",
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

    extract_path = get_extract_path(project_name)
    if extract_path is None:
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


def get_extract_iso_code(extract_file: str) -> str:
    iso_code = extract_file.split("-")[0]
    if len(iso_code) == 2:
        iso_code = ALT_ISO.get_alternative(iso_code) if ALT_ISO.get_alternative(iso_code) else iso_code
    return iso_code


def calculate_tokenization_stats(project_name: str, stats_config: dict, overwrite=False) -> None:
    stats_dir = SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name / "stats"

    if stats_dir.exists() and not overwrite:
        LOGGER.info(f"Stats directory '{stats_dir}' already exists. Skipping stats calculation.")
        return
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    extract_path = get_extract_path(project_name)
    if extract_path is None:
        LOGGER.error(
            f"No extracted corpus found for project '{project_name}'. Skipping tokenization stats calculation."
        )
        return
    extract_file = extract_path.stem

    iso_code = get_extract_iso_code(extract_file)
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


def align_wrapper(
    project_name: str,
    align_config: dict,
    iso_codes: set[str],
    clearml_queue: str,
    clearml_tag: str,
    overwrite=False,
) -> None:
    experiment_name = "OnboardingRequests" / project_name / "alignments"
    align_output_dir = SIL_NLP_ENV.mt_experiments_dir / experiment_name
    if align_output_dir.exists() and not overwrite:
        LOGGER.info(f"Alignments output directory '{align_output_dir}' already exists. Skipping alignments.")
        return
    if not align_output_dir.exists():
        align_output_dir.mkdir(parents=True, exist_ok=True)

    extract_path = get_extract_path(project_name)
    if extract_path is None:
        LOGGER.error(f"No extracted corpus found for project '{project_name}'. Skipping alignments.")
        return

    extract_file = extract_path.stem
    iso_codes.add(get_extract_iso_code(extract_file))

    with open("silnlp/assets/standard_alignments.yml", "r", encoding="utf-8") as f:
        standard_alignments = yaml.safe_load(f)
        alignments = set()
        for iso_code in iso_codes:
            iso_standard_alignments = standard_alignments.get(iso_code, None)
            if iso_standard_alignments is not None:
                iso_standard_alignments = [alignment.strip() for alignment in iso_standard_alignments]
                alignments.update(iso_standard_alignments)
    LOGGER.info(f"Running alignments on {extract_path}.")
    if align_config is None:
        if alignments is None or len(alignments) == 0:
            LOGGER.error(f"No projects found to align with '{project_name}'. Skipping alignments.")
            return
        align_config = {
            "data": {
                "aligner": "eflomal",
                "corpus_pairs": [
                    {
                        "mapping": "many_to_many",
                        "src": extract_file,
                        "trg": alignments,
                        "type": "train",
                    }
                ],
                "tokenize": False,
            }
        }
    else:
        alignment_projects: List[str] = align_config.get("data", {}).get("corpus_pairs", [])[0].get("trg", [])
        if iso_standard_alignments:
            for standard_alignment in iso_standard_alignments:
                if standard_alignment not in alignment_projects:
                    alignment_projects.append(standard_alignment)
        align_config["data"]["corpus_pairs"][0]["trg"] = alignment_projects

    with open(align_output_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(align_config, f, allow_unicode=True)
    align_config: Config = create_config(exp_dir=align_output_dir, config=align_config)
    collect_verse_counts_directory = (
        SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name / "verse_counts"
    )
    shutil.copytree(collect_verse_counts_directory, align_output_dir, dirs_exist_ok=True)
    old_argv = sys.argv
    try:
        sys.argv = [
            str(experiment_name),
            "--create-summaries",
            "--clearml-queue",
            clearml_queue,
            "--clearml-tag",
            clearml_tag,
        ]
        analyze.main()
    finally:
        sys.argv = old_argv


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
            LOGGER.warning(f"{project} is a non-Standard project. Type is '{settings.translation_type}'.")

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
            project_name = Path(project).stem
            if project_name.endswith("_Resource"):
                project_name = project_name.replace("_Resource", "")
            needs_password = any(zinfo.flag_bits & 0x1 for zinfo in zip_ref.infolist())
            if needs_password:
                if zip_password:
                    pwd = zip_password
                if not pwd:
                    pwd = getpass.getpass(prompt=f"Enter password for {project_name}: ")
                zip_ref.extractall(copy_from / project_name, pwd=pwd.encode())
            else:
                zip_ref.extractall(copy_from / project_name)
    if Path(project).stem.endswith("_Resource"):
        resource_hash_path = copy_from / project_name / ".resource_hash" if copy_from else None
        if resource_hash_path and not resource_hash_path.exists():
            resource_hash_path.touch()

    check_for_project_errors(copy_from, project_name)

    local_project_path = copy_from / project_name if copy_from else None

    if "-" in project_name:
        LOGGER.info(f"Project name '{project_name}' contains hyphens. Replacing hyphens with underscores.")
        project_name = project_name.replace("-", "_")
        LOGGER.info(f"New project name: '{project_name}'")
    if datestamp and not is_resource(project_name):
        now = datetime.now()
        datestamp = now.strftime("%Y_%m_%d")
        project_name = f"{project_name}_{datestamp}"
        LOGGER.info(f"Datestamping project. New project name: {project_name}")

    if local_project_path and local_project_path.exists() and local_project_path.name != project_name:
        new_local_project_path = local_project_path.parent / project_name
        if not new_local_project_path.exists():
            new_local_project_path.mkdir(parents=True, exist_ok=True)
        copy_directory(local_project_path, new_local_project_path, overwrite=True)
        local_project_path = new_local_project_path

    return project_name, local_project_path, copy_from


def is_resource(project_name: str, copy_from: Path | None) -> bool:
    resource_hash_path = copy_from / project_name / ".resource_hash" if copy_from else None
    return resource_hash_path.exists() if resource_hash_path else False


def generate_resource_hash(resource_name: str) -> str:
    resource_path = SIL_NLP_ENV.pt_projects_dir / resource_name
    resource_hash = hashlib.sha256()
    for root, dirs, files in os.walk(resource_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file == ".resource_hash":
                continue
            file_path = os.path.join(root, file)
            resource_hash.update(file_path.encode())
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    resource_hash.update(chunk)

    resource_hash = resource_hash.hexdigest()
    with open(resource_path / ".resource_hash", "w") as f:
        f.write(resource_hash)
    return resource_hash


def check_resource_hash(resource_name: str) -> bool:
    new_resource_hash = generate_resource_hash(resource_name)
    old_resource_path = SIL_NLP_ENV.pt_projects_dir / resource_name
    old_resource_hash_path = old_resource_path / ".resource_hash"
    old_resource_hash = None
    if old_resource_hash_path.exists():
        old_resource_hash = old_resource_hash_path.read_text().strip()
    return new_resource_hash == old_resource_hash


def update_resource(resource_name: str) -> None:
    old_resource_path = SIL_NLP_ENV.pt_projects_dir / resource_name

    now = datetime.now()
    datestamp = now.strftime("%Y_%m_%d")
    new_resource_name = f"{old_resource_path.name}_{datestamp}"
    new_resource_path = old_resource_path.parent / new_resource_name
    old_resource_path.rename(new_resource_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performs several steps to onboard a new project before training a model.",
    )

    parser.add_argument(
        "main-project",
        help="The Main Paratext project name for onboarding. The project will be stored on the bucket at Paratext/projects/<project>.",
        default=None,
    )
    parser.add_argument(
        "--projects",
        help="The Paratext project name(s) for onboarding. The project(s) will be stored on the bucket at Paratext/projects/<project>. Alignments will not be run for these projects.",
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
    parser.add_argument("--align", default=None, nargs="+", type=str, help="List of iso codes to align with.")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Only used with --align. Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    parser.add_argument(
        "--clearml-tag",
        metavar="tag",
        choices=TAGS_LIST,
        default=None,
        type=str,
        help=f"Only used with --align. Tag to add to the ClearML Task - {TAGS_LIST}",
    )

    args = parser.parse_args()

    if args.clearml_queue is not None:
        if "cpu" not in args.clearml_queue:
            LOGGER.warning("Running this script on a GPU queue will not speed it up. Please only use CPU queues.")
            exit()
        if args.clearml_tag is None:
            parser.error("Missing ClearML tag. Add a tag using --clearml-tag. Possible tags: " + f"{TAGS_LIST}")

    if not args.extract_corpora:
        if args.collect_verse_counts or args.wildebeest or args.stats or args.align:
            args.extract_corpora = True
    if not args.collect_verse_counts and (args.align or args.wildebeest):
        args.collect_verse_counts = True

    config = get_config(args.config) if args.config else {}

    projects: list = args.projects if args.projects else []
    projects.append(args.main_project)
    for project in projects:
        pwd = config.get("zip_password", None)
        project_name, local_project_path, copy_from = setup_local_project(project, args.copy_from, pwd, args.datestamp)

        onboarding_project_path = SIL_NLP_ENV.mt_experiments_dir / "OnboardingRequests" / project_name

        if onboarding_project_path.exists() and not args.overwrite:
            LOGGER.info(
                f"Onboarding project folder '{onboarding_project_path}' already exists. Skipping onboarding for project '{project_name}'."
            )
            continue

        onboarding_project_path.mkdir(parents=True, exist_ok=True)
        log_file = open(onboarding_project_path / f"{project_name}_onboarding.log", "w", encoding="utf-8")
        log_file.close()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file.name),
            ],
            force=True,
        )

        if not args.no_clean:
            LOGGER.info(f"Cleaning Paratext project: {project_name}.")
            process_single_project_for_cleaning(
                local_project_path,
            )

        if is_resource(project_name) and check_resource_hash(project_name):
            LOGGER.info(f"Resource '{project_name}' is up to date. Skipping onboarding.")
            continue
        elif is_resource(project_name):
            LOGGER.info(f"Resource '{project_name}' is outdated. Continuing onboarding to update resource.")
            update_resource(project_name)

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
            LOGGER.info(f"Collecting verse counts from {project_name}.")
            collect_verse_counts_wrapper(project_name, config.get("verse_counts", {}), args.overwrite)

        if args.wildebeest:
            wildebeest_config: dict = config.get("wildebeest", {})
            wildebeest_analysis_wrapper(project_name, wildebeest_config, args.overwrite)

        if args.stats:
            stats_config: dict = config.get("stats", None)
            calculate_tokenization_stats(project_name, stats_config, args.overwrite)

        if project == args.main_project and args.align:
            align_config: dict = config.get("align", None)
            align_wrapper(
                project_name, align_config, set(args.align), args.clearml_queue, args.clearml_tag, args.overwrite
            )


if __name__ == "__main__":
    main()
