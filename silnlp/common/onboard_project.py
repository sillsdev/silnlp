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
from typing import List

import wildebeest.wb_analysis as wb_ana
import yaml
from machine.corpora import FileParatextProjectSettingsParser

from silnlp.common.analyze import analyze
from silnlp.common.clean_projects import process_single_project_for_cleaning
from silnlp.nmt.clearml_connection import TAGS_LIST, SILClearML
from silnlp.nmt.config import Config

from ..nmt.config_utils import create_config
from .collect_verse_counts import collect_verse_counts
from .environment import SIL_NLP_ENV
from .extract_corpora import extract_corpora
from .iso_info import ALT_ISO, NLLB_TAG_FROM_ISO

LOGGER = logging.getLogger(__package__ + ".onboard_project")


class OnboardingProject:

    def __init__(self, project_name: str, overwrite: bool) -> None:
        self.project_name: str = project_name
        self.local_project_path: Path | None = None
        self.output_folder: Path | None = None
        self.extract_file: Path | None = None
        self.iso_code: str = ""
        self.resource: bool | None = None
        self.overwrite: bool = overwrite

    def get_extract_path(self) -> Path | None:
        if self.extract_file is not None:
            return self.extract_file
        extract_paths = list(SIL_NLP_ENV.mt_scripture_dir.glob(f"*-{self.project_name}.txt"))
        if not extract_paths:
            return None
        self.extract_file = extract_paths[0]
        return self.extract_file

    def extract_corpora_wrapper(self, extract_config: dict) -> None:
        extract_path = self.get_extract_path()
        if extract_path is not None and not self.overwrite:
            LOGGER.info(f"Extracted corpus '{extract_path}' already exists. Skipping corpus extraction.")
            return
        LOGGER.info(f"Extracting corpora for project '{self.project_name}'")

        versification_error_output_path = Path(self.output_folder / f"versification_errors_{self.project_name}.txt")
        extract_path = extract_corpora(
            projects={self.project_name},
            books_to_include=extract_config.get("include", []),
            books_to_exclude=extract_config.get("exclude", []),
            include_markers=extract_config.get("markers", False),
            extract_lemmas=extract_config.get("lemmas", False),
            extract_project_vrefs=extract_config.get("project-vrefs", False),
            extract_surface_forms=extract_config.get("surface-forms", False),
            parent_project=extract_config.get("parent_project", None),
            versification_error_output_path=versification_error_output_path,
        )
        self.extract_file = extract_path

    def wildebeest_analysis_wrapper(self, wildebeest_config: dict) -> None:
        extract_path = self.get_extract_path()
        if extract_path is None:
            LOGGER.error(f"No extracted corpus found for project '{self.project_name}'. Skipping Wildebeest analysis.")
            return
        LOGGER.info(f"Running Wildebeest analysis on {extract_path}.")
        old_argv = sys.argv
        try:
            sys.argv = [
                "wb_ana",
                "-i",
                str(extract_path),
                "-o",
                f"{self.output_folder}/wildebeest_{self.project_name}.txt",
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

    def get_extract_iso_code(self) -> str:
        if self.iso_code:
            return self.iso_code
        extract_path = self.get_extract_path()
        if extract_path is None:
            LOGGER.error(f"No extracted corpus found for project '{self.project_name}'. Cannot determine ISO code.")
            return ""

        extract_file = extract_path.stem
        iso_code = extract_file.split("-")[0]
        if len(iso_code) == 2:
            iso_code = ALT_ISO.get_alternative(iso_code) if ALT_ISO.get_alternative(iso_code) else iso_code
        self.iso_code = iso_code
        return iso_code

    def calculate_tokenization_stats(
        self, stats_config: dict, ref_project_extract_file_names: List[str], ref_isos: List[str]
    ) -> None:
        stats_dir = Path(self.output_folder / "stats")

        if stats_dir.exists() and not self.overwrite:
            LOGGER.info(f"Stats directory '{str(stats_dir)}' already exists. Skipping stats calculation.")
            return
        if not stats_dir.exists():
            stats_dir.mkdir(parents=True, exist_ok=True)

        extract_path = self.get_extract_path()
        if extract_path is None:
            LOGGER.error(
                f"No extracted corpus found for project '{self.project_name}'. Skipping tokenization stats calculation."
            )
            return
        extract_file = extract_path.stem

        iso_codes = [self.get_extract_iso_code()] + ref_isos

        lang_codes = {}

        for iso in iso_codes:
            nllb_tag = NLLB_TAG_FROM_ISO.get(iso, None)
            if nllb_tag:
                lang_codes[iso] = nllb_tag

        if stats_config is None:
            stats_config = {
                "data": {
                    "corpus_pairs": [
                        {
                            "mapping": "many_to_many",
                            "src": ref_project_extract_file_names if ref_project_extract_file_names else [extract_file],
                            "trg": extract_file,
                            "type": "train",
                            "lang_codes": lang_codes,
                        }
                    ],
                },
            }

        LOGGER.info(f"Calculating tokenization stats for project '{self.project_name}'")
        with open(stats_dir / "config.yml", "w", encoding="utf-8") as f:
            yaml.dump(stats_config, f, allow_unicode=True)
        config = create_config(exp_dir=stats_dir, config=stats_config)

        config.set_seed()
        config.preprocess(stats=True, force_align=True)
        # Copy tokenization_stats.csv and tokenization_stats.xlsx to output folder with project name in file name
        tokenization_stats_csv = stats_dir / "tokenization_stats.csv"
        tokenization_stats_xlsx = stats_dir / "tokenization_stats.xlsx"
        if tokenization_stats_csv.exists():
            shutil.move(str(tokenization_stats_csv), str(self.output_folder / "tokenization_stats.csv"))
        if tokenization_stats_xlsx.exists():
            shutil.move(
                str(tokenization_stats_xlsx),
                str(self.output_folder / "tokenization_stats.xlsx"),
            )

    def align_wrapper(
        self,
        align_config: dict,
        ref_project_extract_file_names: List[str],
        iso_codes: set,
    ) -> None:
        align_output_dir = Path(self.output_folder / "alignments")
        if align_output_dir.exists() and not self.overwrite:
            LOGGER.info(f"Alignments output directory '{str(align_output_dir)}' already exists. Skipping alignments.")
            return
        if not align_output_dir.exists():
            align_output_dir.mkdir(parents=True, exist_ok=True)

        extract_path = self.get_extract_path()
        if extract_path is None:
            LOGGER.error(f"No extracted corpus found for project '{self.project_name}'. Skipping alignments.")
            return

        extract_file = extract_path.stem

        with open("silnlp/assets/standard_alignments.yml", "r", encoding="utf-8") as f:
            standard_alignments = yaml.safe_load(f)
            alignments = set()
            alignments.update(ref_project_extract_file_names)
            iso_codes.add("default")
            for iso_code in iso_codes:
                iso_standard_alignments = standard_alignments.get(iso_code, None)
                if iso_standard_alignments is not None:
                    iso_standard_alignments = [alignment.strip() for alignment in iso_standard_alignments]
                    alignments.update(iso_standard_alignments)
        LOGGER.info(f"Running alignments on {extract_path}.")

        if align_config is None:
            if alignments is None or len(alignments) == 0:
                LOGGER.error(f"No projects found to align with '{self.project_name}'. Skipping alignments.")
                return
            align_config = {
                "data": {
                    "aligner": "eflomal",
                    "corpus_pairs": [
                        {
                            "mapping": "many_to_many",
                            "src": extract_file,
                            "trg": list(alignments),
                            "type": "train",
                        }
                    ],
                    "tokenize": False,
                }
            }
        else:
            alignment_projects = set(align_config.get("data", {}).get("corpus_pairs", [])[0].get("trg", []))
            alignment_projects.update(ref_project_extract_file_names)
            if iso_standard_alignments:
                for standard_alignment in iso_standard_alignments:
                    if standard_alignment not in alignment_projects:
                        alignment_projects.add(standard_alignment)
            align_config["data"]["corpus_pairs"][0]["trg"] = alignment_projects

        with open(align_output_dir / "config.yml", "w", encoding="utf-8") as f:
            yaml.dump(align_config, f, allow_unicode=True)
        align_config: Config = create_config(exp_dir=align_output_dir, config=align_config)
        exp_name = f"{self.output_folder.stem}/{self.project_name}/alignments"
        analyze(config=align_config, exp_name=exp_name, create_summaries=True)
        corpus_stats_csv = align_output_dir / "corpus_stats.csv"
        if corpus_stats_csv.exists():
            shutil.move(
                str(corpus_stats_csv),
                str(self.output_folder / "corpus_stats.csv"),
            )

    def check_for_project_errors(self) -> None:
        if self.local_project_path:
            if not self.local_project_path.exists():
                raise FileNotFoundError(f"The project folder '{self.local_project_path}' does not exist.")
            settings_file = self.local_project_path / "Settings.xml"
            if not settings_file.exists():
                raise FileNotFoundError(
                    f"The Settings.xml file was not found in the project folder '{self.local_project_path}'. Please ensure this is a valid Paratext project folder."
                )

            settings = FileParatextProjectSettingsParser(self.local_project_path).parse()

            if settings.translation_type != "Standard":
                LOGGER.warning(f"{self.project_name} is a non-Standard project. Type is '{settings.translation_type}'.")

            book_part = re.sub(r"\d", "[0-9]", settings.file_name_form)
            book_part = re.sub(r"([A-Z])", r"[A-Z]", book_part)
            pattern = f"{settings.file_name_prefix}.*{book_part}.*{settings.file_name_suffix}"
            if not any([re.match(pattern, file.name) for file in self.local_project_path.iterdir()]):
                raise ValueError(
                    f"{self.local_project_path} does not contain any files using the naming convention, '{pattern}', found in the Settings.xml file."
                )

    def setup_local_project(self, project: str, copy_from: Path, datestamp: bool) -> None:
        if project.endswith(".zip") or project.endswith(".p8z"):
            with zipfile.ZipFile(copy_from / Path(project), "r") as zip_ref:
                self.project_name = Path(project).stem
                needs_password = any(zinfo.flag_bits & 0x1 for zinfo in zip_ref.infolist())
                if needs_password:
                    zip_password = os.getenv("PT_ZIP_PASSWORD", None)
                    if zip_password:
                        pwd = zip_password
                    if not pwd:
                        pwd = getpass.getpass(prompt=f"Enter password for {self.project_name}: ")
                    zip_ref.extractall(copy_from / Path(self.project_name), pwd=pwd.encode())
                else:
                    zip_ref.extractall(copy_from / Path(self.project_name))

        self.local_project_path = Path(copy_from) / Path(self.project_name) if copy_from else None

        self.check_for_project_errors()

        if self.project_name.endswith("_Resource"):
            resource_hash_path = copy_from / Path(f"{self.project_name}/.resource_hash") if copy_from else None
            if resource_hash_path and not resource_hash_path.exists():
                resource_hash_path.touch()
            self.project_name = self.project_name.replace("_Resource", "")

        if "-" in self.project_name:
            LOGGER.info(f"Project name '{self.project_name}' contains hyphens. Replacing hyphens with underscores.")
            self.project_name = self.project_name.replace("-", "_")
            LOGGER.info(f"New project name: '{self.project_name}'")
        if datestamp and not self.is_resource():
            self.project_name = append_datestamp(self.project_name)
            LOGGER.info(f"Datestamping project. New project name: {self.project_name}")

        if (
            self.local_project_path
            and self.local_project_path.exists()
            and self.local_project_path.name != self.project_name
        ):
            new_local_project_path = self.local_project_path.parent / self.project_name
            if not new_local_project_path.exists():
                new_local_project_path.mkdir(parents=True, exist_ok=True)
            copy_directory(self.local_project_path, new_local_project_path, overwrite=True)
            self.local_project_path = new_local_project_path

    def is_resource(self) -> bool:
        if self.resource is not None:
            return self.resource
        resource_hash_path = Path(self.local_project_path / ".resource_hash")
        self.resource = resource_hash_path.exists()
        return self.resource

    def generate_resource_hash(self) -> str:
        resource_path = SIL_NLP_ENV.pt_projects_dir / self.project_name
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

    def check_resource_hash(self) -> bool:
        new_resource_hash = self.generate_resource_hash()
        old_resource_path = SIL_NLP_ENV.pt_projects_dir / self.project_name
        old_resource_hash_path = old_resource_path / ".resource_hash"
        old_resource_hash = None
        if old_resource_hash_path.exists():
            old_resource_hash = old_resource_hash_path.read_text().strip()
        return new_resource_hash == old_resource_hash

    def update_resource(self) -> None:
        old_resource_path = SIL_NLP_ENV.pt_projects_dir / self.project_name

        new_resource_name = append_datestamp(old_resource_path.name)
        new_resource_path = old_resource_path.parent / new_resource_name
        old_resource_path.rename(new_resource_path)

    def set_output_folder(self, output_folder: Path) -> None:
        self.output_folder = output_folder


class OnboardingRequest:

    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        onboarding_config = config.get("onboarding", {})
        self.overwrite = onboarding_config.get("overwrite", False)
        main_project_name = onboarding_config.get("main_project", None)
        self.main_project: OnboardingProject = OnboardingProject(
            project_name=main_project_name, overwrite=self.overwrite
        )
        reference_project_names = onboarding_config.get("ref_projects", [])
        self.reference_projects: List[OnboardingProject] = []
        for ref_project_name in reference_project_names:
            reference_project = OnboardingProject(project_name=ref_project_name, overwrite=self.overwrite)
            self.reference_projects.append(reference_project)
        self.no_clean: bool = onboarding_config.get("no_clean", False)
        self.copy_from: Path | None = onboarding_config.get("copy_from", None)
        self.datestamp: bool = onboarding_config.get("datestamp", False)
        self.extract_corpora: bool = onboarding_config.get("extract_corpora", False)
        self.collect_verse_counts: bool = onboarding_config.get("collect_verse_counts", False)
        self.wildebeest: bool = onboarding_config.get("wildebeest", False)
        self.stats: bool = onboarding_config.get("stats", False)
        self.align: bool = onboarding_config.get("align", False)
        self.align_isos: List[str] = onboarding_config.get("align_isos", [])
        output_folder = onboarding_config.get("output_folder", None)
        self.output_folder: Path | None = Path(output_folder) if output_folder else None

    def process_onboarding_request(self) -> None:
        self.create_log_file()
        LOGGER.info(f"Processing onboarding request for main project '{self.main_project.project_name}'")
        for project in [self.main_project] + self.reference_projects:
            if project == self.main_project:
                LOGGER.info(f"Onboarding main project '{self.main_project.project_name}'")
            else:
                LOGGER.info(f"Onboarding reference project '{project.project_name}'")
            self.onboard_project(project)

        if self.collect_verse_counts:
            self.collect_verse_counts_wrapper(self.config.get("verse_counts", {}))

        if self.main_project.get_extract_path() is None and (self.stats or self.align):
            LOGGER.error(
                f"Main Project, {self.main_project.project_name}, has no extract file. Skipping stats and alignments."
            )
            close_logger(self.log_file_path)
            return

        if self.stats:
            self.main_project.calculate_tokenization_stats(
                self.config.get("stats", None),
                [
                    ref_project.get_extract_path().stem
                    for ref_project in self.reference_projects
                    if ref_project.get_extract_path() is not None
                ],
                [ref_project.get_extract_iso_code() for ref_project in self.reference_projects],
            )

        if self.align:
            self.align_main_project()

        close_logger(self.log_file_path)

    def align_main_project(self) -> None:
        iso_codes = set()
        if self.align_isos:
            iso_codes.update(self.align_isos)
        for project in [self.main_project] + self.reference_projects:
            iso_code = project.get_extract_iso_code()
            if iso_code:
                iso_codes.add(iso_code)
        self.main_project.align_wrapper(
            align_config=self.config.get("align", None),
            ref_project_extract_file_names=[
                ref_project.get_extract_path().stem
                for ref_project in self.reference_projects
                if ref_project.get_extract_path() is not None
            ],
            iso_codes=iso_codes,
        )

    def prepare_and_upload_projects(self) -> None:
        if not self.copy_from:
            self.setup_output()
            return
        for project in [self.main_project] + self.reference_projects:
            if project == self.main_project:
                LOGGER.info(f"Preparing and Uploading main project '{self.main_project.project_name}'")
            else:
                LOGGER.info(f"Preparing and Uploading reference project '{project.project_name}'")
            self.upload_project(project)

        self.setup_output()

        self.config["onboarding"]["main_project"] = self.main_project.project_name
        self.config["onboarding"]["ref_projects"] = [project.project_name for project in self.reference_projects]
        self.config["onboarding"]["copy_from"] = None
        self.config["onboarding"]["datestamp"] = False
        self.config["onboarding"]["no_clean"] = False
        self.config["onboarding"]["output_folder"] = str(self.output_folder)

        with open(self.get_config_path(), "w") as f:
            yaml.dump(self.config, f)

    def upload_project(self, onboarding_project: OnboardingProject) -> None:
        original_project_name = onboarding_project.project_name
        if self.copy_from:
            onboarding_project.setup_local_project(original_project_name, self.copy_from, self.datestamp)

        if not self.no_clean:
            LOGGER.info(f"Cleaning Paratext project: {onboarding_project.project_name}.")
            process_single_project_for_cleaning(
                onboarding_project.local_project_path,
            )

        if onboarding_project.is_resource() and onboarding_project.check_resource_hash():
            LOGGER.info(f"Resource '{onboarding_project.project_name}' is up to date. Skipping uploading.")
            return
        elif onboarding_project.is_resource():
            LOGGER.info(
                f"Resource '{onboarding_project.project_name}' is outdated. Uploading new version of the resource."
            )
            onboarding_project.update_resource()

        if self.copy_from:
            LOGGER.info(
                f"Copying project: {onboarding_project.project_name} from {self.copy_from} to {SIL_NLP_ENV.pt_projects_dir}/{onboarding_project.project_name}"
            )
            source_path = Path(self.copy_from)
            if source_path.name != onboarding_project.project_name:
                source_path = Path(source_path / onboarding_project.project_name)
            paratext_project_dir: Path = create_paratext_project_folder_if_not_exists(onboarding_project.project_name)
            copy_paratext_project_folder(source_path, paratext_project_dir, overwrite=self.overwrite)
            if onboarding_project.project_name != original_project_name:
                shutil.rmtree(source_path)

    def onboard_project(self, onboarding_project: OnboardingProject) -> None:
        if self.extract_corpora:
            onboarding_project.extract_corpora_wrapper(self.config.get("extract_corpora", {}))
            if onboarding_project.get_extract_path() is None:
                LOGGER.error(
                    f"No extract file was created for {onboarding_project.project_name}. Stopping onboarding of this project."
                )
                return

        if self.wildebeest:
            onboarding_project.wildebeest_analysis_wrapper(self.config.get("wildebeest", {}))

    def collect_verse_counts_wrapper(self, verse_counts_config: dict) -> None:
        input_folder = verse_counts_config.get("input_folder", SIL_NLP_ENV.mt_scripture_dir)

        file_patterns = verse_counts_config.get("files", None)
        if file_patterns is None:
            file_patterns = [
                project.get_extract_path().name
                for project in [self.main_project] + self.reference_projects
                if project.get_extract_path() is not None
            ]
            file_patterns = ";".join(file_patterns) if file_patterns else None

        input_folder_path = Path(input_folder)
        if not input_folder_path.exists():
            LOGGER.error(f"Input folder '{input_folder_path}' does not exist. Skipping verse counts collection.")
            return

        project_names = ", ".join([project.project_name for project in [self.main_project] + self.reference_projects])
        LOGGER.info(f"Collecting verse counts for project(s) {project_names}")
        collect_verse_counts(
            input_folder=input_folder_path,
            output_folder=self.output_folder,
            file_patterns=file_patterns,
            deutero=verse_counts_config.get("deutero", False),
            recount=verse_counts_config.get("recount", False),
        )

    def get_config_path(self) -> Path:
        return self.output_folder / "onboarding_config.yml"

    def setup_output(self) -> None:
        self.output_folder = Path(
            SIL_NLP_ENV.mt_experiments_dir / "_OnboardingRequests" / f"{self.main_project.project_name}_Request"
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)
        for project in [self.main_project] + self.reference_projects:
            project.set_output_folder(self.output_folder)

    def create_log_file(self) -> None:
        self.log_file_path = self.output_folder / "onboarding.log"
        if not self.log_file_path.exists():
            self.log_file_path.touch()
        set_logger(self.log_file_path)


def append_datestamp(project_name: str) -> str:
    now = datetime.now()
    datestamp = now.strftime("%Y_%m_%d")
    return f"{project_name}_{datestamp}"


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


def get_config(config_path: str) -> dict:
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
        with config_file.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    else:
        return {}


def set_logger(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
        force=True,
    )


def close_logger(log_file: Path) -> None:
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file.absolute()):
            handler.close()
            logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performs several steps to onboard a new project before training a model.",
    )

    parser.add_argument(
        "main_projects",
        help="The Main Paratext project name(s) for onboarding. The project(s) will be stored on the bucket at Paratext/projects/<project>.",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--ref-projects",
        help="The Reference Paratext project name(s) for onboarding the main project(s). The project(s) will be stored on the bucket at Paratext/projects/<project>.",
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
        help="Path(s) to a configuration file in YAML format. This is used to configure the onboarding process.",
        default=None,
        nargs="+",
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
    parser.add_argument(
        "--stats",
        default=False,
        action="store_true",
        help="Compute tokenization statistics on the main project and reference projects.",
    )
    parser.add_argument(
        "--align",
        default=False,
        action="store_true",
        help="Run alignments between the main project and reference projects.",
    )
    parser.add_argument(
        "--align-isos",
        default=None,
        nargs="+",
        help="List of ISO codes to use for determining standard alignment projects to include in the alignments step, along with the reference project isos.",
    )
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    parser.add_argument(
        "--clearml-tag",
        metavar="tag",
        choices=TAGS_LIST,
        default=None,
        type=str,
        help=f"Tag to add to the ClearML Task - {TAGS_LIST}",
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

    if args.config and len(args.main_projects) != len(args.config):
        parser.errror("Number of config paths does not match number of main projects.")

    project_configs = {}
    if args.config:
        for i, main_project in enumerate(args.main_projects):
            project_configs[main_project] = args.config[i]

    onboarding_requests = []
    for main_project in args.main_projects:
        if args.config:
            config_file = Path(project_configs[main_project])
            if not config_file.exists():
                raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
            with config_file.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
        else:
            config = {}
        if config.get("onboarding", None) is None:
            if args.align_isos and not args.align:
                args.align = True
            config["onboarding"] = {
                "main_project": main_project,
                "ref_projects": args.ref_projects if args.ref_projects else [],
                "copy_from": str(args.copy_from) if args.copy_from else None,
                "datestamp": args.datestamp,
                "overwrite": args.overwrite,
                "extract_corpora": args.extract_corpora,
                "collect_verse_counts": args.collect_verse_counts,
                "wildebeest": args.wildebeest,
                "stats": args.stats,
                "align": args.align,
                "align_isos": args.align_isos if args.align_isos else [],
                "output_folder": None,
            }

        onboarding_request = OnboardingRequest(
            config=config,
        )
        onboarding_request.prepare_and_upload_projects()
        onboarding_requests.append(onboarding_request)

    if args.clearml_queue is not None:
        project_names = [onboarding_request.main_project.project_name for onboarding_request in onboarding_requests]
        config_paths = [str(onboarding_request.get_config_path()) for onboarding_request in onboarding_requests]
        sys.argv = [
            "",
            "--config",
            *config_paths,
            "--clearml-queue",
            args.clearml_queue,
            "--clearml-tag",
            args.clearml_tag,
            *project_names,
        ]
        task_name = f"Onboarding - {', '.join([req.main_project.project_name for req in onboarding_requests])}"
        clearml = SILClearML(task_name, args.clearml_queue, tag=args.clearml_tag, skip_config=True)

    for onboarding_request in onboarding_requests:
        onboarding_request.process_onboarding_request()


if __name__ == "__main__":
    main()
