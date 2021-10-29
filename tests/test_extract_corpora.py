import os
import shutil

from silnlp.common.environment import SIL_NLP_ENV
from silnlp.common.paratext import extract_project, extract_term_renderings, get_project_dir

from . import helper

# set scripture directory to temp
SIL_NLP_ENV.set_data_dir()
SIL_NLP_ENV.mt_scripture_dir = SIL_NLP_ENV.mt_dir / "temp_scripture"
SIL_NLP_ENV.mt_scripture_dir.mkdir(exist_ok=True)
scr_truth_dir = SIL_NLP_ENV.mt_dir / "scripture"


def test_extract_corpora():
    shutil.rmtree(SIL_NLP_ENV.mt_scripture_dir, ignore_errors=True)
    os.mkdir(SIL_NLP_ENV.mt_scripture_dir)
    pp_subdirs = [folder for folder in SIL_NLP_ENV.pt_projects_dir.glob("*/")]
    for project in pp_subdirs:
        corpus_filename = extract_project(project, SIL_NLP_ENV.mt_scripture_dir)
        extract_term_renderings(project, corpus_filename)

    helper.compare_folders(truth_folder=scr_truth_dir, computed_folder=SIL_NLP_ENV.mt_scripture_dir)


def test_extract_corpora_error():
    project = SIL_NLP_ENV.pt_projects_dir / "not_a_project"
    error = "no error"
    try:
        extract_project(project, SIL_NLP_ENV.mt_scripture_dir)
    except Exception as e:
        error = e.args[0]
    assert error.startswith("The project directory does not contain a settings file.")
