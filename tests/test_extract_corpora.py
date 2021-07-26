import shutil
import os
from . import helper
from silnlp.common.paratext import extract_project, extract_term_renderings
from silnlp.common.environment import SIL_NLP_ENV


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
        extract_project(project.parts[-1], include_texts="", exclude_texts="", include_markers=False)
        extract_term_renderings(project.parts[-1])

    helper.compare_folders(truth_folder=scr_truth_dir, computed_folder=SIL_NLP_ENV.mt_scripture_dir)


def test_extract_corpora_error():
    project = SIL_NLP_ENV.pt_projects_dir / "not_a_project"
    error = "no error"
    try:
        extract_project(project.parts[-1], include_texts="", exclude_texts="", include_markers=False)
    except Exception as e:
        error = e.args[0]
    assert error.startswith("Error reading file")
