import shutil
import os
from . import helper
from silnlp.common.paratext import extract_project, extract_term_renderings
from silnlp.common.environment import SNE


# set scripture directory to temp
SNE.set_data_dir()
SNE._MT_SCRIPTURE_DIR = SNE._MT_DIR / "temp_scripture"
SNE._MT_SCRIPTURE_DIR.mkdir(exist_ok=True)
scr_truth_dir = SNE._MT_DIR / "scripture"


def test_extract_corpora():
    shutil.rmtree(SNE._MT_SCRIPTURE_DIR, ignore_errors=True)
    os.mkdir(SNE._MT_SCRIPTURE_DIR)
    pp_subdirs = [folder for folder in SNE._PT_PROJECTS_DIR.glob("*/")]
    for project in pp_subdirs:
        extract_project(project.parts[-1], include_texts="", exclude_texts="", include_markers=False)
        extract_term_renderings(project.parts[-1])

    helper.compare_folders(truth_folder=scr_truth_dir, computed_folder=SNE._MT_SCRIPTURE_DIR)


def test_extract_corpora_error():
    project = SNE._PT_PROJECTS_DIR / "not_a_project"
    error = "no error"
    try:
        extract_project(project.parts[-1], include_texts="", exclude_texts="", include_markers=False)
    except Exception as e:
        error = e.args[0]
    assert error.startswith("Error reading file")
