import shutil
import tempfile
import os
from pathlib import Path
from . import helper
from silnlp.common.paratext import extract_project, extract_term_renderings
from silnlp.common.environment import SNE


scr_truth_dir = SNE._MT_DIR / "scripture"

# set scripture directory to temp
with tempfile.TemporaryDirectory() as src_dir:
    SNE._MT_SCRIPTURE_DIR = Path(src_dir)


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
