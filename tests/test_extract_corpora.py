import shutil
import os
from pathlib import Path
from . import helper
from silnlp.common.paratext import extract_project, extract_term_renderings


pp_dir = Path(__file__).parent / "Paratext/projects"

scr_dir = Path(__file__).parent / "MT/scripture"
scr_truth_dir = Path(__file__).parent / "MT/scripture_truth"


def test_extract_corpora():
    shutil.rmtree(scr_dir, ignore_errors=True)
    os.mkdir(scr_dir)
    pp_subdirs = [folder for folder in pp_dir.glob("*/")]
    for project in pp_subdirs:
        extract_project(project.parts[-1], include_texts="", exclude_texts="", include_markers=False)
        extract_term_renderings(project.parts[-1])

    helper.compare_folders(truth_folder=scr_truth_dir, computed_folder=scr_dir)


def test_extract_corpora_error():
    project = pp_dir / "not_a_project"
    error = "no error"
    try:
        extract_project(project.parts[-1], include_texts="", exclude_texts="", include_markers=False)
    except Exception as e:
        error = e.args[0]
    assert error.startswith("Error reading file")
