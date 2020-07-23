import os
import subprocess
from pathlib import Path

from nlp.common.environment import paratextPreprocessedDir


def get_git_revision_hash() -> str:
    script_path = Path(__file__)
    repo_dir = script_path.parent.parent.parent
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()


def get_root_dir(exp_name: str) -> str:
    return os.path.join(paratextPreprocessedDir, "tests", exp_name)
