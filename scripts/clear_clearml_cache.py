import glob
import os
import shutil
from pathlib import Path

from silnlp.nmt.clearml_connection import SILClearML

SILClearML("utilities/clear_cache", "production")


def remove_files(path):
    print("deleting " + path)
    files = glob.glob(path + "/*")
    for f in files:
        if Path.is_dir(Path(f)):
            shutil.rmtree(f)
        else:
            os.remove(f)


# shutil.rmtree("/root/.cache/huggingface")
# shutil.rmtree("/var/cache/apt/archives")
remove_files("/root/.cache/pip")
remove_files("/root/.cache/pypoetry")
remove_files("/root/.clearml/pip-download-cache")
# shutil.rmtree("/clearml_agent_cache")
# shutil.rmtree("/root/.clearml/venvs-cache")
# shutil.rmtree("/root/.cache/vcs-cache")
