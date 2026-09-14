import os
import subprocess
import sys


def test_importing_corpora_does_not_require_a_configured_data_directory():
    # Unit tests have to import the corpus model without the bucket mounted.
    environment = {key: value for key, value in os.environ.items() if key != "SIL_NLP_DATA_PATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import silnlp.nmt.corpora"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
