import logging
import subprocess
from pathlib import Path


# Initialize logger
LOGGER = logging.getLogger("silnlp")
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

# Update or add dotnet machine environment
try:
    result = subprocess.run(
        ["dotnet", "tool", "restore"], cwd=Path(__file__).parent.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    LOGGER.debug(result.stdout)
except:
    LOGGER.error(
        "The .NET Core SDK needs to be installed (https://dotnet.microsoft.com/download) to be able to use the functionality in SIL.Machine.Tool (most of silnlp)."
    )
