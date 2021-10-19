import logging
import pandas as pd
import numpy as np
from pathlib import Path

LOGGER = logging.getLogger("silnlp")

scripture_dir = Path("C:\\Users\\johnm\\Documents\\repos\\bible-parallel-corpus-internal\\corpus\\scripture")
alignment_dir = scripture_dir / "..\\..\\"

verses = [l.strip() for l in (scripture_dir / "vref.txt").open().readlines()]

alignment_folders = [x[0] for x in os.walk(alignment_dir) if "alignment.scores.txt" in x[2]]
alignment_names = [x.split("\\")[-1] for x in alignment_folders]
data = {}
for i, name in enumerate(alignment_names):
    data[name] = [float(x.strip()) for x in (Path(alignment_folders[i]) / "alignment.scores.txt").open().readlines()]

al_df = pd.DataFrame(index=verses, data=data)
al_df[al_df == -1] = np.nan

al_df.iloc[:50000:1000, :10].plot()
al_df.min(axis=0)