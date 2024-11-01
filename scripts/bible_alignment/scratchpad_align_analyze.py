import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

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
al_df_norm = al_df.copy()
al_df["book"] = al_df.index.str.extract(r"(^\w+)")[0].to_numpy()
books = list(al_df["book"].unique())
al_df_norm = al_df - al_df.mean(axis=0)
al_df_norm["book"] = al_df["book"]

al_means = al_df_norm.groupby(by="book").agg(np.nanmean)
al_means = al_means.loc[books, :]
al_means = al_means[~al_means.mean(axis=1).isna()]
fig = plt.figure(figsize=(20, 8))
ax = fig.gca()
al_means = al_means.iloc[:, ::10]
ax.pcolor(al_means)
plt.yticks(np.arange(0.5, len(al_means.index), 1), al_means.index)
plt.xticks(np.arange(0.5, len(al_means.columns), 1), al_means.columns)
plt.show()
