import os
from typing import Dict

import pandas as pd

from nlp.common.environment import align_experiments_dir


def main() -> None:
    translations = [
        "jvb",
        "mcl",
        "ccb",
        "cuvmp",
        "rcuv",
        "esv",
        "kjv",
        "niv11",
        "niv84",
        "nrsv",
        "rsv",
        "hovr",
        "shk",
        "khov",
        "nrt",
    ]
    aligners = ["PT", "CLEAR", "FastAlign", "SMT", "IBM-4", "HMM", "IBM-1", "IBM-2"]

    for testament in ["nt", "ot"]:
        data: Dict[str, pd.DataFrame] = {}
        for translation in translations:
            scores_path = os.path.join(align_experiments_dir, translation + "." + testament, "scores.csv")
            if os.path.isfile(scores_path):
                df = pd.read_csv(scores_path, index_col=0)
                data[translation] = df

        for metric in ["AER", "F-Score", "Precision", "Recall"]:
            output_path = os.path.join(align_experiments_dir, testament + "." + metric + ".csv")
            with open(output_path, "w") as output_file:
                output_file.write("Model," + ",".join(filter(lambda t: t in data, translations)) + "\n")
                for aligner in aligners:
                    output_file.write(aligner)
                    for translation in translations:
                        df = data.get(translation)
                        if df is None:
                            continue
                        output_file.write(",")
                        output_file.write(str(df.at[aligner, metric]))
                    output_file.write("\n")


if __name__ == "__main__":
    main()
