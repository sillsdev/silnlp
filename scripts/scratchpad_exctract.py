import logging
import re
from pathlib import Path
from typing import List, Set
import multiprocessing
import shutil
import json
import pandas as pd
import numpy as np

from silnlp.common.corpus import count_lines
from silnlp.common.environment import SIL_NLP_ENV
from silnlp.common.paratext import extract_project

LOGGER = logging.getLogger("silnlp")

re_versification_setting = re.compile("\<Versification\>[0-9]+\<")
vref_file = SIL_NLP_ENV.assets_dir / "vref.txt"
expected_verse_count = count_lines(vref_file)


def extract_worker(kwargs):
    return extract_project_with_logging(**kwargs)


def extract_project_with_logging(project_dir, output_dir):
    LOGGER.info(f"Extracting {project_dir.name}...")
    try:
        corpus_filename, verse_count = extract_project(project_dir, output_dir, record_dropped_verses=True)
        # check if the number of lines in the file is correct (the same as vref.txt)
        LOGGER.info(f"# of Verses: {verse_count}")
        if verse_count != expected_verse_count:
            LOGGER.error(f"The number of verses is {verse_count}, but should be {expected_verse_count}.")
        LOGGER.info("Done.")
    except:
        LOGGER.exception(f"Error processing {project_dir.name}")


def extract_projects(project_parent_dir: Path, output_dir: Path, multiprocess=True):
    # Which projects have data we can find?
    project_dirs = [dir for dir in project_parent_dir.iterdir() if dir.is_dir()]

    if multiprocess:
        all_kwargs = []
        for project_dir in project_dirs:
            all_kwargs.append({"project_dir": project_dir, "output_dir": output_dir})
        cpu_num = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cpu_num)
        result = pool.map_async(extract_worker, all_kwargs)
        result.get()
        pool.close()
        pool.join()
    else:
        for project_dir in project_dirs:
            extract_project_with_logging(project_dir, output_dir)


def build_analysis_table(output_dir: Path, original_extraction_dir: Path):
    vref_indexes = [l.strip() for l in vref_file.open("r", encoding="utf-8").readlines()]
    len_index = len(vref_indexes)
    versification_files = list(output_dir.glob("*versification.txt"))
    extraction_files = list(original_extraction_dir.glob("*.txt"))

    re_name_extraction = re.compile("[\._]")
    translation_dict = {}
    for f in versification_files + extraction_files:
        name = re_name_extraction.split(f.name)[0]
        if name not in translation_dict:
            translation_dict[name] = {}
        if f.name.endswith("versification.txt"):
            translation_dict[name]["versification"] = f
        else:
            translation_dict[name]["extract"] = f

    extraction_dict = {}
    versification_dict = {}
    for k in list(translation_dict.keys()):
        if "versification" not in translation_dict[k] or "extract" not in translation_dict[k]:
            print(f"Dropped files - missing versification or extract: {translation_dict.pop(k)}")
            continue

        temp = [l.strip() != "" for l in translation_dict[k]["extract"].open("r", encoding="utf-8").readlines()]
        if len(temp) == len_index:
            extraction_dict[k] = temp
        else:
            print(f"Dropping {k}: length should be {len_index} but is {len(temp)}")
            continue

        temp = [l.strip() for l in translation_dict[k]["versification"].open("r", encoding="utf-8").readlines()]
        versification_dict[k] = {}
        versification_dict[k]["versification_name"] = temp[0]
        versification_dict[k]["dropped_verses"] = [l.split("; ")[0] for l in temp[1:]]

    ext_pres_df = pd.DataFrame(data=extraction_dict, index=vref_indexes)
    chp_gb = ext_pres_df.groupby([i.split(":")[0] for i in ext_pres_df.index], sort=False)
    gps = [g for g in chp_gb]
    missing_verses = {c: [] for c in ext_pres_df.columns}
    for g in gps:
        print(g[0])
        in_KJV = g[1].loc[:, "en-engKJV"]
        has_verses = g[1].loc[:, g[1].any(axis=0)]
        missing = has_verses.lt(in_KJV, axis=0)
        missing = missing.loc[:, missing.any(axis=0)]
        for m in missing.iteritems():
            rising = m[1].index[m[1] & ~m[1].shift(1).fillna(False)]
            falling_verse = m[1].index[m[1] & ~m[1].shift(-1).fillna(False)]
            temp_verses = []
            for i, ref in enumerate(rising):
                if ref == falling_verse[i]:
                    temp_verses.append(ref)
                else:
                    temp_verses.append(f"{ref}-{falling_verse[i].split(':')[-1]}")
            missing_verses[m[0]].extend(temp_verses)
    with (output_dir / "missing.json").open("w+") as mj_fh:
        json.dump(missing_verses, mj_fh, indent=2)

    # chapters_df = ext_pres_df.groupby([i.split(':')[0] for i in ext_pres_df.index],sort=False).sum().transpose()
    books_df = ext_pres_df.groupby([i.split(" ")[0] for i in ext_pres_df.index], sort=False).sum().transpose()
    books_df["Versification"] = [versification_dict[k]["versification_name"] for k in versification_dict]
    books_df["MissingVerses"] = [" ".join(missing_verses[key]) for key in missing_verses]
    books_df["DroppedVerses"] = [" ".join(versification_dict[k]["dropped_verses"]) for k in versification_dict]
    books_df.to_csv(output_dir / "versification_report.csv")


if __name__ == "__main__":
    project_parent_dir = Path(r"C:\Users\johnm\Documents\repos\UnzippedParatext")
    output_dir = Path(r"C:\Users\johnm\Documents\repos\Extractions\reextracted")
    original_extraction_dir = Path(r"C:\Users\johnm\Documents\repos\Extractions\Paratext_versification")

    extract_projects(project_parent_dir=project_parent_dir, output_dir=output_dir, multiprocess=True)
    # build_analysis_table(output_dir=output_dir,original_extraction_dir=original_extraction_dir)
