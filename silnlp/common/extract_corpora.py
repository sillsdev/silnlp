# Extracts parallel corpora from Paratext projects.
# This script is dependent on the SIL.Machine.Translator tool being installed.
# To install:
# 1. Install .NET Core SDK (https://dotnet.microsoft.com/download)
# 2. Run "dotnet tool install SIL.Machine.Translator -g"


import os
import subprocess
import xml.etree.ElementTree as ET

from nlp.common.environment import paratextPreprocessedDir, paratextUnzippedDir


def get_iso(project_dir):
    tree = ET.parse(os.path.join(project_dir, "Settings.xml"))
    iso_elem = tree.getroot().find("LanguageIsoCode")
    iso = iso_elem.text
    return iso.split(":::")[0]


def extract_corpus(output_dir, src_iso, src_dir, trg_iso, trg_dir):
    print("Extracting", src_iso, "<->", trg_iso)
    subprocess.run(
        [
            "translator",
            "extract",
            "-s",
            f"pt,{src_dir}",
            "-t",
            f"pt,{trg_dir}",
            "-so",
            os.path.join(output_dir, f"all-{src_iso}-{trg_iso}.{src_iso}"),
            "-to",
            os.path.join(output_dir, f"all-{src_iso}-{trg_iso}.{trg_iso}"),
            "-st",
            "null",
            "-tt",
            "null",
        ]
    )


def main():
    langs1 = {"bru"}
    langs2 = {"ctu", "cuk", "en", "ifa", "kek", "mps", "nch", "qxn", "rop", "xon"}
    my_output_dir = os.path.join(paratextPreprocessedDir, "1-to-n-2020.03.16-15.28.09.276777")
    projects1 = list()
    projects2 = list()
    for path in os.listdir(paratextUnzippedDir):
        project_dir = os.path.join(paratextUnzippedDir, path)
        if os.path.isdir(project_dir):
            iso = get_iso(project_dir)
            if iso in langs1:
                projects1.append((iso, project_dir))
            if iso in langs2:
                projects2.append((iso, project_dir))

    projects1.sort(key=lambda p: p[0])
    projects2.sort(key=lambda p: p[0])
    extracted = set()
    for iso1, project_dir1 in projects1:
        for iso2, project_dir2 in projects2:
            if iso1 == iso2:
                continue
            key = tuple(sorted([iso1, iso2]))
            if not key in extracted:
                extract_corpus(my_output_dir, iso1, project_dir1, iso2, project_dir2)
                extracted.add(key)


if __name__ == "__main__":
    main()
