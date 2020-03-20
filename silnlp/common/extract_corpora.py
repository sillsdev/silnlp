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


def extract_corpus(output_dir, iso, project_dir):
    name = os.path.basename(project_dir)
    print("Extracting", name, f"({iso})")
    ref_dir = os.path.join(paratextUnzippedDir, "Ref")
    subprocess.run(
        [
            "translator",
            "extract",
            "-s",
            f"pt,{ref_dir}",
            "-t",
            f"pt,{project_dir}",
            "-to",
            os.path.join(output_dir, f"{iso}-{name}.txt"),
            "-st",
            "null",
            "-tt",
            "null",
            "-as",
        ]
    )


def main():
    langs = {"en", "es", "fr", "bru", "ctu", "cuk", "ifa", "kek", "mps", "nch", "qxn", "rop", "xon"}
    output_dir = os.path.join(paratextPreprocessedDir, "data")
    os.makedirs(output_dir, exist_ok=True)
    for path in os.listdir(paratextUnzippedDir):
        project_dir = os.path.join(paratextUnzippedDir, path)
        if os.path.isdir(project_dir):
            iso = get_iso(project_dir)
            if langs is None or iso in langs:
                extract_corpus(output_dir, iso, project_dir)


if __name__ == "__main__":
    main()
