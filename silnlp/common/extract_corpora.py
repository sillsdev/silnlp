# Extracts parallel corpora from Paratext projects.
# This script is dependent on the SIL.Machine.Translator tool being installed.
# To install:
# 1. Install .NET Core SDK (https://dotnet.microsoft.com/download)
# 2. Run "dotnet tool install SIL.Machine.Translator -g"


import argparse
import os
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional

from nlp.common.environment import paratextPreprocessedDir, paratextUnzippedDir


def get_iso(project_dir: str) -> Optional[str]:
    tree = ET.parse(os.path.join(project_dir, "Settings.xml"))
    iso_elem = tree.getroot().find("LanguageIsoCode")
    if iso_elem is None:
        return None
    iso = iso_elem.text
    if iso is None:
        return None

    index = iso.index(":")
    return iso[:index]


def extract_corpus(output_dir: str, iso: str, project_dir: str, include_texts: str, exclude_texts: str) -> None:
    name = os.path.basename(project_dir)
    print("Extracting", name, f"({iso})")
    ref_dir = os.path.join(paratextUnzippedDir, "Ref")
    arg_list = [
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
    if len(include_texts) > 0:
        arg_list.append("-i")
        arg_list.append(include_texts)
    if len(exclude_texts) > 0:
        arg_list.append("-e")
        arg_list.append(exclude_texts)
    subprocess.run(arg_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts text corpora from Paratext projects")
    parser.add_argument("projects", nargs="*", metavar="name", help="Paratext project")
    parser.add_argument(
        "--include", metavar="texts", default="", help="The texts to include; e.g., '*NT*', '*OT*', 'GEN,EXO'"
    )
    parser.add_argument(
        "--exclude", metavar="texts", default="", help="The texts to exclude; e.g., '*NT*', '*OT*', 'GEN,EXO'"
    )
    args = parser.parse_args()

    projects = set(args.projects)

    # Which projects have data we can find?
    projects_found = []

    output_dir = os.path.join(paratextPreprocessedDir, "data")
    os.makedirs(output_dir, exist_ok=True)
    for path in os.listdir(paratextUnzippedDir):
        if path == "Ref" or (len(projects) > 0 and path not in projects):
            continue
        else:
            projects_found.append(path)

    # Process the projects that have data and tell the user.
    if projects_found:
        output_dir = os.path.join(paratextPreprocessedDir, "data")
        os.makedirs(output_dir, exist_ok=True)
        for project in projects_found:
            project_dir = os.path.join(paratextUnzippedDir, project)
            if os.path.isdir(project_dir):
                iso = get_iso(project_dir)
                if iso is not None:
                    extract_corpus(output_dir, iso, project_dir, args.include, args.exclude)
                    print(f"Processed: {project_dir}\nOutput saved in: {output_dir}")
    else:
        print(f"Couldn't find any data to process for any project.")

    # Tell the user which projects couldn't be found.
    for project in projects:
        if project not in projects_found:
            print(f"Couldn't find project {project}")


if __name__ == "__main__":
    main()
