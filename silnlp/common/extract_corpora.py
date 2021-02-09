# Extracts parallel corpora from Paratext projects.
# This script is dependent on the Machine tool being installed.
# To install:
# 1. Install .NET Core SDK (https://dotnet.microsoft.com/download)
# 2. Run "dotnet tool restore"


import argparse
import os
import re
import subprocess
from typing import Optional, Set, Tuple
from xml.etree import ElementTree

from ..common.environment import PT_PREPROCESSED_DIR, PT_UNZIPPED_DIR
from ..common.utils import get_repo_dir


def get_iso(project_dir: str) -> Optional[str]:
    tree = ElementTree.parse(os.path.join(project_dir, "Settings.xml"))
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
    ref_dir = os.path.join(PT_UNZIPPED_DIR, "Ref")
    arg_list = ["dotnet", "machine", "extract", ref_dir, project_dir, "-sf", "pt", "-tf", "pt", "-as", "-ie"]
    output_basename = f"{iso}-{name}"
    if len(include_texts) > 0 or len(exclude_texts) > 0:
        output_basename += "_"
    if len(include_texts) > 0:
        arg_list.append("-i")
        arg_list.append(include_texts)
        for text in include_texts.split(","):
            text = text.strip("*")
            output_basename += f"+{text}"
    if len(exclude_texts) > 0:
        arg_list.append("-e")
        arg_list.append(exclude_texts)
        for text in exclude_texts.split(","):
            text = text.strip("*")
            output_basename += f"-{text}"

    arg_list.append("-to")
    arg_list.append(os.path.join(output_dir, f"{output_basename}.txt"))

    subprocess.run(arg_list, cwd=get_repo_dir())


def extract_names(output_dir: str, iso: str, project_dir: str) -> None:
    terms_path = os.path.join(project_dir, "ProjectBiblicalTerms.xml")
    renderings_path = os.path.join(project_dir, "TermRenderings.xml")
    if not os.path.isfile(terms_path) or not os.path.isfile(renderings_path):
        return

    name = os.path.basename(project_dir)
    vern_terms_path = os.path.join(output_dir, f"{iso}-{name}-names.txt")
    en_terms_path = os.path.join(output_dir, f"en-{name}-names.txt")

    names: Set[Tuple[str, str]] = set()
    with open(vern_terms_path, "w", encoding="utf-8", newline="\n") as vern_terms_file, open(
        en_terms_path, "w", encoding="utf-8", newline="\n"
    ) as en_terms_file:
        terms_tree = ElementTree.parse(terms_path)
        renderings_tree = ElementTree.parse(renderings_path)
        for term_elem in terms_tree.getroot().findall("Term"):
            cat = term_elem.findtext("Category")
            if cat != "PN":
                continue
            id = term_elem.get("Id")
            rendering_elem = renderings_tree.getroot().find(f"TermRendering[@Id='{id}']")
            if rendering_elem is None:
                continue
            if rendering_elem.get("Guess") != "false":
                continue
            gloss_str = term_elem.findtext("Gloss")
            if gloss_str is None:
                continue
            renderings_str = rendering_elem.findtext("Renderings")
            if renderings_str is None:
                continue
            glosses = re.split("[;,/]", gloss_str.strip())
            renderings = renderings_str.strip().split("||")
            for gloss in glosses:
                gloss = gloss.strip()
                for rendering in renderings:
                    rendering = rendering.strip()
                    rendering = re.sub("\s*\(.*\)", "", rendering)
                    rendering = rendering.strip("*")
                    if (gloss, rendering) not in names:
                        vern_terms_file.write(rendering + "\n")
                        en_terms_file.write(gloss + "\n")
                        names.add((gloss, rendering))


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

    output_dir = os.path.join(PT_PREPROCESSED_DIR, "data")
    os.makedirs(output_dir, exist_ok=True)
    for path in os.listdir(PT_UNZIPPED_DIR):
        if path == "Ref" or (len(projects) > 0 and path not in projects):
            continue
        else:
            projects_found.append(path)

    # Process the projects that have data and tell the user.
    if projects_found:
        output_dir = os.path.join(PT_PREPROCESSED_DIR, "data")
        os.makedirs(output_dir, exist_ok=True)
        for project in projects_found:
            project_dir = os.path.join(PT_UNZIPPED_DIR, project)
            if os.path.isdir(project_dir):
                iso = get_iso(project_dir)
                if iso is not None:
                    extract_corpus(output_dir, iso, project_dir, args.include, args.exclude)
                    extract_names(output_dir, iso, project_dir)
                    print(f"Extracted {os.path.basename(project_dir)}")
    else:
        print("Couldn't find any data to process for any project.")

    # Tell the user which projects couldn't be found.
    for project in projects:
        if project not in projects_found:
            print(f"Couldn't find project {project}")


if __name__ == "__main__":
    main()
