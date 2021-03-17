import os
import re
import subprocess
from collections import OrderedDict
from typing import Dict, List, Optional
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from .corpus import get_terms_glosses_path, get_terms_metadata_path, load_corpus
from .environment import PT_BIBLICAL_TERMS_LISTS_DIR, PT_PREPROCESSED_DIR, PT_UNZIPPED_DIR
from .utils import get_repo_dir

_TERMS_LISTS = {
    "Major": "BiblicalTerms.xml",
    "All": "AllBiblicalTerms.xml",
    "SilNt": "BiblicalTermsSILNT.xml",
    "Pt6": "BiblicalTermsP6NT.xml",
    "Project": "ProjectBiblicalTerms.xml",
}


def get_project_dir(project: str) -> str:
    return os.path.join(PT_UNZIPPED_DIR, project)


def get_iso(settings_tree: ElementTree.ElementTree) -> str:
    iso = settings_tree.getroot().findtext("LanguageIsoCode")
    assert iso is not None
    index = iso.index(":")
    return iso[:index]


def extract_project(project: str, include_texts: str, exclude_texts: str) -> None:
    project_dir = get_project_dir(project)
    settings_tree = ElementTree.parse(os.path.join(project_dir, "Settings.xml"))
    iso = get_iso(settings_tree)

    ref_dir = os.path.join(PT_UNZIPPED_DIR, "Ref")
    arg_list = ["dotnet", "machine", "build-corpus", ref_dir, project_dir, "-sf", "pt", "-tf", "pt", "-as", "-ie"]
    output_basename = f"{iso}-{project}"
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
    arg_list.append(os.path.join(PT_PREPROCESSED_DIR, "data", f"{output_basename}.txt"))

    subprocess.run(arg_list, cwd=get_repo_dir())


def escape_id(id: str) -> str:
    return escape(id).replace("\n", "&#xA;")


def clean_term(term_str: str) -> str:
    term_str = term_str.strip()
    term_str = re.sub("\(.*?\)", "", term_str)
    return " ".join(term_str.split())


def extract_terms_list(list_type: str, project: Optional[str] = None) -> None:
    list_file_name = _TERMS_LISTS.get(list_type)
    if list_file_name is None:
        return

    list_name = list_type
    if project is not None:
        list_name = project

    dir = str(PT_BIBLICAL_TERMS_LISTS_DIR) if project is None else os.path.join(PT_UNZIPPED_DIR, project)
    terms_xml_path = os.path.join(dir, list_file_name)

    terms_metadata_path = get_terms_metadata_path(list_name)
    terms_glosses_path = get_terms_glosses_path(list_name)

    with open(terms_metadata_path, "w", encoding="utf-8", newline="\n") as terms_metadata_file, open(
        terms_glosses_path, "w", encoding="utf-8", newline="\n"
    ) as terms_glosses_file:
        terms_tree = ElementTree.parse(terms_xml_path)
        for term_elem in terms_tree.getroot().findall("Term"):
            id = term_elem.get("Id")
            if id is None:
                continue
            id = escape_id(id)
            cat = term_elem.findtext("Category", "?")
            if cat == "":
                cat = "?"
            domain = term_elem.findtext("Domain", "?")
            if domain == "":
                domain = "?"
            gloss_str = term_elem.findtext("Gloss", "")
            match = re.match("\[(.+?)\]", gloss_str)
            if match is not None:
                gloss_str = match.group(1)
            gloss_str = gloss_str.replace("?", "")
            gloss_str = clean_term(gloss_str)
            gloss_str = re.sub("\s+\d+(\.\d+)*$", "", gloss_str)
            glosses = re.split("[;,/]", gloss_str)
            glosses = [gloss.strip() for gloss in glosses if gloss.strip() != ""]
            terms_metadata_file.write(f"{id}\t{cat}\t{domain}\n")
            terms_glosses_file.write("\t".join(OrderedDict.fromkeys(glosses)) + "\n")


def extract_terms_list_from_renderings(project: str, renderings_tree: ElementTree.ElementTree) -> None:
    terms_metadata_path = get_terms_metadata_path(project)
    with open(terms_metadata_path, "w", encoding="utf-8", newline="\n") as terms_metadata_file:
        for rendering_elem in renderings_tree.getroot().findall("TermRendering"):
            id = rendering_elem.get("Id")
            if id is None:
                continue
            id = escape_id(id)
            if rendering_elem.get("Guess") != "false" or rendering_elem.findtext("Renderings", "") == "":
                continue

            terms_metadata_file.write(f"{id}\t?\t?\n")


def extract_term_renderings(project: str) -> None:
    project_dir = get_project_dir(project)
    renderings_path = os.path.join(project_dir, "TermRenderings.xml")
    if not os.path.isfile(renderings_path):
        return

    renderings_tree = ElementTree.parse(renderings_path)
    rendering_elems: Dict[str, ElementTree.Element] = {}
    for elem in renderings_tree.getroot().findall("TermRendering"):
        id = elem.get("Id")
        if id is None:
            continue
        id = escape_id(id)
        rendering_elems[id] = elem

    settings_path = os.path.join(project_dir, "Settings.xml")
    settings_tree = ElementTree.parse(settings_path)
    iso = get_iso(settings_tree)
    terms_setting = settings_tree.getroot().findtext("BiblicalTermsListSetting", "Major::BiblicalTerms.xml")

    list_type, terms_project, _ = terms_setting.split(":", maxsplit=3)
    list_name = list_type
    if list_type == "Project":
        if terms_project == project:
            extract_terms_list(list_type, project)
        else:
            extract_terms_list_from_renderings(project, renderings_tree)
        list_name = project

    terms_metadata_path = get_terms_metadata_path(list_name)
    terms_renderings_path = os.path.join(PT_PREPROCESSED_DIR, "terms", f"{iso}-{project}-{list_type}-renderings.txt")
    count = 0
    with open(terms_renderings_path, "w", encoding="utf-8", newline="\n") as terms_renderings_file:
        for line in load_corpus(terms_metadata_path):
            id, _, _ = line.split("\t", maxsplit=3)
            rendering_elem = rendering_elems.get(id)
            renderings: List[str] = []
            if rendering_elem is not None and rendering_elem.get("Guess", "false") == "false":
                renderings_str = rendering_elem.findtext("Renderings", "")
                if renderings_str != "":
                    renderings_str = renderings_str.replace("*", "")
                    renderings_str = clean_term(renderings_str)
                    renderings = renderings_str.strip().split("||")
                    renderings = [rendering.strip() for rendering in renderings if rendering.strip() != ""]
            terms_renderings_file.write("\t".join(OrderedDict.fromkeys(renderings)) + "\n")
            if len(renderings) > 0:
                count += 1
    if count == 0:
        os.remove(terms_renderings_path)
        if list_type == "Project":
            os.remove(terms_metadata_path)
            terms_glosses_path = get_terms_glosses_path(list_name)
            if os.path.isfile(terms_glosses_path):
                os.remove(terms_glosses_path)
    print(f"# of Terms written: {count}")
