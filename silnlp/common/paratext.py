import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from xml.sax.saxutils import escape

from lxml import etree
from machine.corpora import FilteredTextCorpus, ParallelTextCorpus, ParatextTextCorpus, Text
from machine.scripture import VerseRef, book_id_to_number
from machine.tokenization import NullTokenizer

from .corpus import get_terms_glosses_path, get_terms_metadata_path, get_terms_vrefs_path, load_corpus
from .environment import SIL_NLP_ENV
from .utils import unique_list

_TERMS_LISTS = {
    "Major": "BiblicalTerms.xml",
    "All": "AllBiblicalTerms.xml",
    "SilNt": "BiblicalTermsSILNT.xml",
    "Pt6": "BiblicalTermsP6NT.xml",
    "Project": "ProjectBiblicalTerms.xml",
}

LOGGER = logging.getLogger(__name__)


def get_project_dir(project: str) -> Path:
    return SIL_NLP_ENV.pt_projects_dir / project


def get_iso(settings_tree: etree.ElementTree) -> str:
    iso = settings_tree.getroot().findtext("LanguageIsoCode")
    assert iso is not None
    index = iso.index(":")
    return iso[:index]


def extract_project(project: str, include_texts: str, exclude_texts: str, include_markers: bool) -> Path:
    project_dir = get_project_dir(project)
    settings_tree = etree.parse(str(project_dir / "Settings.xml"))
    iso = get_iso(settings_tree)

    ref_dir = SIL_NLP_ENV.assets_dir / "Ref"

    tokenizer = NullTokenizer()
    ref_corpus = ParatextTextCorpus(tokenizer, ref_dir)
    project_corpus = ParatextTextCorpus(tokenizer, project_dir, include_markers=include_markers)

    output_basename = f"{iso}-{project}"
    if len(include_texts) > 0 or len(exclude_texts) > 0:
        output_basename += "_"
        include_texts_set: Optional[Set[str]] = None
        if len(include_texts) > 0:
            include_texts_set = set()
            for text in include_texts.split(","):
                include_texts_set.add(text)
                text = text.strip("*")
                output_basename += f"+{text}"
        exclude_texts_set: Optional[Set[str]] = None
        if len(exclude_texts) > 0:
            exclude_texts_set = set()
            for text in exclude_texts.split(","):
                exclude_texts_set.add(text)
                text = text.strip("*")
                output_basename += f"-{text}"

        def filter_corpus(text: Text) -> bool:
            if exclude_texts_set is not None and text.id in exclude_texts_set:
                return False

            if include_texts_set is not None and text.id in include_texts_set:
                return True

            return include_texts_set is None

        ref_corpus = FilteredTextCorpus(ref_corpus, filter_corpus)
        project_corpus = FilteredTextCorpus(project_corpus, filter_corpus)

    if include_markers:
        output_basename += "-m"
    output_filename = SIL_NLP_ENV.mt_scripture_dir / f"{output_basename}.txt"

    parallel_corpus = ParallelTextCorpus(ref_corpus, project_corpus)
    segment_count = 0
    with output_filename.open("w", encoding="utf-8", newline="\n") as output_stream, parallel_corpus.get_segments(
        all_source_segments=True
    ) as segments:
        cur_ref: Optional[Any] = None
        cur_target_line = ""
        cur_target_line_range = True
        for segment in segments:
            if cur_ref is not None and segment.segment_ref != cur_ref:
                output_stream.write(("<range>" if cur_target_line_range else cur_target_line) + "\n")
                segment_count += 1
                cur_target_line = ""
                cur_target_line_range = True

            cur_ref = segment.segment_ref
            if not segment.is_target_in_range or segment.is_target_range_start or len(segment.target_segment) > 0:
                if len(segment.target_segment) > 0:
                    if len(cur_target_line) > 0:
                        cur_target_line += " "
                    cur_target_line += segment.target_segment[0]
                cur_target_line_range = False
        output_stream.write(("<range>" if cur_target_line_range else cur_target_line) + "\n")
        segment_count += 1

    # check if the number of lines in the file is correct (the same as vref.txt - 31104 ending at REV 22:21)
    LOGGER.info(f"# of Segments: {segment_count}")
    if segment_count != 31104:
        LOGGER.error(f"The number of segments is {segment_count}, but should be 31104 (number of verses in the Bible).")
    return output_filename


def escape_id(id: str) -> str:
    return escape(id).replace("\n", "&#xA;")


def strip_parens(term_str: str, left="(", right=")") -> str:
    parens: int = 0
    end: int = -1
    for i in reversed(range(len(term_str))):
        c = term_str[i]
        if c == right:
            if parens == 0:
                end = i + 1
            parens += 1
        elif c == left:
            parens -= 1
            if parens == 0:
                term_str = term_str[:i] + term_str[end:]
                end = -1
    return term_str


def clean_term(term_str: str) -> str:
    term_str = term_str.strip()
    term_str = strip_parens(term_str)
    return " ".join(term_str.split())


def extract_terms_list(list_type: str, project: Optional[str] = None) -> Dict[str, List[VerseRef]]:
    list_file_name = _TERMS_LISTS.get(list_type)
    if list_file_name is None:
        return {}

    list_name = list_type
    if project is not None:
        list_name = project

    dir = SIL_NLP_ENV.pt_terms_dir if project is None else SIL_NLP_ENV.pt_projects_dir / project
    terms_xml_path = dir / list_file_name

    terms_metadata_path = get_terms_metadata_path(list_name)
    terms_glosses_path = get_terms_glosses_path(list_name)
    terms_vrefs_path = get_terms_vrefs_path(list_name)

    references: Dict[str, List[VerseRef]] = {}
    with terms_metadata_path.open("w", encoding="utf-8", newline="\n") as terms_metadata_file, terms_glosses_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as terms_glosses_file, terms_vrefs_path.open("w", encoding="utf-8", newline="\n") as terms_vrefs_file:
        if os.path.exists(terms_xml_path):
            terms_tree = etree.parse(str(terms_xml_path))
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
                refs_elem = term_elem.find("References")
                refs_list: List[VerseRef] = []
                if refs_elem is not None:
                    for verse_elem in refs_elem.findall("Verse"):
                        bbbcccvvv = int(verse_elem.text[:9])
                        refs_list.append(VerseRef.from_bbbcccvvv(bbbcccvvv))
                    references[id] = refs_list
                glosses = _process_gloss_string(gloss_str)
                terms_metadata_file.write(f"{id}\t{cat}\t{domain}\n")
                terms_glosses_file.write("\t".join(glosses) + "\n")
                terms_vrefs_file.write("\t".join(str(vref) for vref in refs_list) + "\n")
    return references


def extract_major_terms_per_language(iso: str) -> None:
    # extract Biblical Terms for the langauage
    terms_xml_path = SIL_NLP_ENV.pt_terms_dir / f"BiblicalTerms{iso.capitalize()}.xml"
    terms_tree = etree.parse(str(terms_xml_path))

    # build glosses dict
    terms_dict = {}
    for term_elem in terms_tree.getroot().findall("Terms")[0].findall("Localization"):
        id = term_elem.get("Id")
        if id is None:
            continue
        terms_dict[escape_id(id)] = _process_gloss_string(term_elem.get("Gloss", ""))

    terms_glosses_path = get_terms_glosses_path(list_name="Major", iso=iso)

    with terms_glosses_path.open("w", encoding="utf-8", newline="\n") as terms_glosses_file:
        # import major metadata to line up terms to it
        with (SIL_NLP_ENV.assets_dir / "Major-metadata.txt").open("r", encoding="utf-8", newline="\n") as mm_file:
            major_metadata = mm_file.readlines()
        for line in major_metadata:
            id = line.split("\t")[0]
            if id in terms_dict:
                terms_glosses_file.write("\t".join(terms_dict[id]) + "\n")
            else:
                terms_glosses_file.write("\n")


def _process_gloss_string(gloss_str: str) -> List[str]:
    match = re.match(r"\[(.+?)\]", gloss_str)
    if match is not None:
        gloss_str = match.group(1)
    gloss_str = gloss_str.replace("?", "")
    gloss_str = clean_term(gloss_str)
    gloss_str = strip_parens(gloss_str, left="[", right="]")
    gloss_str = re.sub(r"\s+\d+(\.\d+)*$", "", gloss_str)
    glosses = re.split("[;,/]", gloss_str)
    glosses = unique_list([gloss.strip() for gloss in glosses if gloss.strip() != ""])
    return glosses


def extract_terms_list_from_renderings(project: str, renderings_tree: etree.ElementTree) -> None:
    terms_metadata_path = get_terms_metadata_path(project)
    with terms_metadata_path.open("w", encoding="utf-8", newline="\n") as terms_metadata_file:
        for rendering_elem in renderings_tree.getroot().findall("TermRendering"):
            id = rendering_elem.get("Id")
            if id is None:
                continue
            id = escape_id(id)
            if rendering_elem.get("Guess") != "false" or rendering_elem.findtext("Renderings", "") == "":
                continue

            terms_metadata_file.write(f"{id}\t?\t?\n")


def extract_term_renderings(project_folder: str, corpus_filename: Path) -> None:
    project_dir = get_project_dir(project_folder)
    renderings_path = project_dir / "TermRenderings.xml"
    if not renderings_path.is_file():
        return

    renderings_tree = etree.parse(str(renderings_path), parser=etree.XMLParser(encoding="utf-8"))
    rendering_elems: Dict[str, etree.Element] = {}
    for elem in renderings_tree.getroot().findall("TermRendering"):
        id = elem.get("Id")
        if id is None:
            continue
        id = escape_id(id)
        rendering_elems[id] = elem

    settings_tree = etree.parse(str(project_dir / "Settings.xml"))
    iso = get_iso(settings_tree)
    project_name = settings_tree.getroot().findtext("Name", project_folder)
    terms_setting = settings_tree.getroot().findtext("BiblicalTermsListSetting", "Major::BiblicalTerms.xml")

    list_type, terms_project, _ = terms_setting.split(":", maxsplit=3)
    list_name = list_type
    references: Dict[str, List[VerseRef]] = {}
    if list_type == "Project":
        if terms_project == project_name:
            references = extract_terms_list(list_type, project_folder)
        else:
            extract_terms_list_from_renderings(project_folder, renderings_tree)
        list_name = project_folder

    corpus: Dict[VerseRef, str] = {}
    if len(references) > 0:
        prev_verse_str = ""
        for ref_str, verse_str in zip(load_corpus(SIL_NLP_ENV.assets_dir / "vref.txt"), load_corpus(corpus_filename)):
            if verse_str == "<range>":
                verse_str = prev_verse_str
            corpus[VerseRef.from_string(ref_str)] = verse_str
            prev_verse_str = verse_str

    terms_metadata_path = get_terms_metadata_path(list_name)
    terms_renderings_path = SIL_NLP_ENV.mt_terms_dir / f"{iso}-{project_folder}-{list_type}-renderings.txt"
    count = 0
    with terms_renderings_path.open("w", encoding="utf-8", newline="\n") as terms_renderings_file:
        for line in load_corpus(terms_metadata_path):
            id, _, _ = line.split("\t", maxsplit=3)
            rendering_elem = rendering_elems.get(id)
            refs_list = references.get(id, [])

            renderings: Set[str] = set()
            if rendering_elem is not None and rendering_elem.get("Guess", "false") == "false":
                renderings_str = rendering_elem.findtext("Renderings", "")
                if renderings_str != "":
                    for rendering in renderings_str.strip().split("||"):
                        rendering = clean_term(rendering).strip()
                        if len(refs_list) > 0 and "*" in rendering:
                            regex = (
                                re.escape(rendering).replace("\\ \\*\\*\\ ", "(?:\\ \\w+)*\\ ").replace("\\*", "\\w*")
                            )
                            for ref in refs_list:
                                verse_str = corpus.get(ref, "")
                                for match in re.finditer(regex, verse_str):
                                    surface_form = match.group()
                                    renderings.add(surface_form)

                        else:
                            rendering = rendering.replace("*", "").strip()
                            if rendering != "":
                                renderings.add(rendering)
            terms_renderings_file.write("\t".join(renderings) + "\n")
            if len(renderings) > 0:
                count += 1
    if count == 0:
        terms_renderings_path.unlink()
        if list_type == "Project":
            terms_metadata_path.unlink()
            terms_glosses_path = get_terms_glosses_path(list_name)
            if terms_glosses_path.is_file():
                terms_glosses_path.unlink()
    LOGGER.info(f"# of Terms: {count}")


def book_file_name_digits(book_num: int) -> str:
    if book_num < 10:
        return f"0{book_num}"
    if book_num < 40:
        return str(book_num)
    if book_num < 100:
        return str(book_num + 1)
    if book_num < 110:
        return f"A{book_num - 100}"
    if book_num < 120:
        return f"B{book_num - 110}"
    return f"C{book_num - 120}"


def get_book_path(project: str, book: str) -> Path:
    project_dir = get_project_dir(project)
    settings_tree = etree.parse(os.path.join(project_dir, "Settings.xml"))
    naming_elem = settings_tree.find("Naming")
    assert naming_elem is not None

    pre_part = naming_elem.get("PrePart", "")
    post_part = naming_elem.get("PostPart", "")
    book_name_form = naming_elem.get("BookNameForm")
    assert book_name_form is not None

    book_num = book_id_to_number(book)
    if book_name_form == "MAT":
        book_name = book
    elif book_name_form == "40" or book_name_form == "41":
        book_name = book_file_name_digits(book_num)
    else:
        book_name = f"{book_file_name_digits(book_num)}{book}"

    book_file_name = f"{pre_part}{book_name}{post_part}"

    return SIL_NLP_ENV.pt_projects_dir / project / book_file_name
