import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple
from xml.sax.saxutils import escape

import regex as re
from lxml import etree
from machine.corpora import (
    DictionaryTextCorpus,
    FilteredTextCorpus,
    MemoryText,
    ParallelTextCorpus,
    ParatextTextCorpus,
    Text,
    TextCorpus,
    TextSegment,
    UsfmFileTextCorpus,
)
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, book_id_to_number, get_books
from machine.tokenization import NullTokenizer, WhitespaceTokenizer

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

_MORPH_INFO_PATTERN = re.compile(r"<[^>]+>")

_NON_LETTER_PATTERN = re.compile(r"([^\p{L}\p{M}]*)[\p{L}\p{M}]+([^\p{L}\p{M}]*)")

LOGGER = logging.getLogger(__name__)


def get_project_dir(project: str) -> Path:
    return SIL_NLP_ENV.pt_projects_dir / project


def get_iso(settings_tree: etree.ElementTree) -> str:
    iso = settings_tree.getroot().findtext("LanguageIsoCode")
    assert iso is not None
    index = iso.index(":")
    return iso[:index]


def extract_project(
    project_dir: Path,
    output_dir: Path,
    include_books: List[str] = [],
    exclude_books: List[str] = [],
    include_markers: bool = False,
    extract_lemmas: bool = False,
    output_project_vrefs: bool = False,
) -> Tuple[Path, int]:
    settings_tree = parse_project_settings(project_dir)
    iso = get_iso(settings_tree)

    ref_dir = SIL_NLP_ENV.assets_dir / "Ref"

    tokenizer = NullTokenizer()
    ref_corpus = ParatextTextCorpus(tokenizer, ref_dir)

    ltg_dir = project_dir / "LTG"
    if extract_lemmas and ltg_dir.is_dir():
        project_corpus = get_lemma_text_corpus(project_dir)
    else:
        project_corpus = ParatextTextCorpus(tokenizer, project_dir, include_markers=include_markers)

    output_basename = f"{iso}-{project_dir.name}"
    if len(include_books) > 0 or len(exclude_books) > 0:
        output_basename += "_"
        include_books_set: Optional[Set[int]] = None
        if len(include_books) > 0:
            include_books_set = get_books(include_books)
            for text in include_books:
                output_basename += f"+{text}"
        exclude_books_set: Optional[Set[int]] = None
        if len(exclude_books) > 0:
            exclude_books_set = get_books(exclude_books)
            for text in exclude_books:
                output_basename += f"-{text}"

        def filter_corpus(text: Text) -> bool:
            book_num = book_id_to_number(text.id)
            if exclude_books_set is not None and book_num in exclude_books_set:
                return False

            if include_books_set is not None and book_num in include_books_set:
                return True

            return include_books_set is None

        ref_corpus = FilteredTextCorpus(ref_corpus, filter_corpus)
        project_corpus = FilteredTextCorpus(project_corpus, filter_corpus)

    if include_markers:
        output_basename += "-m"
    elif extract_lemmas and ltg_dir.is_dir():
        output_basename += "-lemmas"
    output_filename = output_dir / f"{output_basename}.txt"
    output_vref_filename = output_dir / f"{output_basename}.vref.txt"

    try:
        parallel_corpus = ParallelTextCorpus(ref_corpus, project_corpus)
        segment_count = 0
        with ExitStack() as stack:
            output_stream = stack.enter_context(output_filename.open("w", encoding="utf-8", newline="\n"))
            segments = stack.enter_context(parallel_corpus.get_segments(all_source_segments=True))
            output_vref_stream: Optional[TextIO] = None
            if output_project_vrefs:
                output_vref_stream = stack.enter_context(output_vref_filename.open("w", encoding="utf-8", newline="\n"))

            cur_ref: Optional[VerseRef] = None
            cur_trg_ref: Optional[VerseRef] = None
            cur_target_line = ""
            cur_target_line_range = True
            for segment in segments:
                ref: VerseRef = segment.segment_ref
                if cur_ref is not None and ref.compare_to(cur_ref, compare_segments=False) != 0:
                    output_stream.write(("<range>" if cur_target_line_range else cur_target_line) + "\n")
                    if output_vref_stream is not None:
                        output_vref_stream.write(("" if cur_trg_ref is None else str(cur_trg_ref)) + "\n")
                    segment_count += 1
                    cur_target_line = ""
                    cur_target_line_range = True
                    cur_trg_ref = None

                cur_ref = ref
                if cur_trg_ref is None:
                    cur_trg_ref = segment.target_segment_ref
                elif segment.target_segment_ref is not None and cur_trg_ref != segment.target_segment_ref:
                    cur_trg_ref.simplify()
                    if cur_trg_ref < segment.target_segment_ref:
                        start_ref = cur_trg_ref
                        end_ref = segment.target_segment_ref
                    else:
                        start_ref = segment.target_segment_ref
                        end_ref = cur_trg_ref
                    cur_trg_ref = VerseRef.from_range(start_ref, end_ref)
                if not segment.is_target_in_range or segment.is_target_range_start or len(segment.target_segment) > 0:
                    if len(segment.target_segment) > 0:
                        if len(cur_target_line) > 0:
                            cur_target_line += " "
                        cur_target_line += segment.target_segment[0]
                    cur_target_line_range = False
            output_stream.write(("<range>" if cur_target_line_range else cur_target_line) + "\n")
            if output_vref_stream is not None:
                output_vref_stream.write(("" if cur_trg_ref is None else str(cur_trg_ref)) + "\n")
            segment_count += 1
        return output_filename, segment_count
    except:
        if output_filename.is_file():
            output_filename.unlink()
        if output_vref_filename.is_file():
            output_vref_filename.unlink()
        raise


def get_lemma_text_corpus(project_dir: Path) -> TextCorpus:
    tokenizer = WhitespaceTokenizer()
    surface_corpus = ParatextTextCorpus(tokenizer, project_dir)
    lemma_corpus = UsfmFileTextCorpus(
        tokenizer, "usfm.sty", "utf-8-sig", project_dir / "LTG", surface_corpus.versification, glob_pattern="*.LTG"
    )
    parallel_corpus = ParallelTextCorpus(surface_corpus, lemma_corpus)
    new_texts: List[Text] = []
    for text in parallel_corpus.texts:
        new_segments: List[TextSegment] = []
        with text.segments as segments:
            for segment in segments:
                if len(segment.source_segment) != len(segment.target_segment):
                    raise RuntimeError("The lemma file is invalid.")
                lemmas: List[str] = []
                for surface, lemma in zip(segment.source_segment, segment.target_segment):
                    lemma = lemma.split("|")[0]
                    lemma = strip_morph_info(lemma)
                    match = _NON_LETTER_PATTERN.fullmatch(surface)
                    if match is not None:
                        lemma = match.group(1) + lemma + match.group(2)
                    lemmas.append(lemma)
                lemmas_text = " ".join(lemmas)
                new_segments.append(
                    TextSegment(
                        text.id,
                        segment.segment_ref,
                        [] if len(lemmas_text) == 0 else [lemmas_text],
                        segment.is_source_sentence_start,
                        segment.is_source_in_range,
                        segment.is_source_range_start,
                        len(lemmas_text) == 0,
                    )
                )
        new_texts.append(MemoryText(text.id, new_segments))
    return DictionaryTextCorpus(new_texts)


def strip_morph_info(lemma: str) -> str:
    lemma = lemma[1:]
    lemma = lemma.replace("+", "")
    lemma = _MORPH_INFO_PATTERN.sub("", lemma)
    return lemma


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


def extract_terms_list(
    list_type: str, output_dir: Path, project_dir: Optional[Path] = None
) -> Dict[str, List[VerseRef]]:
    list_file_name = _TERMS_LISTS.get(list_type)
    if list_file_name is None:
        return {}

    list_name = list_type
    if project_dir is not None:
        list_name = project_dir.name

    dir = SIL_NLP_ENV.pt_terms_dir if project_dir is None else project_dir
    terms_xml_path = dir / list_file_name

    terms_metadata_path = get_terms_metadata_path(list_name, mt_terms_dir=output_dir)
    terms_glosses_path = get_terms_glosses_path(list_name, mt_terms_dir=output_dir)
    terms_vrefs_path = get_terms_vrefs_path(list_name, mt_terms_dir=output_dir)

    references: Dict[str, List[VerseRef]] = {}
    with terms_metadata_path.open("w", encoding="utf-8", newline="\n") as terms_metadata_file, terms_glosses_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as terms_glosses_file, terms_vrefs_path.open("w", encoding="utf-8", newline="\n") as terms_vrefs_file:
        if os.path.exists(terms_xml_path):
            with terms_xml_path.open("rb") as terms_file:
                terms_tree = etree.parse(terms_file)
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
                        vref = VerseRef.from_bbbcccvvv(bbbcccvvv)
                        vref.change_versification(ORIGINAL_VERSIFICATION)
                        refs_list.append(vref)
                    references[id] = refs_list
                glosses = _process_gloss_string(gloss_str)
                terms_metadata_file.write(f"{id}\t{cat}\t{domain}\n")
                terms_glosses_file.write("\t".join(glosses) + "\n")
                terms_vrefs_file.write("\t".join(str(vref) for vref in refs_list) + "\n")
    return references


def extract_major_terms_per_language(iso: str) -> None:
    # extract Biblical Terms for the langauage
    terms_xml_path = SIL_NLP_ENV.pt_terms_dir / f"BiblicalTerms{iso.capitalize()}.xml"
    with terms_xml_path.open("rb") as terms_file:
        terms_tree = etree.parse(terms_file)

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


def extract_terms_list_from_renderings(project: str, renderings_tree: etree.ElementTree, output_dir: Path) -> None:
    terms_metadata_path = get_terms_metadata_path(project, mt_terms_dir=output_dir)
    with terms_metadata_path.open("w", encoding="utf-8", newline="\n") as terms_metadata_file:
        for rendering_elem in renderings_tree.getroot().findall("TermRendering"):
            id = rendering_elem.get("Id")
            if id is None:
                continue
            id = escape_id(id)
            if rendering_elem.get("Guess") != "false" or rendering_elem.findtext("Renderings", "") == "":
                continue

            terms_metadata_file.write(f"{id}\t?\t?\n")


def extract_term_renderings(project_dir: Path, corpus_filename: Path, output_dir: Path) -> int:
    renderings_path = project_dir / "TermRenderings.xml"
    if not renderings_path.is_file():
        return 0

    try:
        with renderings_path.open("rb") as renderings_file:
            renderings_tree = etree.parse(renderings_file)
    except etree.XMLSyntaxError:
        # Try forcing the encoding to UTF-8 during parsing
        with renderings_path.open("rb") as renderings_file:
            renderings_tree = etree.parse(renderings_file, parser=etree.XMLParser(encoding="utf-8"))
    rendering_elems: Dict[str, etree.Element] = {}
    for elem in renderings_tree.getroot().findall("TermRendering"):
        id = elem.get("Id")
        if id is None:
            continue
        id = escape_id(id)
        rendering_elems[id] = elem

    settings_tree = parse_project_settings(project_dir)
    iso = get_iso(settings_tree)
    project_name = settings_tree.getroot().findtext("Name", project_dir.name)
    terms_setting = settings_tree.getroot().findtext("BiblicalTermsListSetting", "Major::BiblicalTerms.xml")

    list_type, terms_project, _ = terms_setting.split(":", maxsplit=3)
    list_name = list_type
    references: Dict[str, List[VerseRef]] = {}
    if list_type == "Project":
        if terms_project == project_name:
            references = extract_terms_list(list_type, output_dir, project_dir)
        else:
            extract_terms_list_from_renderings(project_dir.name, renderings_tree, output_dir)
        list_name = project_dir.name

    corpus: Dict[VerseRef, str] = {}
    if len(references) > 0:
        prev_verse_str = ""
        for ref_str, verse_str in zip(load_corpus(SIL_NLP_ENV.assets_dir / "vref.txt"), load_corpus(corpus_filename)):
            if verse_str == "<range>":
                verse_str = prev_verse_str
            corpus[VerseRef.from_string(ref_str, ORIGINAL_VERSIFICATION)] = verse_str
            prev_verse_str = verse_str

    terms_metadata_path = get_terms_metadata_path(list_name, mt_terms_dir=output_dir)
    terms_renderings_path = output_dir / f"{iso}-{project_dir.name}-{list_type}-renderings.txt"
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
            terms_glosses_path = get_terms_glosses_path(list_name, mt_terms_dir=output_dir)
            if terms_glosses_path.is_file():
                terms_glosses_path.unlink()
    return count


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
    settings_tree = parse_project_settings(project_dir)
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


def parse_project_settings(project_dir: Path) -> Any:
    settings_filename = project_dir / "Settings.xml"
    if not settings_filename.is_file():
        settings_filename = next(project_dir.glob("*.ssf"), Path())
    if not settings_filename.is_file():
        raise RuntimeError("The project directory does not contain a settings file.")

    with settings_filename.open("rb") as settings_file:
        settings_tree = etree.parse(settings_file)
    return settings_tree
