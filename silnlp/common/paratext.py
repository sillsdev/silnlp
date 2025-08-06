import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO, Tuple
from xml.sax.saxutils import escape

import regex as re
from lxml import etree
from machine.corpora import (
    DictionaryTextCorpus,
    FileParatextProjectSettingsParser,
    MemoryText,
    ParatextTextCorpus,
    Text,
    TextCorpus,
    TextRow,
    UsfmFileTextCorpus,
    create_versification_ref_corpus,
    extract_scripture_corpus,
)
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, VersificationType, book_id_to_number, get_books
from machine.tokenization import WhitespaceTokenizer

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


def get_iso(project_dir: Path) -> str:
    return FileParatextProjectSettingsParser(project_dir).parse().language_code


def extract_project(
    project_dir: Path,
    output_dir: Path,
    include_books: List[str] = [],
    exclude_books: List[str] = [],
    include_markers: bool = False,
    extract_lemmas: bool = False,
    output_project_vrefs: bool = False,
) -> Tuple[Path, int]:
    iso = get_iso(project_dir)

    ref_corpus: TextCorpus = create_versification_ref_corpus()

    ltg_dir = project_dir / "LTG"
    if extract_lemmas and ltg_dir.is_dir():
        project_corpus = get_lemma_text_corpus(project_dir)
    else:
        project_corpus = ParatextTextCorpus(project_dir, include_markers=include_markers)

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

        ref_corpus = ref_corpus.filter_texts(filter_corpus)
        project_corpus = project_corpus.filter_texts(filter_corpus)

    if include_markers:
        output_basename += "-m"
    elif extract_lemmas and ltg_dir.is_dir():
        output_basename += "-lemmas"
    output_filename = output_dir / f"{output_basename}.txt"
    output_vref_filename = output_dir / f"{output_basename}.vref.txt"

    try:
        segment_count = 0
        with ExitStack() as stack:
            output_stream = stack.enter_context(output_filename.open("w", encoding="utf-8", newline="\n"))
            output = stack.enter_context(extract_scripture_corpus(project_corpus, ref_corpus))
            output_vref_stream: Optional[TextIO] = None
            if output_project_vrefs:
                output_vref_stream = stack.enter_context(output_vref_filename.open("w", encoding="utf-8", newline="\n"))

            for line, _, project_vref in output:
                output_stream.write(line + "\n")
                if output_vref_stream is not None:
                    output_vref_stream.write(("" if project_vref is None else str(project_vref)) + "\n")
                segment_count += 1
        return output_filename, segment_count
    except Exception:
        if output_filename.is_file():
            output_filename.unlink()
        if output_vref_filename.is_file():
            output_vref_filename.unlink()
        raise


def get_lemma_text_corpus(project_dir: Path) -> TextCorpus:
    tokenizer = WhitespaceTokenizer()
    pt_corpus = ParatextTextCorpus(project_dir)
    lemma_corpus: TextCorpus = UsfmFileTextCorpus(
        project_dir / "LTG", versification=pt_corpus.versification, file_pattern="*.LTG"
    )
    surface_corpus = pt_corpus.tokenize(tokenizer)
    lemma_corpus = lemma_corpus.tokenize(tokenizer)
    new_texts: List[Text] = []
    for surface_text, lemma_text in zip(surface_corpus.texts, lemma_corpus.texts):
        new_rows: List[TextRow] = []
        with surface_text.get_rows() as surface_rows, lemma_text.get_rows() as lemma_rows:
            for surface_row, lemma_row in zip(surface_rows, lemma_rows):
                if len(surface_row.segment) != len(lemma_row.segment) or surface_row.ref != lemma_row.ref:
                    raise RuntimeError("The lemma file is invalid.")
                lemmas: List[str] = []
                for surface, lemma in zip(surface_row.segment, lemma_row.segment):
                    lemma = lemma.split("|")[0]
                    lemma = strip_morph_info(lemma)
                    match = _NON_LETTER_PATTERN.fullmatch(surface)
                    if match is not None:
                        lemma = match.group(1) + lemma + match.group(2)
                    lemmas.append(lemma)
                lemmas_text = " ".join(lemmas)
                new_rows.append(
                    TextRow(
                        surface_text.id,
                        surface_row.ref,
                        [] if len(lemmas_text) == 0 else [lemmas_text],
                        surface_row.flags,
                    )
                )
        new_texts.append(MemoryText(surface_text.id, new_rows))
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
            if parens > 0:
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
    terms_glosses_path = get_terms_glosses_path(
        list_name,
        mt_terms_dir=output_dir,
        iso=get_iso(project_dir) if project_dir is not None and list_type != "Project" else "en",
    )
    terms_vrefs_path = get_terms_vrefs_path(list_name, mt_terms_dir=output_dir)

    references: Dict[str, List[VerseRef]] = {}
    with (
        terms_metadata_path.open("w", encoding="utf-8", newline="\n") as terms_metadata_file,
        terms_glosses_path.open("w", encoding="utf-8", newline="\n") as terms_glosses_file,
        terms_vrefs_path.open("w", encoding="utf-8", newline="\n") as terms_vrefs_file,
    ):
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
                        if verse_elem.text is None:
                            continue
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


def extract_terms_list_from_renderings(project: str, renderings_tree: etree._ElementTree, output_dir: Path) -> None:
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

    settings = FileParatextProjectSettingsParser(project_dir).parse()
    list_type = settings.biblical_terms_list_type
    list_name = list_type
    references: Dict[str, List[VerseRef]] = {}
    if list_type == "Project":
        if settings.biblical_terms_project_name == settings.name:
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
    terms_renderings_path = output_dir / f"{settings.language_code}-{project_dir.name}-{list_type}-renderings.txt"
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
            terms_glosses_path = get_terms_glosses_path(list_name, mt_terms_dir=output_dir, iso=settings.language_code)
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
    settings = FileParatextProjectSettingsParser(project_dir).parse()
    book_file_name = settings.get_book_file_name(book)

    return SIL_NLP_ENV.pt_projects_dir / project / book_file_name


def get_last_verse(project_dir: str, book: str, chapter: int) -> int:
    last_verse = "0"
    book_path = get_book_path(project_dir, book)
    try:
        with book_path.open("r", encoding="utf-8-sig", newline="\n", errors="ignore") as book_file:
            in_chapter = False
            for line in book_file:
                chapter_marker = re.search(r"\\c ? ?([0-9]+).*", line)
                if chapter_marker:
                    if chapter_marker.group(1) == str(chapter):
                        in_chapter = True
                    else:
                        in_chapter = False

                verse_marker = re.search(r"\\v ? ?([0-9]+).*", line)
                if verse_marker:
                    if in_chapter:
                        last_verse = verse_marker.group(1)
    except OSError as e:
        LOGGER.warning(f"Unable to open {book_path}: {e}")
    return int(last_verse)


# OT versification detection algorithm from:
# https://github.com/BibleNLP/ebible/blob/main/code/notebooks/eBible%20-%20Extract%20projects.ipynb
def detect_OT_versification(project_dir: str) -> Tuple[VersificationType, List[str]]:
    dan_3 = get_last_verse(project_dir, "DAN", 3)
    dan_5 = get_last_verse(project_dir, "DAN", 5)
    dan_13 = get_last_verse(project_dir, "DAN", 13)

    key_last_verses = []

    if dan_3 == 30:
        versification = VersificationType.ENGLISH
        key_last_verses.append("Daniel 3:" + str(dan_3))
    elif dan_3 == 33 and dan_5 == 30:
        versification = VersificationType.ORIGINAL
        key_last_verses.append("Daniel 3:" + str(dan_3))
        key_last_verses.append("Daniel 5:" + str(dan_5))
    elif dan_3 == 33 and dan_5 == 31:
        versification = VersificationType.RUSSIAN_PROTESTANT
        key_last_verses.append("Daniel 3:" + str(dan_3))
        key_last_verses.append("Daniel 5:" + str(dan_5))
    elif dan_3 == 97:
        versification = VersificationType.SEPTUAGINT
        key_last_verses.append("Daniel 3:" + str(dan_3))
    elif dan_3 == 100:
        if dan_13 == 65:
            versification = VersificationType.VULGATE
        else:
            versification = VersificationType.RUSSIAN_ORTHODOX
        key_last_verses.append("Daniel 3:" + str(dan_3))
        key_last_verses.append("Daniel 13:" + str(dan_13))
    else:
        versification = VersificationType.UNKNOWN

    return versification, key_last_verses


# NT versification detection algorithm from:
# https://github.com/BibleNLP/ebible/blob/main/code/notebooks/eBible%20-%20Extract%20projects.ipynb
def detect_NT_versification(project_dir: str) -> Tuple[List[VersificationType], List[str]]:
    jhn_6 = get_last_verse(project_dir, "JHN", 6)
    act_19 = get_last_verse(project_dir, "ACT", 19)
    rom_16 = get_last_verse(project_dir, "ROM", 16)

    key_last_verses = []

    if jhn_6 == 72:
        versification = [VersificationType.VULGATE]
        key_last_verses.append("John 6:" + str(jhn_6))
    elif act_19 == 41:
        versification = [VersificationType.ENGLISH]
        key_last_verses.append("Acts 19:" + str(act_19))
    elif rom_16 == 24:
        versification = [VersificationType.RUSSIAN_PROTESTANT, VersificationType.RUSSIAN_ORTHODOX]
        key_last_verses.append("Romans 16:" + str(rom_16))
    elif jhn_6 == 71 and act_19 == 40:
        versification = [VersificationType.ORIGINAL, VersificationType.SEPTUAGINT]
        key_last_verses.append("John 6:" + str(jhn_6))
        key_last_verses.append("Acts 19:" + str(act_19))
    else:
        versification = [VersificationType.UNKNOWN]

    return versification, key_last_verses


def check_versification(project_dir: str) -> Tuple[bool, List[VersificationType]]:
    settings = FileParatextProjectSettingsParser(project_dir).parse()

    check_ot, check_nt, matching = False, False, False

    dan_book_path = get_book_path(project_dir, "DAN")
    check_ot = bool(dan_book_path.is_file())

    jhn_book_path = get_book_path(project_dir, "JHN")
    act_book_path = get_book_path(project_dir, "ACT")
    rom_book_path = get_book_path(project_dir, "ROM")
    check_nt = bool(jhn_book_path.is_file() and act_book_path.is_file() and rom_book_path.is_file())

    if check_ot:
        ot_versification: VersificationType
        ot_versification, key_ot_verses = detect_OT_versification(project_dir)
        if ot_versification == VersificationType.UNKNOWN:
            LOGGER.warning(f"Unknown versification detected for {project_dir}.")
            return (matching, [ot_versification])
    if check_nt:
        nt_versification: List[VersificationType]
        nt_versification, key_nt_verses = detect_NT_versification(project_dir)
        if nt_versification[0] == VersificationType.UNKNOWN:
            LOGGER.warning(f"Unknown versification detected for {project_dir}.")
            return (matching, nt_versification)

    detected_versification: List[VersificationType] = [VersificationType.UNKNOWN]
    if check_ot and check_nt:
        if ot_versification not in nt_versification:
            LOGGER.warning(
                f"The detected OT versification {ot_versification} and the detected NT versification(s) "
                f"{', '.join([str(int(versification)) for versification in nt_versification])} do not match. "
                f"The detected versifications were based on {', '.join(key_ot_verses + key_nt_verses)} "
                "being the last verse of their respective chapters."
            )
            return (matching, [ot_versification] + nt_versification)
        detected_versification = [ot_versification]
        key_verses = key_ot_verses + key_nt_verses
    elif not check_ot and check_nt:
        detected_versification = nt_versification
        key_verses = key_nt_verses
    elif check_ot and not check_nt:
        detected_versification = [ot_versification]
        key_verses = key_ot_verses
    else:
        LOGGER.warning(
            f"Insufficient information to detect versification for {project_dir}. "
            "Versification detection for the OT requires the book of Daniel. "
            "Versification detection for the NT requires the books of John, Acts, and Romans."
        )
        return (matching, detected_versification)

    if settings.versification.type not in detected_versification:
        if not (
            settings.versification.type == VersificationType.UNKNOWN
            and settings.versification.base_versification.type in detected_versification
        ):
            LOGGER.warning(
                f"Project versification setting {settings.versification.type} does not match detected versification(s) "
                f"{', '.join([str(int(versification)) for versification in detected_versification])}. "
                f"The detected versification(s) were based on {', '.join(key_verses)} "
                f"being the last verse of {'their' if len(key_verses)>=2 else 'its'} "
                f"respective chapter{'s' if len(key_verses)>=2 else ''}."
            )
            return (matching, detected_versification)

    matching = True
    return (matching, detected_versification)
