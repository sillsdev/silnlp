import argparse
import os
from pathlib import Path
import re
from ..common.environment import SIL_NLP_ENV

BIBLE = {
    "GEN": 50,
    "EXO": 40,
    "LEV": 27,
    "NUM": 36,
    "DEU": 34,
    "JOS": 24,
    "JDG": 21,
    "RUT": 4,
    "1SA": 31,
    "2SA": 24,
    "1KI": 22,
    "2KI": 25,
    "1CH": 29,
    "2CH": 36,
    "EZR": 10,
    "NEH": 13,
    "EST": 10,
    "JOB": 42,
    "PSA": 150,
    "PRO": 31,
    "ECC": 12,
    "SNG": 8,
    "ISA": 66,
    "JER": 52,
    "LAM": 5,
    "EZK": 48,
    "DAN": 12,
    "HOS": 14,
    "JOL": 4,
    "AMO": 9,
    "OBA": 1,
    "JON": 4,
    "MIC": 7,
    "NAM": 3,
    "HAB": 3,
    "ZEP": 3,
    "HAG": 2,
    "ZEC": 14,
    "MAL": 3,
    "MAT": 28,
    "MRK": 16,
    "LUK": 24,
    "JHN": 21,
    "ACT": 28,
    "ROM": 16,
    "1CO": 16,
    "2CO": 13,
    "GAL": 6,
    "EPH": 6,
    "PHP": 4,
    "COL": 4,
    "1TH": 5,
    "2TH": 3,
    "1TI": 6,
    "2TI": 4,
    "TIT": 3,
    "PHM": 1,
    "HEB": 13,
    "JAS": 5,
    "1PE": 5,
    "2PE": 3,
    "1JN": 5,
    "2JN": 1,
    "3JN": 1,
    "JUD": 1,
    "REV": 22,
}

OT = dict(list(BIBLE.items())[0:39])
NT = dict(list(BIBLE.items())[39:])

def get_sfm_files(project_dir):
    return [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]]

def get_vrefs():
    vref_file = SIL_NLP_ENV.assets_dir / "vref.txt"
    with open(vref_file) as vref_f:
        return [line.strip() for line in vref_f.readlines()]


def get_vref_dict() -> dict:
    vref_dict = {}
    vrefs = get_vrefs()
    for vref in vrefs:
        book, chapter, verse = parse_vref(vref)
        if book not in vref_dict:
            vref_dict[book] = {}
        if chapter not in vref_dict[book]:
            vref_dict[book][chapter] = []

        vref_dict[book][chapter].append(verse)
    return vref_dict


def parse_vref(vref):

    vref_pattern = r"(?P<book>\w+)\s+(?P<chapter>\d+):(?P<verse>\d+)"
    match = re.match(vref_pattern, vref)

    if match:
        book = match.group("book")
        chapter = int(match.group("chapter"))
        verse = int(match.group("verse"))
        return book, chapter, verse
    else:
        raise ValueError(f"Invalid vref format: {vref}")


def get_max_chapters() -> dict:
    vrefs = get_vrefs()
    max_chapters = {}
    for vref in vrefs:
        book, chapter, _ = parse_vref(vref)
        max_chapters[book] = chapter
    return max_chapters


def get_max_chapter(book) -> dict:
    vref_dict = get_vref_dict()
    return max(vref_dict[book])


def get_max_verse(book, chapter) -> int:
    vref_dict = get_vref_dict()
    return max(vref_dict[book][chapter])


def get_files_from_folder(folder, ext):
    files = [file for file in folder.glob(f"*{ext}") if file.is_file]
    return files

def parse_extract(file):

    with open(file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    
    usfm_data = dict()
    vrefs = get_vrefs()
    if not len(lines) == len(vrefs):
        raise RuntimeError(f"There are {len(lines)} and {len(vrefs)} verse references. These should match.")
    
    #print(f"There are {len(lines)} lines and {len(vrefs)} verse references.")
    
    verses_in_range = 0
    for line_no, line in enumerate(lines):
        
        # while line == '<range>':
        #     verses_in_range += 1
        #     continue

        if line == '':
            continue
            
        vref = vrefs[line_no]
        #print(vref, line)
        
        book, chapter, verse = parse_vref(vref)
        if book not in usfm_data:
            usfm_data[book] = {}
        if chapter not in usfm_data[book]:
            usfm_data[book][chapter] = {}
        if verse not in usfm_data[book][chapter]:
            usfm_data[book][chapter][verse] = line.strip()
            #print(f"{line_no}  Added {book} {chapter}:{verse}  {line} ")

    if not book:
        print(f"Warning: Could not parse book ID from {file}. ")
        return None

    return usfm_data


def save_to_usfm(book_id, usfm_data, output_file):

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"\\id {book_id}\n")
        for chapter in sorted(usfm_data[book_id], key=int):
            f.write(f"\\c {chapter}\n")
            for verse in sorted(usfm_data[book_id][chapter], key=int):
                f.write(f"\\v {verse} {usfm_data[book_id][chapter][verse]}\n")
    return True


def write_settings_file(language_name, iso_code, sfm_folder):
    settings = f'<ScriptureText>\n  <Language>{language_name}</Language>\n  <Encoding>65001</Encoding>\n  <LanguageIsoCode>{iso_code}:::</LanguageIsoCode>\n  <Versification>4</Versification>\n  <Naming PrePart="" PostPart=".sfm" BookNameForm="41MAT" />\n</ScriptureText>'
    settings_file = sfm_folder / "Settings.xml"

    if settings_file.is_file():
        # print(f"Settings file {settings_file} already exists.")
        return False
    else:
        with open(settings_file, "w", encoding="utf-8") as f_out:
            f_out.write(settings)
        print(f"Wrote Settings.xml file: {settings_file}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Convert vref files to Paratext projects")
    parser.add_argument(
        "--file",
        type=Path,
        help="The vref file to convert.",
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        help="The vref file to convert.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        help="The path to the Paratext projects folder. The project folder will be created inside this folder.",
    )

    parser.add_argument(
        "--language",
        type=str,
        help="The language name - for the Settings.xml file both language name and iso_code are required.",
    )
    parser.add_argument(
        "--iso",
        type=str,
        help="The iso code for the language. https://en.wikipedia.org/wiki/ISO_639-3/",
    )

    args = parser.parse_args()

    if args.file:
        input_files = [Path(args.file)]

    elif args.input_folder:
        input_folder = Path(args.input_folder)
        input_files = [file for file in input_folder.glob("*.txt")]

    for input_file in input_files:
        if 'OpenBible' in input_file.name:
            # ben-OpenBible_Bengali_Latn
            project_name_pattern = r"(?P<iso>.\w{2,3})-OpenBible_(?P<Language>.+)"
            match = re.match(project_name_pattern, input_file.stem)
            iso = match.group('iso')
            language = match.group('Language')
            project_folder_name = f"OpenBible_{language}"
        else:
            project_name_pattern = r"(?P<iso>.\w{2,3})-(?P<project>.+)"
            match = re.match(project_name_pattern, input_file.stem)
            project_folder_name = match.group('project')
            iso = args.iso if args.iso else match.group('iso')
            language = args.language

        project_folder = Path(args.output_folder) / project_folder_name
        if not project_folder.is_dir():
            print(f"Creating folder for output: {project_folder}")
            os.makedirs(project_folder, exist_ok=True)

        usfm_data = parse_extract(input_file)
        count = 0 
        for book in usfm_data:
            book_number = list(BIBLE.keys()).index(book) + 1
            
            sfm_file = project_folder / f"{book_number:02}{book}.sfm"
            if save_to_usfm(book, usfm_data, sfm_file):
                count += 1
        
        print(f"Saved {count} sfm files to {sfm_file.parent}")

        write_settings_file(language, iso, project_folder)
            
        
        #    print(f"Specify a Language in order to write the Settings.xml file.")


if __name__ == "__main__":
    main()
