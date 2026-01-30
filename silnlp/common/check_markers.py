import argparse
import logging
import shutil
import time
from pathlib import Path

from machine.corpora import FileParatextProjectSettingsParser, UsfmStylesheet, UsfmToken, UsfmTokenizer, UsfmTokenType
import machine.corpora.usfm_stylesheet as uss
print(uss.__file__)

from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
from .paratext import get_project_dir
from .split_verses_v3 import get_books_to_process

def check_end_markers(tokens, filepath, stylesheet):
    "Check that all CHARACTER/NOTE markers are properly closed before PARAGRAPH/CHAPTER/VERSE"
    open_markers = []
    for tok in tokens:
        if tok.type in [UsfmTokenType.CHARACTER, UsfmTokenType.NOTE]:
            tag = stylesheet.get_tag(tok.marker)
            if tag.end_marker: open_markers.append((tok.marker, tok.line_number, tag.end_marker))
        elif tok.type == UsfmTokenType.END:
            if not open_markers:
                print(f"{filepath}:{tok.line_number}: Unexpected end marker \\{tok.marker} with no matching opening")
                exit(1)
            expected_marker = open_markers[-1][0]
            if tok.marker != open_markers[-1][2]:
                print(f"{filepath}:{tok.line_number}: End marker \\{tok.marker} doesn't match opening \\{expected_marker}")
                exit(1)
            open_markers.pop()
        elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE]:
            if open_markers:
                marker,line,end = open_markers[-1]
                print(f"{filepath}:{line}: Missing end marker \\{end} for \\{marker}")
                exit(1)
    if open_markers:
        marker,line,end = open_markers[-1]
        print(f"{filepath}:{line}: Missing end marker \\{end} for \\{marker} at end of file")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="Split long paragraphs in USFM files")
    parser.add_argument("project", help="Paratext project name")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    )

    args = parser.parse_args()
    print(args)

    # Get project directory
    project_dir = Path(args.project)
    settings_file = project_dir / "Settings.xml"
    if project_dir.is_dir() and settings_file.is_file():
        print(f"Found {project_dir} with Settings.xml")

    elif project_dir.is_dir() and not settings_file.is_file():
        raise RuntimeError(f"No Settings.xml file was found in {project_dir}")

    elif not project_dir.is_dir():
        project_dir = get_project_dir(args.project)
        settings_file = project_dir / "Settings.xml"
        if project_dir.is_dir() and not settings_file.is_file():
            raise RuntimeError(f"No Settings.xml file was found in {project_dir}")

    custom_sty_path = project_dir / "custom.sty"
    if custom_sty_path.is_file():
        stylesheet = UsfmStylesheet("usfm.sty", custom_sty_path)
    else:
        stylesheet = UsfmStylesheet("usfm.sty")
    tokenizer = UsfmTokenizer(stylesheet)

    # Parse project settings to get book IDs
    settings = FileParatextProjectSettingsParser(project_dir).parse()
    sfm_files = get_books_to_process(settings, project_dir, args.books)

    # Check each file
    for sfm_file in sfm_files:
        
        # Read and tokenize the file
        with open(sfm_file, "r", encoding="utf-8") as f: usfm_text = f.read()
        tokens = list(tokenizer.tokenize(usfm_text))

        check_end_markers(tokens, sfm_file, stylesheet)

if __name__ == "__main__":
    main()
