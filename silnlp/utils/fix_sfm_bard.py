import argparse
import textwrap
from pathlib import Path
from typing import List

from .. import sfm
from ..common.paratext import get_book_path, get_project_dir
from ..common.translator import get_stylesheet
from ..sfm import usfm
from .collect_verse_counts import NT_canon, OT_canon

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)


def quick_check(book_path, stylesheet):

    with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
        try:
            doc: List[sfm.Element] = list(usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False))
            return False
        except Exception as err:
            return err


class Project:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.sfm_files = self._find_sfm_files()
        self.stylesheet = self._get_stylesheet()
        self.settings_file = self._get_settings_file()

    def _find_sfm_files(self) -> List[Path]:
        """Finds SFM files within the project directory."""
        return [
            file for file in self.project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]
        ]

    def _get_stylesheet(self) -> dict:
        """Retrieves the project's stylesheet."""
        # try:
        return get_stylesheet(self.project_dir)
        # except UnicodeDecodeError as err:
        #    print(f"A Unicode Decoding error occured while trying to read the stylesheet\nFurther Errors may be due to this error\n{err}\n")

    def _get_settings_file(self) -> Path:
        """Finds the project's Settings.xml file (if it exists)."""
        settings_file = self.project_dir / "Settings.xml"
        return settings_file if settings_file.is_file() else None

    def check_books(self, verbose: bool = False) -> List[str]:
        """Checks the validity of books within the project."""
        error_count = 0
        messages = []
        books_found = [file.name[2:5] for file in self.sfm_files]
        books_to_check = [book for book in valid_books if book in books_found]

        for book_to_check in books_to_check:
            book_path = get_book_path(self.project_dir, book_to_check)
            try:
                error = quick_check(book_path, self.stylesheet)
            except FileNotFoundError as err:
                error_count += 1
                if verbose:
                    messages.append(f"{err}")
            if error:
                error_count += 1
                messages.append(f"Book {book_path} failed to parse. {error}")

        if error_count:
            messages.append(
                f"In project {self.project_dir} there are {error_count} books that contain at least one error.\n"
            )
        else:
            messages.append(f"No errors were found in any of the books in project {self.project_dir}\n")

        return messages

    def show_stylesheet_markers(self):

        print(f"{self.stylesheet.keys()}")
        # Each marker contains style dictionary.
        # This is an example for the 'id' marker:
        #'id': {'Endmarker': None, 'Name': 'id - File - Identification', 'Description': 'File identification information (BOOKID, FILENAME, EDITOR, MODIFICATION DATE)', 'OccursUnder': {None}, 'TextProperties': {'nonvernacular', 'paragraph', 'book', 'nonpublishable'}, 'TextType': 'Other', 'StyleType': 'Paragraph', 'FontSize': '12'},
        # The 'TextProperties' contains info about whether it is : {'publishable', 'paragraph', 'vernacular', 'level_3'} for example.
        # 'paragraph is omitted for inline markers.
        #
        # The 'TextType': contains 'VerseText',
        # The 'StyleType': 'Paragraph'
        # If both TextProperties includes 'paragraph' and StyleType = 'Paragraph' then probably these markers should only appear at the beginnig of a line.
        all_markers = {marker for marker in self.stylesheet.keys()}
        markers_as_para = {marker for marker in all_markers if "paragraph" in self.stylesheet[marker]["TextProperties"]}
        markers_with_para_style = {
            marker for marker in all_markers if "Paragraph" in self.stylesheet[marker]["StyleType"]
        }
        para_markers_with_end_marker = {
            marker for marker in markers_as_para if self.stylesheet[marker]["Endmarker"] is not None
        }
        both_para_markers = markers_as_para.intersection(markers_with_para_style)
        inline_markers = all_markers.difference(both_para_markers)
        para_style_only_markers = markers_with_para_style.difference(markers_as_para)
        line_only_markers = markers_as_para.difference(markers_with_para_style)

        print(f"\nThese are the paragraph markers. {sorted(markers_as_para)} {len(markers_as_para)}")
        print(f"\nThese are the paragraph only markers. {line_only_markers} {len(line_only_markers)}")
        print(
            f"\nThese are the paragraph markers with an Endmarker. {para_markers_with_end_marker} {len(para_markers_with_end_marker)}"
        )

        for para_marker_with_end_marker in sorted(para_markers_with_end_marker):
            print(f"{para_marker_with_end_marker}  :  {self.stylesheet[para_marker_with_end_marker]['Endmarker']}")

        print(f"\nThese are the Paragraph style markers. {markers_with_para_style} {len(markers_with_para_style)}")
        print(f"\nThese are the Paragraph style only markers. {para_style_only_markers} {len(para_style_only_markers)}")
        print(
            f"\nThese are the markers that are both line and line style markers. {both_para_markers} {len(both_para_markers)}"
        )
        print(f"\nThese are the inline style markers. {inline_markers} {len(inline_markers)}")
        for inline_marker in sorted(inline_markers):
            if self.stylesheet[inline_marker]["Endmarker"]:
                print(f"{inline_marker}  :  {self.stylesheet[inline_marker]['Endmarker']}")


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="fix_sfm",
        description="Do a quick check of books in a project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", type=str, help="A folder containing projects to check and fix.")

    args = parser.parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise RuntimeError(f"Can't find the input directory {input_dir}")

    s_project_dir = get_project_dir("")
    if input_dir == s_project_dir:
        print(f"\nWARNIING: Operating on the live S:\ drive: {s_project_dir}. Please make a copy of the project.\n")

    # Find which folders 'look like' Paratext project.
    initial_project = Project(input_dir)
    possible_projects = [Project(item) for item in input_dir.glob("*") if item.is_dir()]

    # possible_projects = [initial_project]
    # possible_projects.extend(Project(item) for item in input_dir.glob("*") if item.is_dir())

    # print(f"Dirs found are :")
    # for possible_project_dir in possible_project_dirs:
    #     print(possible_project_dir)

    projects = list()
    for possible_project in possible_projects:
        if len(possible_project.sfm_files) > 0 and possible_project.settings_file:
            projects.append(possible_project)

    print(f"\nFound {len(projects)} Paratext project folders in {input_dir}")

    # For each book in each project  do a quick check and report errors with the file name and line number.
    for project in projects:
        for message in project.check_books():
            print(message)


if __name__ == "__main__":
    main()
