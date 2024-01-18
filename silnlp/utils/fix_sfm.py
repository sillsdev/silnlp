import argparse
import textwrap
from pathlib import Path
from typing import List

from .. import sfm
from ..common.paratext import get_book_path, get_project_dir
from ..common.translator import get_stylesheet
from ..sfm import usfm
from .collect_verse_counts import DT_canon, NT_canon, OT_canon

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)
# valid_books.extend(DT_canon)


def get_sfm_files(project_dir):
    return [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]]


def quick_check(book_path, stylesheet):

    with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
        try:
            doc: List[sfm.Element] = list(usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False))
            return False
        except Exception as err:
            return err


def show_stylesheet_markers(stylesheet):
    print(f"{stylesheet.keys()}")
    # Each marker contains style dictionary.
    # This is an example for the 'id' marker:
    #'id': {'Endmarker': None, 'Name': 'id - File - Identification', 'Description': 'File identification information (BOOKID, FILENAME, EDITOR, MODIFICATION DATE)', 'OccursUnder': {None}, 'TextProperties': {'nonvernacular', 'paragraph', 'book', 'nonpublishable'}, 'TextType': 'Other', 'StyleType': 'Paragraph', 'FontSize': '12'},
    # The 'TextProperties' contains info about whether it is : {'publishable', 'paragraph', 'vernacular', 'level_3'} for example.
    # 'paragraph is omitted for inline markers.
    #
    # The 'TextType': contains 'VerseText',
    # The 'StyleType': 'Paragraph'
    # If both TextProperties includes 'paragraph' and StyleType = 'Paragraph' then probably these markers should only appear at the beginnig of a line.
    all_markers = {marker for marker in stylesheet.keys()}
    markers_as_para = {marker for marker in all_markers if "paragraph" in stylesheet[marker]["TextProperties"]}
    markers_with_para_style = {marker for marker in all_markers if "Paragraph" in stylesheet[marker]["StyleType"]}
    para_markers_with_end_marker = {marker for marker in markers_as_para if stylesheet[marker]["Endmarker"] is not None}
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
        print(f"{para_marker_with_end_marker}  :  {stylesheet[para_marker_with_end_marker]['Endmarker']}")

    print(f"\nThese are the Paragraph style markers. {markers_with_para_style} {len(markers_with_para_style)}")
    print(f"\nThese are the Paragraph style only markers. {para_style_only_markers} {len(para_style_only_markers)}")
    print(
        f"\nThese are the markers that are both line and line style markers. {both_para_markers} {len(both_para_markers)}"
    )
    print(f"\nThese are the inline style markers. {inline_markers} {len(inline_markers)}")
    for inline_marker in sorted(inline_markers):
        if stylesheet[inline_marker]["Endmarker"]:
            print(f"{inline_marker}  :  {stylesheet[inline_marker]['Endmarker']}")


def get_project_details(project_dir):

    sfm_files = get_sfm_files(project_dir)

    if sfm_files:
        # Check for a Settings.xml file (should make this case insensitive)
        settings_file = project_dir / "Settings.xml"
        if not settings_file.is_file():
            settings_file = None

        stylesheet = get_stylesheet(project_dir)
        return {
            "project_dir": project_dir,
            "sfm_files": sfm_files,
            "stylesheet": stylesheet,
            "settings_file": settings_file,
        }


def check_project(project_folder, verbose):
    error_count = 0
    messages = list()
    
    sfm_files = get_sfm_files(project_folder)
    books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]

    # Get list of books to check.
    books_to_check = [book for book in valid_books if book in books_found]

    for book_to_check in books_to_check:
        error = False
        book_path = get_book_path(project_folder, book_to_check)

        try:
            error = quick_check(book_path, get_stylesheet(project_folder))
            #print(error)
        except Exception as err:
            error_count += 1
            if verbose:
                messages.append(f"{err}")
            if error:
                error_count += 1
                messages.append(f"Book {book_path} failed to parse. {error}")

    if error_count:
        messages.append(f"In project {project_folder} there are {error_count} books that contain at least one error.\n")
    else:
        messages.append(f"No errors were found in any of the books {' '.join(book for book in books_to_check)} in project {project_folder}\n")

    return messages


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="fix_sfm",
        description="Do a quick check of books in a project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", type=str, help="A folder containing projects to check and fix.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Describe the process in more detail.")
    parser.add_argument(
        "--very-verbose", action="store_true", default=False, help="Show the markers for the first sytlesheet and exit."
    )

    projects = list()
    args = parser.parse_args()

    verbose = args.verbose
    very_verbose = args.very_verbose
    if very_verbose:
        verbose = True

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise RuntimeError(f"Can't find the input directory {input_dir}")

    s_project_dir = get_project_dir("")
    if input_dir == s_project_dir:
        print(f"Will not operate on the live S:\ drive: {s_project_dir} . Please make a copy of the projects first.")
        exit()

    # Find which folders 'look like' Paratext project.

    possible_project_dirs = [input_dir]
    possible_project_dirs.extend(item for item in input_dir.glob("*") if item.is_dir())
    # print(f"Dirs found are :")
    # for possible_project_dir in possible_project_dirs:
    #     print(possible_project_dir)
    
    project_folders = list()
    for possible_project_dir in possible_project_dirs:
        sfm_files = get_sfm_files(possible_project_dir)
        if len(sfm_files) > 0 :
            # Check for a Settings.xml file (should make this case insensitive)
            settings_file = possible_project_dir / "Settings.xml"
            if settings_file.is_file():
                project_folders.append(possible_project_dir)
        
    print(f"\nFound {len(project_folders)} Paratext project folders in {input_dir}")
    
    # For each of those folders do a quick check on each of the SFM files.
    for project_folder in project_folders:
        if verbose:
            print(f"Checking {project_folder}")
        messages = check_project(project_folder,verbose)
        for message in messages:
            print(message)


if __name__ == "__main__":
    main()

# Simple errors that we could fix:
# Repeated markers: ie. "\c \c 1" or "\v\v 1" or "\v \v 1"
# Text after Chapter marker "\c 1 text" or "\c 1\r\ntext"  Could be replaced with "\c 1\r\n\\rem "
# missing verse number after \v
# orphan end marker \{marker}*: no matching opening marker \{marker}
# \v 47 text \47  causes
# \v 2. text or punctuation rather than a space after the verse number.
# \v29 space missing between \v and verse number.
# Pan\kinggan backslash used and second word gets interpreted as an unknown marker.  Replace with a space.
# \s Amg Ikapitung Trumpeta \P    '\P' shouldn't appear at the end of a line, and shouldn't be uppercase - delete it.

# Empty verse marker appears between two verses:
# \v 22 “राजाले त्‍यो नोकरलाई भन्‍यो, ‘दुष्‍ट नोकर, तेरो मुखको वचनले म तलाई दण्‍ड दिनेछु। यदि म जे दिंदैन त्‍यही लिन्‍छु, र जे छर्दैन त्‍यसैको कटनी गर्छु भनेर तैले जानेको थिइस भने,
# \v
# \v 23 मैले दिएको पैँसा बैंकमा किन राखेनस्? राख्थिस् भने, म आएर ब्‍याजसहित पैँसा फिर्ता पाउँनेथिएँ।’

# \S1 shouldn't be uppercase or in the middle of a line.
# \v 24 ꤓꤌꤣ꤭ꤚꤢꤪ ꤢ꤬ ... ꤔꤌꤣ꤬ꤒꤢ꤬ꤟꤢꤩ꤬꤯ \S1 ꤋꤝꤤꤢ꤬ꤗꤤ꤬ꤐꤢ꤬ ꤞꤛꤢꤩ꤭
# Replace with
# \v 24 ꤓꤌꤣ꤭ꤚꤢꤪ ꤢ꤬ ... ꤔꤌꤣ꤬ꤒꤢ꤬ꤟꤢꤩ꤬꤯
# \s1 ꤋꤝꤤꤢ꤬ꤗꤤ꤬ꤐꤢ꤬ ꤞꤛꤢꤩ꤭

# Missing space after chapter number '1' - incorrect error message - nothing should follow the chapter number.
# \c 1:

# Missing verse number after \v
# \v “27 अब म राजा भएको मन नपराउनेहरुलाई मेरो छेउमा ल्‍याएर मार’।”

# Space on line before marker:
#  \ss बप्‍तिष्‍मा दिने यूहन्‍ना

# Marker not in stylesheet. Replace \ss with \s
# \ss बप्‍तिष्‍मा दिने यूहन्‍ना

# Bold inline marker on a line of its own - delete line.
# \b

# \q1 marker not followed by a space.
# \v 2 text \q1<<Ang
