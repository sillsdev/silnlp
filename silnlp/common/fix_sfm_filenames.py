"""
Script to fix SFM filenames in a Paratext project folder according to the Naming convention specified in Settings.xml.

Usage:
    python fix_sfm_filenames.py <project_folder> [--rename]

- <project_folder> can be an absolute path or a folder name relative to SIL_NLP_PT_DIR/projects
- By default, performs a dry run and prints proposed renames
- Use --rename to actually perform renaming

Filenames are constructed as PrePart + BookNameForm + PostPart (no separators)
Only files in the project root with extensions: .sfm, .usfm, .txt, .ptx, .ptu, .ptw, .pt7 (case-insensitive)
Only files whose first line begins with "\\id <bookid>" are considered
Conflicts are reported and skipped
Requires valid Settings.xml with <Naming> tag
"""
import argparse
import logging
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import re

from .environment import SIL_NLP_ENV

# Try to import machine.scripture utilities
try:
    from machine.scripture import get_books, BOOK_NUM_TO_ID
except ImportError:
    get_books = None
    BOOK_NUM_TO_ID = None

EXTENSIONS = {".sfm", ".usfm", ".txt", ".ptx", ".ptu", ".ptw", ".pt7"}

LOGGER = logging.getLogger("fix_sfm_filenames")
logging.basicConfig(level=logging.INFO)

def get_project_path(project_arg):
    p = Path(project_arg)
    if p.is_absolute():
        return p
    return SIL_NLP_ENV.pt_projects_dir / p

def parse_naming(settings_path):
    tree = ET.parse(settings_path)
    root = tree.getroot()
    # Find ScriptureText tag regardless of namespace or depth
    st = None
    for elem in root.iter():
        if elem.tag.endswith("ScriptureText"):
            st = elem
            break
    if st is None:
        return None
    naming = None
    for elem in st.iter():
        if elem.tag.endswith("Naming"):
            naming = elem
            break
    if naming is None:
        return None
    pre = naming.attrib.get("PrePart", "")
    post = naming.attrib.get("PostPart", "")
    book_form = naming.attrib.get("BookNameForm", "")
    # book_form is just an example, not a filter
    return pre, post

def get_bookid_from_first_line(first_line):
    # Extract bookid from first line like "\id MAT"
    m = re.match(r"^\\id\s+([A-Z0-9]{3,})", first_line.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None

def get_files_with_bookids(project_path):
    files = []
    for f in project_path.iterdir():
        if f.is_file() and f.suffix.lower() in EXTENSIONS:
            try:
                with f.open("r", encoding="utf-8") as fin:
                    first_line = fin.readline().strip()
                bookid = get_bookid_from_first_line(first_line)
                if bookid:
                    files.append((f, bookid))
            except Exception:
                continue
    return files

def main():
    parser = argparse.ArgumentParser(description="Fix SFM filenames in Paratext project folder.")
    parser.add_argument("project", help="Project folder (absolute or relative to SIL_NLP_ENV.pt_projects_dir)")
    parser.add_argument("--rename", action="store_true", help="Actually perform renaming (default: dry run)")
    args = parser.parse_args()

    project_path = get_project_path(args.project)
    if not project_path.is_dir():
        LOGGER.error(f"Project folder not found: {project_path}")
        sys.exit(1)

    settings_path = project_path / "Settings.xml"
    if not settings_path.is_file():
        LOGGER.error(f"Settings.xml not found in {project_path}")
        sys.exit(1)

    naming = parse_naming(settings_path)
    if not naming:
        LOGGER.error(f"No valid <Naming> tag found in Settings.xml")
        sys.exit(1)
    pre, post = naming

    files_with_bookids = get_files_with_bookids(project_path)
    if not files_with_bookids:
        LOGGER.info(f"No files found to rename in {project_path}")
        return

    # Build bookid -> booknum mapping using BOOK_NUM_TO_ID
    bookid_to_num = {}
    if BOOK_NUM_TO_ID:
        for num, bid in BOOK_NUM_TO_ID.items():
            bookid_to_num[bid.upper()] = num

    any_conflict = False
    rename_plan = []
    for f, bookid in files_with_bookids:
        booknum = bookid_to_num.get(bookid.upper()) if bookid_to_num else None
        if not booknum:
            continue
        # Construct target filename using template: PrePart + booknum + bookid + PostPart
        target_name = f"{pre}{booknum}{bookid.upper()}{post}"
        # Skip if file already matches the target name
        if f.name == target_name:
            continue
        target_path = project_path / target_name
        if target_path.exists():
            LOGGER.warning(f"Conflict: {target_path} already exists. Skipping {f.name}.")
            any_conflict = True
            continue
        rename_plan.append((f.name, target_name))
        if args.rename:
            f.rename(target_path)
            LOGGER.info(f"Renamed {f.name} -> {target_name}")
    if not args.rename:
        if rename_plan:
            print("Proposed renames:")
            for old, new in rename_plan:
                print(f"  {old} -> {new}")
        else:
            print("No files to rename.")
        if any_conflict:
            print("Some files were skipped due to conflicts.")

if __name__ == "__main__":
    main()
