"""
Script to standardize and correct filenames of Scripture text files (SFM/USFM format) 
in a Paratext project folder according to the naming convention specified in Settings.xml.

This script ensures that all book files follow a consistent format, which is important 
for interoperability, automation, and clarity in Bible translation projects.

Usage:
    python fix_filenames.py <project_folder> [--rename] [--check-settings]

- <project_folder> can be an absolute path or a folder name relative to SIL_NLP_ENV.pt_projects_dir
- By default, performs a dry run and prints proposed renames
- Use --rename to actually perform renaming
- Use --check-settings to analyze if Settings.xml should be updated instead of renaming files

The script validates:
- Settings.xml exists and has valid <Naming> tag
- PrePart is empty (warns and exits if not)
- BookNameForm follows supported patterns (e.g., "41MAT" or "41-MAT")
- BookID in filename matches BookID in file content (\id marker)
- Checks if Settings.xml should be updated to match consistent file naming

Only files with extensions: .sfm, .usfm, .txt, .ptx, .ptu, .ptw, .pt7 (case-insensitive)
Only files whose first line begins with "\\id <bookid>" are considered
Conflicts are reported and skipped
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from .environment import SIL_NLP_ENV

# machine.scripture utilities

from machine.scripture import BOOK_NUMBERS, book_id_to_number, is_book_id_valid, is_ot
from machine.corpora import FileParatextProjectSettingsParser

# Supported file extensions for Scripture files
EXTENSIONS = {".sfm", ".usfm", ".txt", ".ptx", ".ptu", ".ptw", ".pt7"}

# Logger setup
LOGGER = logging.getLogger("fix_filenames")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class NamingConvention:
    """Represents the naming convention from Settings.xml"""
    def __init__(self, pre_part: str, post_part: str, book_name_form: str):
        self.pre_part = pre_part
        self.post_part = post_part
        self.book_name_form = book_name_form
        self.has_delimiter = self._detect_delimiter()
    
    def _detect_delimiter(self) -> bool:
        """Detect if the BookNameForm uses a delimiter between book number and ID"""
        # Look for patterns like "41-MAT" vs "41MAT"
        pattern = r'^\d+[-_]?[A-Z0-9]+$'
        if re.match(pattern, self.book_name_form):
            return '-' in self.book_name_form or '_' in self.book_name_form
        return False
    
    def construct_filename(self, book_num: str, book_id: str) -> str:
        """Construct filename according to the naming convention"""
        if self.has_delimiter:
            delimiter = '-' if '-' in self.book_name_form else '_'
            return f"{self.pre_part}{book_num}{delimiter}{book_id.upper()}{self.post_part}"
        else:
            return f"{self.pre_part}{book_num}{book_id.upper()}{self.post_part}"


class FileInfo:
    """Information about a Scripture file"""
    def __init__(self, path: Path, filename_book_id: Optional[str], content_book_id: Optional[str]):
        self.path = path
        self.filename_book_id = filename_book_id
        self.content_book_id = content_book_id
        self.is_valid = content_book_id is not None
        self.ids_match = filename_book_id == content_book_id if both_exist(filename_book_id, content_book_id) else False


def both_exist(a, b) -> bool:
    """Helper to check if both values are not None"""
    return a is not None and b is not None


def get_project_path(project_arg: str) -> Path:
    """
    Resolve project path from argument.
    
    Args:
        project_arg: Project folder path (absolute or relative to SIL_NLP_ENV.pt_projects_dir)
        
    Returns:
        Path: Resolved project path
    """
    p = Path(project_arg)
    if p.is_absolute():
        return p
    return SIL_NLP_ENV.pt_projects_dir / p


def parse_settings_xml(settings_path: Path) -> Optional[NamingConvention]:
    """
    Parse Settings.xml file to extract naming convention.
    
    Args:
        settings_path: Path to Settings.xml file
        
    Returns:
        NamingConvention object or None if parsing fails
    """
    try:
        tree = ET.parse(settings_path)
        root = tree.getroot()
        
        # Find ScriptureText tag regardless of namespace or depth
        scripture_text = None
        for elem in root.iter():
            if elem.tag.endswith("ScriptureText"):
                scripture_text = elem
                break
        
        if scripture_text is None:
            LOGGER.error("No ScriptureText element found in Settings.xml")
            return None
        
        # Look for Naming tag
        naming_elem = None
        for elem in scripture_text.iter():
            if elem.tag.endswith("Naming"):
                naming_elem = elem
                break
        
        if naming_elem is None:
            LOGGER.error("No <Naming> tag found in Settings.xml")
            return None
        
        # Extract naming attributes
        pre_part = naming_elem.attrib.get("PrePart", "")
        post_part = naming_elem.attrib.get("PostPart", "")
        book_name_form = naming_elem.attrib.get("BookNameForm", "")
        
        return NamingConvention(pre_part, post_part, book_name_form)
        
    except ET.ParseError as e:
        LOGGER.error(f"Failed to parse Settings.xml: {e}")
        return None
    except Exception as e:
        LOGGER.error(f"Error reading Settings.xml: {e}")
        return None


def validate_naming_convention(naming: NamingConvention) -> bool:
    """
    Validate the naming convention meets requirements.
    
    Args:
        naming: NamingConvention object to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check PrePart is empty
    if naming.pre_part != "":
        LOGGER.error(f"PrePart is not empty: '{naming.pre_part}'. Please check that the filenames have a PrePart and update this code.")
        return False
    
    # Validate BookNameForm pattern
    # Should match patterns like "41MAT", "41-MAT", "041MAT", etc.
    pattern = r'^\d{2,3}[-_]?[A-Z0-9]{3,}$'
    if not re.match(pattern, naming.book_name_form):
        LOGGER.error(f"BookNameForm '{naming.book_name_form}' does not match expected pattern (e.g., '41MAT' or '41-MAT')")
        return False
    
    return True


def extract_book_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract book ID from filename.
    
    Args:
        filename: The filename to parse
        
    Returns:
        Book ID if found, None otherwise
    """
    # Remove extension
    name_without_ext = Path(filename).stem.upper()
    
    # First try to extract book number and then find the book ID
    # Pattern to match book number at start: 2-3 digits, optional delimiter
    book_num_match = re.match(r'^(\d{2,3})[-_]?(.*)$', name_without_ext)
    if book_num_match:
        book_num_str = book_num_match.group(1)
        remainder = book_num_match.group(2)
        
        # Try to find a valid book ID at the start of the remainder
        if BOOK_NUMBERS:
            for book_id in BOOK_NUMBERS.keys():
                if remainder.startswith(book_id):
                    return book_id
    
    # Fallback: try to match just the book ID without book number
    if BOOK_NUMBERS:
        for book_id in BOOK_NUMBERS.keys():
            if name_without_ext.startswith(book_id):
                return book_id
    
    # Last resort: try common patterns
    patterns = [
        r'^([A-Z0-9]{3})(?:[A-Z]*)?$',    # 3-letter book ID followed by optional text
        r'^([A-Z][0-9][A-Z]{2})(?:[A-Z]*)?$',  # Pattern like 1PE, 2CO, etc.
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name_without_ext)
        if match:
            candidate = match.group(1).upper()
            if BOOK_NUMBERS and candidate in BOOK_NUMBERS:
                return candidate
    
    return None


def extract_book_id_from_content(file_path: Path) -> Optional[str]:
    """
    Extract book ID from the first line of the file (\id marker).
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        Book ID if found, None otherwise
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        
        # Look for \id marker pattern
        match = re.match(r'^\\id\s+([A-Z0-9]{3,})', first_line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
    except Exception as e:
        LOGGER.warning(f"Could not read file {file_path}: {e}")
    
    return None


def scan_project_files(project_path: Path) -> List[FileInfo]:
    """
    Scan project directory for Scripture files and extract book information.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        List of FileInfo objects
    """
    files = []
    
    for file_path in project_path.iterdir():
        if not file_path.is_file():
            continue
        
        # Check if file has valid extension
        if file_path.suffix.lower() not in EXTENSIONS:
            continue
        
        # Extract book IDs from filename and content
        filename_book_id = extract_book_id_from_filename(file_path.name)
        content_book_id = extract_book_id_from_content(file_path)
        
        # Only include files that have content book ID (valid \id marker)
        if content_book_id is not None:
            file_info = FileInfo(file_path, filename_book_id, content_book_id)
            files.append(file_info)
            
            # Warn if filename and content book IDs don't match
            if filename_book_id and filename_book_id != content_book_id:
                LOGGER.warning(f"BookID mismatch in {file_path.name}: filename='{filename_book_id}', content='{content_book_id}'")
    
    return files


def get_book_number(book_id: str) -> Optional[str]:
    """
    Get book number for a given book ID.
    
    Args:
        book_id: Book ID (e.g., "MAT", "GEN")
        
    Returns:
        Book number as string with appropriate padding, or None if not found
    """
    if book_id_to_number is None:
        LOGGER.error("machine.scripture.book_id_to_number not available")
        return None
    

    try:
        # Use the machine.scripture function to get book number
        num = book_id_to_number(book_id.upper())
        if num == 0:
            raise ValueError(f"Invalid book ID: {book_id} returned 0 as book number.")
        
        if not is_ot(num):
            num += 1  # Adjust for NT and later books if needed
        
        # Format number with appropriate padding
        if num < 10:
            return f"0{num}"
        elif num < 100:
            return str(num)
        else:
            # Handle 3-digit book numbers
            return str(num)
    except Exception as e:
        LOGGER.warning(f"Could not get book number for {book_id}: {e}")
        return None


def analyze_file_consistency(files: List[FileInfo]) -> Optional[NamingConvention]:
    """
    Analyze files to determine if they follow a consistent naming pattern.
    
    Args:
        files: List of FileInfo objects
        
    Returns:
        NamingConvention if files are consistent, None otherwise
    """
    if not files:
        return None
    
    # Extract patterns from existing filenames
    patterns = set()
    post_parts = set()
    
    for file_info in files:
        if not file_info.content_book_id:
            continue
        
        filename = file_info.path.name
        book_num = get_book_number(file_info.content_book_id)
        
        if book_num is None:
            continue
        
        # Try to determine the pattern
        stem = file_info.path.stem
        suffix = file_info.path.suffix
        
        # Check if filename starts with book number
        if stem.startswith(book_num):
            remainder = stem[len(book_num):]
            if remainder.startswith('-') or remainder.startswith('_'):
                # Has delimiter
                delimiter = remainder[0]
                book_part = remainder[1:]
                if book_part == file_info.content_book_id:
                    patterns.add(f"{book_num}{delimiter}{file_info.content_book_id}")
                    post_parts.add(suffix)
            else:
                # No delimiter
                if remainder == file_info.content_book_id:
                    patterns.add(f"{book_num}{file_info.content_book_id}")
                    post_parts.add(suffix)
    
    # Check if all files follow the same pattern
    if len(post_parts) == 1 and len(patterns) >= len(files) * 0.8:  # 80% consistency threshold
        post_part = list(post_parts)[0]
        # Use first pattern as template
        sample_pattern = list(patterns)[0]
        
        # Determine if delimiter is used
        has_delimiter = '-' in sample_pattern or '_' in sample_pattern
        
        return NamingConvention("", post_part, sample_pattern)
    
    return None


def main():
    """Main function to handle command line arguments and orchestrate the renaming process."""
    parser = argparse.ArgumentParser(
        description="Standardize Scripture file names in Paratext project according to Settings.xml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_filenames.py my_project                    # Dry run
  python fix_filenames.py my_project --rename           # Actually rename files
  python fix_filenames.py /path/to/project --rename     # Use absolute path
  python fix_filenames.py my_project --check-settings   # Check if Settings.xml should be updated

The script validates that:
- Settings.xml exists with valid <Naming> tag
- PrePart is empty (required)
- BookNameForm follows supported patterns (e.g., "41MAT" or "41-MAT")
- BookID in filename matches BookID in file content

Supported file extensions: .sfm, .usfm, .txt, .ptx, .ptu, .ptw, .pt7
        """
    )
    
    parser.add_argument(
        "project", 
        help="Project folder (absolute path or relative to SIL_NLP_ENV.pt_projects_dir)"
    )
    parser.add_argument(
        "--rename", 
        action="store_true", 
        help="Actually perform renaming (default: dry run showing proposed changes)"
    )
    parser.add_argument(
        "--check-settings",
        action="store_true",
        help="Analyze if Settings.xml should be updated instead of renaming files"
    )
    
    args = parser.parse_args()
    
    # Resolve project path
    project_path = get_project_path(args.project)
    if not project_path.is_dir():
        LOGGER.error(f"Project folder not found: {project_path}")
        sys.exit(1)
    
    # Check for Settings.xml
    settings_path = project_path / "Settings.xml"
    if not settings_path.is_file():
        LOGGER.error(f"Settings.xml not found in {project_path}")
        sys.exit(1)
    
    # Parse Settings.xml
    naming = parse_settings_xml(settings_path)
    if naming is None:
        sys.exit(1)
    
    # Validate naming convention
    if not validate_naming_convention(naming):
        sys.exit(1)
    
    # Scan project files
    files = scan_project_files(project_path)
    if not files:
        LOGGER.info(f"No valid Scripture files found in {project_path}")
        return
    
    LOGGER.info(f"Found {len(files)} valid Scripture files")
    
    # Check if Settings.xml should be updated instead
    if args.check_settings:
        consistent_naming = analyze_file_consistency(files)
        if consistent_naming:
            LOGGER.info("Files follow a consistent naming pattern.")
            LOGGER.info(f"Current Settings.xml: PrePart='{naming.pre_part}', PostPart='{naming.post_part}', BookNameForm='{naming.book_name_form}'")
            LOGGER.info(f"Detected pattern: PrePart='{consistent_naming.pre_part}', PostPart='{consistent_naming.post_part}', BookNameForm='{consistent_naming.book_name_form}'")
            
            if (naming.pre_part != consistent_naming.pre_part or 
                naming.post_part != consistent_naming.post_part or
                naming.has_delimiter != consistent_naming.has_delimiter):
                LOGGER.info("Consider updating Settings.xml to match the file naming pattern instead of renaming files.")
                exit(1)
            else:
                LOGGER.info("Settings.xml already matches the file naming pattern.")
        else:
            LOGGER.info("Files do not follow a consistent naming pattern. File renaming may be needed.")
        return
    
    # Generate rename plan
    rename_plan = []
    conflicts = []
    
    for file_info in files:
        if not file_info.content_book_id:
            continue
        
        book_num = get_book_number(file_info.content_book_id)
        if book_num is None:
            LOGGER.warning(f"Could not determine book number for {file_info.content_book_id} in {file_info.path.name}")
            continue
        
        
        # Construct target filename
        target_filename = naming.construct_filename(book_num, file_info.content_book_id)
        
        # Skip if already correct
        if file_info.path.name == target_filename:
            continue
        
        target_path = project_path / target_filename
        
        # Check for conflicts
        if target_path.exists():
            conflicts.append((file_info.path.name, target_filename))
            continue
        
        rename_plan.append((file_info.path, target_filename))
    
    # Report results
    if conflicts:
        LOGGER.warning("The following files have naming conflicts and will be skipped:")
        for old_name, new_name in conflicts:
            LOGGER.warning(f"  {old_name} -> {new_name} (target already exists)")
    
    if not rename_plan:
        if conflicts:
            LOGGER.info("No files can be renamed due to conflicts.")
        else:
            LOGGER.info("All files already have correct names.")
        return
    
    # Execute or display plan
    if args.rename:
        LOGGER.info(f"Renaming {len(rename_plan)} files:")
        for old_path, new_name in rename_plan:
            new_path = project_path / new_name
            try:
                old_path.rename(new_path)
                LOGGER.info(f"  {old_path.name} -> {new_name}")
            except Exception as e:
                LOGGER.error(f"  Failed to rename {old_path.name}: {e}")
    else:
        LOGGER.info(f"Proposed renames ({len(rename_plan)} files):")
        for old_path, new_name in rename_plan:
            LOGGER.info(f"  {old_path.name} -> {new_name}")
        LOGGER.info("Use --rename to actually perform these renames.")


if __name__ == "__main__":
    main()