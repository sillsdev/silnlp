from pathlib import Path

from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
VALID_CANONS = ["OT", "NT", "DT"]
VALID_BOOKS = OT_CANON + NT_CANON + DT_CANON


def expand_book_list(books):
    """Parse books argument and expand NT/OT/DT into full book lists"""
    books_to_check = []
    canons_to_add = [canon for canon in books if canon in ["NT", "OT", "DT"]]
    for canon_to_add in canons_to_add:
        if canon_to_add == "OT":
            books_to_check += OT_CANON
        if canon_to_add == "NT":
            books_to_check += NT_CANON
        if canon_to_add == "DT":
            books_to_check += DT_CANON
    books_to_check += [book for book in books if book in VALID_BOOKS]
    return [book for book in VALID_BOOKS if book in set(books_to_check)]


def get_sfm_files_to_process(settings, project_dir, specified_books):
    sfm_suffix = Path(settings.file_name_suffix).suffix.lower()[1:]
    # print(f"suffix is {sfm_suffix}")

    # Find all SFM/USFM files
    sfm_files = [
        file
        for file in project_dir.glob("*")
        if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm", sfm_suffix]
    ]

    # Parse books argument
    if specified_books:
        book_list = expand_book_list(specified_books)

        # Get book IDs for found files
        ids_of_books_found = [settings.get_book_id(sfm_file.name) for sfm_file in sfm_files]
        return [sfm_file for sfm_file in sfm_files if settings.get_book_id(sfm_file.name) in book_list]

    # No books are specified or filtered,  return all of them.
    else:
        return sfm_files