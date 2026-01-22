import argparse
import logging
from pathlib import Path

from machine.corpora import UsfmTokenizer, UsfmToken, UsfmTokenType
from machine.corpora import FileParatextProjectSettingsParser, UsfmFileText
from machine.scripture import book_number_to_id, get_chapters
from sympy import expand

from .paratext import get_project_dir
from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
from .check_books import group_bible_books
VALID_BOOKS = OT_CANON + NT_CANON + DT_CANON

LOGGER = logging.getLogger(__package__ + ".split_long_verses")


def find_split_point(text, max_length=250):
    """Find a sentence break near max_length"""
    if len(text) <= max_length:
        return None
    
    # First try to find period before max_length
    split_pos = text.rfind('.', 0, max_length)
    
    if split_pos > 0:
        return split_pos + 1
    
    # If not found, look for first period after max_length
    split_pos = text.find('.', max_length)
    
    if split_pos > 0:
        return split_pos + 1
    
    return None  # No period found at all


def split_text_recursively(text, max_length=275):
    """Recursively split text and return list of text chunks"""
    if len(text) <= max_length:
        return [text]
    
    split_pos = find_split_point(text, max_length=max_length)
    if split_pos is None:
        return [text]  # Can't split, return as-is
    
    first_part = text[:split_pos]
    rest = text[split_pos:]
    
    # Recursively split the rest
    return [first_part] + split_text_recursively(rest, max_length)


def expand_book_list(books):
    """Parse books argument and expand NT/OT/DT into full book lists"""
    books_to_check = []
    canons_to_add = [canon for canon in books if canon in ["NT", "OT", "DT"]]
    for canon_to_add in canons_to_add:
        if canon_to_add == "OT": books_to_check += OT_CANON
        if canon_to_add == "NT": books_to_check += NT_CANON
        if canon_to_add == "DT": books_to_check += DT_CANON
    books_to_check += [book for book in books if book in VALID_BOOKS]
    return [book for book in VALID_BOOKS if book in set(books_to_check)]


def process_file(input_path, output_path, max_length):
    """Process a single USFM file, splitting long paragraphs"""
    # Read and tokenize the file
    with open(input_path, 'r', encoding='utf-8') as f:
        usfm_text = f.read()
    
    tokenizer = UsfmTokenizer()
    tokens = list(tokenizer.tokenize(usfm_text))
    
    # Process tokens, splitting long TEXT tokens
    new_tokens = []
    for token in tokens:
        if token.type == UsfmTokenType.TEXT and token.text and len(token.text) > max_length:
            text_chunks = split_text_recursively(token.text, max_length=max_length)
            if len(text_chunks) > 1:
                # Add first chunk as TEXT
                new_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=text_chunks[0]))
                # For each remaining chunk: add PARAGRAPH + TEXT
                for text_chunk in text_chunks[1:]:
                    new_tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker='p'))
                    new_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=text_chunk))
            else:
                new_tokens.append(token)
        else:
            # Keep token as-is
            new_tokens.append(token)
    

    # Write tokens to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, token in enumerate(new_tokens):
            # Add newline before PARAGRAPH markers (except first token)
            if i > 0 and token.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE]:
                f.write('\n')
            usfm_str = token.to_usfm()
            if i < len(new_tokens) - 1 and new_tokens[i + 1].type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE]:
                usfm_str = usfm_str.rstrip(' ')
            f.write(usfm_str)


def main():
    parser = argparse.ArgumentParser(description='Split long paragraphs in USFM files')
    parser.add_argument('project', help='Paratext project name')
    parser.add_argument('--max', type=int, default=275, help='Maximum paragraph length (default: 275)')
    parser.add_argument('--books', help='Books to process (e.g., "MRK,LUK" or "NT")')
    
    args = parser.parse_args()

    # Get project directory
    project_dir = get_project_dir(args.project)
    
    # Parse project settings to get book IDs
    settings = FileParatextProjectSettingsParser(project_dir).parse()
    
    # Find all SFM/USFM files
    sfm_files = [
        file for file in project_dir.glob("*") 
        if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]
    ]
    
    # Get book IDs for found files
    books_found = [settings.get_book_id(sfm_file.name) for sfm_file in sfm_files]
    
    # Parse books argument
    books = args.books.replace(',', ' ').replace(';', ' ').split()
    if books:
        specified_books = expand_book_list(books)
        books_to_process = [book for book in specified_books if book in books_found]
        if not books_to_process:
            print(f"None of the specified books: {specified_books} were found in the project folder: {project_dir}")
    else:
        print("No books specified, all books will be processed.")
        books_to_process = books_found

    if books_to_process:
        print(f"Will process these books:\n{books_to_process}")

    # Process each file
    for sfm_file in sfm_files:
        book_id = settings.get_book_id(sfm_file.name)
        if book_id in books_to_process:

            # Create output filename with _split suffix
            output_path = sfm_file.parent / f"{sfm_file.stem}_split{sfm_file.suffix}"
            print(f"Processing {sfm_file.name} -> {output_path.name}")
            process_file(sfm_file, output_path, args.max)
    
    print(f"Done! Processed {len(books_to_process)} books.")


if __name__ == '__main__':
    main()
