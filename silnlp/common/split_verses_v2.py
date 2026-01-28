import argparse
import logging
import shutil
from pathlib import Path

from collections import Counter
from machine.corpora import UsfmTokenizer, UsfmToken, UsfmTokenType
from machine.corpora import FileParatextProjectSettingsParser, UsfmFileText
from machine.scripture import book_number_to_id, get_chapters
from sympy import expand

from .paratext import get_project_dir
from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
from .check_books import group_bible_books
VALID_CANONS = ["OT", "NT", "DT"]
VALID_BOOKS = OT_CANON + NT_CANON + DT_CANON

LOGGER = logging.getLogger(__package__ + ".split_verses_v2")

# Mapping of paragraph markers to their split markers
SPLIT_MARKER_MAP = {
    'ip': 'ip',  # intro paragraphs stay as intro paragraphs
    'm': 'p',    # continuation of m markers become regular paragraphs
    'p': 'p',    # Splitting long paragraphs remain as paragraphs
    'v': 'p',    # Splitting verses into paragraphs
}

SENTENCE_ENDINGS = ['.', '!', '?', '।']
WORD_BREAKS = [' ', ',', ';', ':', '-']
MAX_LENGTH = 200

def copy_folder(source: Path, destination: Path):
    """
    Copies a source folder to a destination folder using pathlib and shutil.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    shutil.copytree(source, destination, dirs_exist_ok=True)
    

def get_split_marker(original_marker):
    """Get the marker to use for split text, default to 'p'"""
    return SPLIT_MARKER_MAP.get(original_marker, 'p')


def find_split_point(text, max_length=275, hard_max=325):
    """Find a sentence or word break near max_length"""
    if len(text) <= max_length:
        return None
    
    # First try to find sentence ending before hard_max
    for punct in SENTENCE_ENDINGS:
        split_pos = text.rfind(punct, 0, hard_max)
        if split_pos > 0:
            return split_pos + 1
    
    # If not found, look for word break before max_length
    for break_char in WORD_BREAKS:
        split_pos = text.rfind(break_char, 0, max_length)
        if split_pos > 0:
            return split_pos + 1
    
    return None  # Can't split


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


def split_into_sentences(text):
    sentences,current = [],0
    for i,char in enumerate(text):
        if char in SENTENCE_ENDINGS and (i == len(text)-1 or text[i+1] == ' '):
            sentences.append(text[current:i+1])
            current = i+1
    if current < len(text): sentences.append(text[current:])
    return [s.strip() for s in sentences if s.strip()]


def split_text_balanced(text, max_length=MAX_LENGTH):
    """Split text into balanced groups of sentences"""
    if len(text) <= max_length:
        return [text]
    
    # Split into sentences first
    sentences = split_into_sentences(text)
    
    # Calculate total length
    total_length = sum(len(s) for s in sentences)
    
    # Calculate optimal number of groups
    import math
    num_groups = math.ceil(total_length / max_length)
    target_length = total_length / num_groups
    
    # Greedy grouping with target length
    groups = []
    current_group = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # Check if adding this sentence would exceed max_length
        if current_length + sentence_len > max_length:
            # Save current group and start new one
            if current_group:
                groups.append(' '.join(current_group))
            current_group = [sentence]
            current_length = sentence_len
        # Check if we should start a new group based on target
        elif current_length > 0 and current_length >= target_length:
            groups.append(' '.join(current_group))
            current_group = [sentence]
            current_length = sentence_len
        else:
            # Add to current group
            current_group.append(sentence)
            current_length += sentence_len
    
    # Add final group
    if current_group:
        groups.append(' '.join(current_group))
    
    return groups if groups else [text]


def split_text_optimally(text, max_length=MAX_LENGTH):
    if len(text) <= max_length: return [text]
    
    sentence_positions = []
    for punct in SENTENCE_ENDINGS:
        pos = 0
        while True:
            pos = text.find(punct, pos)
            if pos == -1: break
            sentence_positions.append(pos + 1)
            pos += 1
    
    if not sentence_positions:
        for break_char in WORD_BREAKS:
            pos = 0
            while True:
                pos = text.find(break_char, pos)
                if pos == -1: break
                sentence_positions.append(pos + 1)
                pos += 1
    
    if not sentence_positions: return [text]
    
    sentence_positions.sort()
    chunks,current_start = [],0
    
    while current_start < len(text):
        target = current_start + max_length
        best_split = min([pos for pos in sentence_positions if pos > current_start and pos <= target + 50], 
                        key=lambda p: abs(p - target), default=None)
        if best_split is None: best_split = min([pos for pos in sentence_positions if pos > current_start], default=len(text))
        chunks.append(text[current_start:best_split])
        current_start = best_split
    
    return chunks


def process_file(input_path, max_length, method='sentence', verbosity=0):
    """Process a single USFM file, splitting long paragraphs"""

    output_path = input_path

    # Read and tokenize the file
    with open(input_path, 'r', encoding='utf-8') as f:
        usfm_text = f.read()
    
    tokenizer = UsfmTokenizer()
    tokens = list(tokenizer.tokenize(usfm_text))
    
    # Process tokens, splitting long TEXT tokens
    new_tokens = []
    current_para_marker = None  # Track current paragraph marker
    split_counter = Counter()

    for token in tokens:
        # Track paragraph markers
        if token.type == UsfmTokenType.PARAGRAPH:
            current_para_marker = token.marker
            #print(current_para_marker)
            new_tokens.append(token)
        elif token.type == UsfmTokenType.TEXT and token.text and len(token.text) > max_length:
            # Determine split marker based on current paragraph
            split_marker = get_split_marker(current_para_marker)
            split_counter[split_marker] += 1
            if method == 'sentence':
                text_chunks = split_into_sentences(token.text)
            elif method == 'optimal':
                text_chunks = split_text_optimally(token.text, max_length=max_length)
            elif method == 'recursive':
                text_chunks = split_text_recursively(token.text, max_length=max_length)
            elif method == 'balanced':
                text_chunks = split_text_balanced(token.text, max_length=max_length)
            else:
                print(f"Method {method} isn't one of the valid methods: sentence, optimal, recursive.")
                exit(0)
            if verbosity >= 3:    
                for i, text_chunk in enumerate(text_chunks, 1):
                    print(i,len(text_chunk),text_chunk)

            # Add first chunk as TEXT
            new_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=text_chunks[0]))
            # For each remaining chunk: add PARAGRAPH + TEXT
            for text_chunk in text_chunks[1:]:
                new_tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker=split_marker))
                new_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=text_chunk))
        else:
            # Keep token as-is
            new_tokens.append(token)
    
    if len(split_counter) > 0:
        if verbosity >= 2:
            print(f"Saving {output_path} after splitting lines. {split_counter}")
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
    else:
        if verbosity >= 2:
            print(f"No changes were needed to for {input_path}")

def main():
    parser = argparse.ArgumentParser(description='Split long paragraphs in USFM files')
    parser.add_argument('project', help='Paratext project name')
    parser.add_argument('--max', type=int, default=225, help='Maximum paragraph length.')
    parser.add_argument("--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'")
    parser.add_argument('--methods', nargs='+', default=['sentence'], help='Methods used to split long paragraphs, must be one of sentence, optimal, recursive, balanced.')
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity level (e.g., -v, -vv, -vvv)")

    args = parser.parse_args()
    print(args)

    # Set verbosity level
    verbosity = args.verbose

    # Get project directory
    project_dir = get_project_dir(args.project)
    output_dir = project_dir.parent / f"{project_dir.name}_split{args.max}_{args.method}"
    
    # Copying the folder ensures that all necessary files are present.
    copy_folder(project_dir, output_dir)
   
    # All processing now is within the copied folder.
    project_dir = output_dir

    # Parse project settings to get book IDs
    settings = FileParatextProjectSettingsParser(project_dir).parse()
    
    # Find all SFM/USFM files
    sfm_files = [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]]
    books_found = [settings.get_book_id(sfm_file.name) for sfm_file in sfm_files]
    
    # Get book IDs for found files
    books_found = [settings.get_book_id(sfm_file.name) for sfm_file in sfm_files]
    books_to_process = []

    # Parse books argument
    books = args.books
    if args.books:
        if books:
            specified_books = expand_book_list(books)
            books_to_process = [book for book in specified_books if book in books_found]
            if not books_to_process:
                print(f"None of the specified books: {specified_books} were found in the project folder: {project_dir}")
    else:
        print("No books specified, all books will be processed.")
        books_to_process = books_found

    if books_to_process:
        if verbosity >= 1:
           print(f"Will process these books:\n{books_to_process}")

    # Process each file
    for method in args.methods:
        for sfm_file in sfm_files:
            book_id = settings.get_book_id(sfm_file.name)
            if book_id in books_to_process:
                output_path = output_dir / sfm_file.name
                if verbosity >= 1:
                    print(f"Processing {sfm_file}")
                process_file(sfm_file, args.max, method=method, verbosity=verbosity)

    print(f"Done! Processed {len(books_to_process)} books to {output_dir}")

if __name__ == '__main__':
    main()


