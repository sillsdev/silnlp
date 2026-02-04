# Exported from split_verses.py_dup5.

import argparse
import logging
import shutil
import time
from pathlib import Path

from machine.corpora import FileParatextProjectSettingsParser, UsfmStylesheet, UsfmToken, UsfmTokenizer, UsfmTokenType

from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
from .paratext import get_project_dir

LOGGER = logging.getLogger(__package__ + ".split_verses_v5")

# from .check_books import group_bible_books
VALID_CANONS = ["OT", "NT", "DT"]
VALID_BOOKS = OT_CANON + NT_CANON + DT_CANON

# Mapping of paragraph markers to their split markers
SPLIT_MARKER_MAP = {
    "ip": "ip",  # Split intro paragraphs into multiple intro paragraphs
    "v": "p",  # Split verses into paragraphs
    "m": "p",  # Split m markers into paragraphs
    "p": "p",  # Split long paragraphs into mulitple paragraphs
    "ef": "ef",  # Split extended footnotes into multiple extended footnotes
}

SENTENCE_ENDINGS = [".", "!", "?", "।"]
WORD_BREAKS = [" ", ",", ";", ":", "-"]
MAX_LENGTH = 200

SENTENCE_ENDINGS = '.!?'
SENTENCE_ENDS = '.!?'
CLOSE_QUOTES = '"\'"'
WORD_BREAKS = ' '

def s(x): return x or ''

def show_tokens_header():
    print(" No. |  Marker | Style |  TextType      | Data    | Text")

def show_tokens(settings, tokens, index=0):
    "Display token structure with types, markers, and text"
    for idx, token in enumerate(tokens, index):
        text_type = settings.stylesheet.get_tag(token.marker).text_type.name if token.marker else ''
        print(f"{s(idx):4} | {s(token.marker):7} | {token.type.name[:4] if token.type else '':5} | {text_type:14} | {s(token.data):7} | {s(token.text)}")


def copy_folder(source: Path, destination: Path):
    """
    Copies a source folder to a destination folder using pathlib and shutil.
    Includes a delay for rclone and S3 processing.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    shutil.copytree(source, destination, dirs_exist_ok=True)
    time.sleep(2)

    return destination

def get_paragraph_tokens(tokens, start_idx):
    "Return tokens from PARAGRAPH at start_idx until next PARAGRAPH (exclusive)"
    result = [tokens[start_idx]]
    for i in range(start_idx + 1, len(tokens)):
        if tokens[i].type == UsfmTokenType.PARAGRAPH: break
        result.append(tokens[i])
    return result


def get_paragraph_text(para_tokens):
    "Extract main text from paragraph tokens (TEXT after PARAGRAPH or END markers only)"
    texts = []
    for i,t in enumerate(para_tokens):
        if t.type != UsfmTokenType.TEXT: continue
        prev = para_tokens[i-1]
        if prev.type in (UsfmTokenType.PARAGRAPH, UsfmTokenType.END): texts.append(t.text)
    return ''.join(texts)


def get_paragraph_parts(para_tokens):
    "Return list of (end_idx, text_len) for each part ending at END marker or paragraph end"
    parts, text_len = [], 0
    for i,t in enumerate(para_tokens):
        if t.type == UsfmTokenType.TEXT:
            prev = para_tokens[i-1]
            if prev.type in (UsfmTokenType.PARAGRAPH, UsfmTokenType.END): text_len += len(t.text or '')
        if t.type == UsfmTokenType.END:
            parts.append((i, text_len))
            text_len = 0
    if text_len > 0 or (parts and parts[-1][0] < len(para_tokens)-1):
        parts.append((len(para_tokens)-1, text_len))
    return parts


def optimal_grouping(parts, max_len):
    "Return part indices after which to split to keep groups under max_len"
    splits, acc = [], 0
    for i,(end_idx, text_len) in enumerate(parts[:-1]):
        acc += text_len
        if acc + parts[i+1][1] > max_len:
            splits.append(i)
            acc = 0
    return splits


def split_paragraph_tokens(para_tokens, parts, split_after):
    "Split para_tokens into multiple paragraphs at given part indices"
    if not split_after: return [para_tokens]
    para_marker = para_tokens[0]
    boundaries = [0] + [parts[i][0]+1 for i in split_after] + [len(para_tokens)]
    result = []
    for i in range(len(boundaries)-1):
        chunk = para_tokens[boundaries[i]:boundaries[i+1]]
        if i > 0: chunk = [para_marker] + chunk
        result.append(chunk)
    return result


def find_break_positions(text):
    "Return list of (position, is_sentence_break) for valid break points"
    breaks = []
    i = 0
    while i < len(text):
        if text[i] in SENTENCE_ENDS:
            j = i + 1
            while j < len(text) and text[j] in CLOSE_QUOTES: j += 1
            if j < len(text) and text[j] in WORD_BREAKS:
                breaks.append((j, True))  # sentence break
        elif text[i] in WORD_BREAKS:
            breaks.append((i, False))  # word break only
        i += 1
    return breaks

#Mostly for testing
def process_long_paragraphs(tokens, settings, max_len=200, start_idx=0):
    "Find first paragraph exceeding max_len from start_idx, split it, display results, return next index"
    for idx in range(start_idx, len(tokens)):
        if tokens[idx].type != UsfmTokenType.PARAGRAPH: continue
        
        para = get_paragraph_tokens(tokens, idx)
        para_text = get_paragraph_text(para)
        if len(para_text) <= max_len: continue
        
        # Found one too long
        print(f"Token {idx}: \\{tokens[idx].marker} has {len(para_text)} chars (max {max_len})\n")
        print("ORIGINAL:")
        show_tokens_header()
        show_tokens(settings, para)
        
        parts = get_paragraph_parts(para)
        splits = optimal_grouping(parts, max_len)
        new_paras = split_paragraph_tokens(para, parts, splits)
        
        print(f"\nSPLIT INTO {len(new_paras)} PARAGRAPHS:")
        for i, new_para in enumerate(new_paras):
            text = get_paragraph_text(new_para)
            print(f"\n--- Paragraph {i+1} ({len(text)} chars) ---")
            show_tokens_header()
            show_tokens(settings, new_para)
        
        return idx + 1  # next index to continue from
    
    print("No more long paragraphs found.")
    return None


def split_long_text(text, max_len):
    "Split text into chunks under max_len, preferring sentence breaks over word breaks"
    if len(text) <= max_len: return [text]
    
    breaks = find_break_positions(text)
    sentence_breaks = [pos for pos, is_sentence in breaks if is_sentence]
    word_breaks = [pos for pos, is_sentence in breaks if not is_sentence]
    
    # Calculate optimal number of chunks
    num_chunks = (len(text) + max_len - 1) // max_len
    target_len = len(text) // num_chunks
    
    chunks, start = [], 0
    while start < len(text):
        if len(text) - start <= max_len:
            chunks.append(text[start:])
            break
        
        # Find best break near target position
        target = start + target_len
        best = None
        
        # Prefer sentence breaks
        for pos in sentence_breaks:
            if start < pos <= start + max_len:
                if best is None or abs(pos - target) < abs(best - target):
                    best = pos
        
        # Fall back to word breaks
        if best is None:
            for pos in word_breaks:
                if start < pos <= start + max_len:
                    if best is None or abs(pos - target) < abs(best - target):
                        best = pos
        
        if best is None: best = start + max_len  # hard cut as last resort
        
        chunks.append(text[start:best])
        start = best
        while start < len(text) and text[start] in WORD_BREAKS: start += 1
    
    return chunks


def process_long_paragraphs(tokens, settings, max_len=200, start_idx=0):
    "Find first paragraph exceeding max_len, split it, display results, return next index"
    for idx in range(start_idx, len(tokens)):
        if tokens[idx].type != UsfmTokenType.PARAGRAPH: continue
        
        para = get_paragraph_tokens(tokens, idx)
        para_text = get_paragraph_text(para)
        if len(para_text) <= max_len: continue
        
        print(f"Token {idx}: \\{tokens[idx].marker} has {len(para_text)} chars (max {max_len})\n")
        print("ORIGINAL:")
        show_tokens_header()
        show_tokens(settings, para)
        
        # First try splitting at END markers
        parts = get_paragraph_parts(para)
        splits = optimal_grouping(parts, max_len)
        new_paras = split_paragraph_tokens(para, parts, splits)
        
        # Check if any part still exceeds max_len and needs text splitting
        final_paras = []
        for new_para in new_paras:
            text = get_paragraph_text(new_para)
            if len(text) > max_len:
                # Need to split the text itself
                text_chunks = split_long_text(text, max_len)
                print(f"\n(Splitting {len(text)} char text into {len(text_chunks)} chunks)")
                for i, chunk in enumerate(text_chunks):
                    final_paras.append((new_para[0], chunk))  # (para_marker, text_chunk)
            else:
                final_paras.append((new_para, None))  # keep as token list
        
        print(f"\nSPLIT INTO {len(final_paras)} PARAGRAPHS:")
        for i, item in enumerate(final_paras):
            if item[1] is None:  # token list
                para_tokens = item[0]
                text = get_paragraph_text(para_tokens)
                print(f"\n--- Paragraph {i+1} ({len(text)} chars) ---")
                show_tokens_header()
                show_tokens(settings, para_tokens)
            else:  # text chunk
                marker, chunk = item
                print(f"\n--- Paragraph {i+1} ({len(chunk)} chars) ---")
                print(f"\\{marker.marker} {chunk}")
        
        return idx + 1
    
    print("No more long paragraphs found.")
    return None


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
    

def get_tokens(settings, sfm_file):
        
    if not sfm_file.is_file():
        raise RuntimeError(f"Could not find sfm_file: {sfm_file}")
    
    # Read and tokenize the file
    with open(sfm_file, "r", encoding="utf-8") as f:
        usfm_text = f.read()

    if not usfm_text:
        raise RuntimeError(f"No text read from sfm_file: {sfm_file}")

    tokenizer = UsfmTokenizer(settings.stylesheet)
    return list(tokenizer.tokenize(usfm_text))


def process_file(tokens, max_len=MAX_LENGTH):
    "Process the tokens of a single USFM file and return the split tokens."
    for idx in range(len(tokens)):
        if tokens[idx].type != UsfmTokenType.PARAGRAPH: continue
        
        para = get_paragraph_tokens(tokens, idx)
        para_text = get_paragraph_text(para)
        if len(para_text) <= max_len: continue
        
        print(f"Token {idx}: \\{tokens[idx].marker} has {len(para_text)} chars (max {max_len})\n")
        print("ORIGINAL:")
        show_tokens_header()
        show_tokens(settings, para)
        
        # First try splitting at END markers
        parts = get_paragraph_parts(para)
        splits = optimal_grouping(parts, max_len)
        new_paras = split_paragraph_tokens(para, parts, splits)
        
        # Check if any part still exceeds max_len and needs text splitting
        final_paras = []
        for new_para in new_paras:
            text = get_paragraph_text(new_para)
            if len(text) > max_len:
                # Need to split the text itself
                text_chunks = split_long_text(text, max_len)
                print(f"\n(Splitting {len(text)} char text into {len(text_chunks)} chunks)")
                for i, chunk in enumerate(text_chunks):
                    final_paras.append((new_para[0], chunk))  # (para_marker, text_chunk)
            else:
                final_paras.append((new_para, None))  # keep as token list
        
        print(f"\nSPLIT INTO {len(final_paras)} PARAGRAPHS:")
        for i, item in enumerate(final_paras):
            if item[1] is None:  # token list
                para_tokens = item[0]
                text = get_paragraph_text(para_tokens)
                print(f"\n--- Paragraph {i+1} ({len(text)} chars) ---")
                show_tokens_header()
                show_tokens(settings, para_tokens)
            else:  # text chunk
                marker, chunk = item
                print(f"\n--- Paragraph {i+1} ({len(chunk)} chars) ---")
                print(f"\\{marker.marker} {chunk}")
        
        return idx + 1
    
    print("No more long paragraphs found.")
    return None


def process_tokens(tokens, max_len=200):
    "Process all tokens, splitting paragraphs that exceed max_len. Returns new token list."
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type != UsfmTokenType.PARAGRAPH:
            result.append(token)
            i += 1
            continue
        
        para = get_paragraph_tokens(tokens, i)
        para_text = get_paragraph_text(para)
        
        if len(para_text) <= max_len:
            # Keep original tokens
            result.extend(para)
            i += len(para)
            continue
        
        # Split this paragraph
        parts = get_paragraph_parts(para)
        splits = optimal_grouping(parts, max_len)
        new_paras = split_paragraph_tokens(para, parts, splits)

        for new_para in new_paras:
            text = get_paragraph_text(new_para)
            if len(text) > max_len:
            # Need text-level splitting
                text_chunks = split_long_text(text, max_len)
                para_marker = new_para[0]
                for chunk in text_chunks:
                    result.append(UsfmToken(type=UsfmTokenType.PARAGRAPH, marker=para_marker.marker))
                    result.append(UsfmToken(type=UsfmTokenType.TEXT, text=chunk))
                    #print(UsfmToken(type=UsfmTokenType.PARAGRAPH, marker=para_marker.marker))
                    #print(UsfmToken(type=UsfmTokenType.TEXT, text=chunk))
            else:
                result.extend(new_para)
        
        i += len(para)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Split long paragraphs in USFM files")
    parser.add_argument("project", type=str, help="Paratext project name - the files in this folder will be modified in place.")
    parser.add_argument("--max", type=int, default=MAX_LENGTH, help="Maximum paragraph length.")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    )
    # parser.add_argument('--methods', nargs='+', default=['balanced'], help='Methods used to split long paragraphs, must be one of sentence, optimal, recursive, balanced.')
    parser.add_argument(
        "--show-from",
        type=int,
        help="Show the tokens found at given token number. Use show_limit to change the number shown.",
    )
    parser.add_argument(
        "--show-limit", default=25, help="Set the number of tokens to show, only has an effect when --show is used."
    )
    parser.add_argument(
        "--show-split",
        type=int,
        help="Show the tokens found at given token number and show the result after spliting, along with the next token number.",
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
        

    # Parse project settings to get book IDs
    settings = FileParatextProjectSettingsParser(project_dir).parse()
    sfm_files =  get_sfm_files_to_process(settings, project_dir, args.books)

    first_sfm_file = sfm_files[0]
    if args.show_from is not None:
        # Only SHOW tokens from the first book.
        print(f"Showing {args.show_limit} tokens from {first_sfm_file} beginning at token {args.show_from}\n")
        tokens = get_tokens(settings, first_sfm_file)
        show_tokens_header()
        show_tokens(tokens, start=args.show_from, limit=args.show_limit)
        exit()

    if args.show_split is not None:
        search_from = args.show_split
        print(f"Searching for next split after token {search_from} from {first_sfm_file}\n")
        tokens = get_tokens(settings, first_sfm_file)
        process_long_paragraphs(tokens, settings, max_len=args.max, start_idx=search_from)
        exit()

    output_dir = project_dir.parent / f"{project_dir.name}_split_{args.max}"
    

    # Copying the folder ensures that all necessary files are present.
    copy_folder(project_dir, output_dir)
    
    # Process each file
    for sfm_file in sfm_files:

        print(f"Processing sfm_file: {sfm_file}")
        tokens = get_tokens(settings, sfm_file)
        split_tokens = process_tokens(tokens, max_len=args.max)
        usfm_out = [token.to_usfm(include_newlines=True).replace('\r\n', '\n') for token in split_tokens]
        output_sfm_file = output_dir / sfm_file.name
        with open(output_sfm_file, 'w', encoding='utf-8') as f: f.write(''.join(usfm_out))
                    
    print(f"Done! Processed {len(sfm_files)} books in {output_dir}")


if __name__ == "__main__":
    main()



