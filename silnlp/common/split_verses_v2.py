import argparse
import logging
import shutil
from pathlib import Path
import time

from collections import Counter
from .environment import SIL_NLP_ENV
from machine.corpora import UsfmTokenizer, UsfmToken, UsfmTokenType, UsfmStylesheet
from machine.corpora import FileParatextProjectSettingsParser, UsfmFileText
from regex import P


#from machine.scripture import book_number_to_id, get_chapters

from .paratext import get_project_dir
from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
#from .check_books import group_bible_books
VALID_CANONS = ["OT", "NT", "DT"]
VALID_BOOKS = OT_CANON + NT_CANON + DT_CANON

LOGGER = logging.getLogger(__package__ + ".split_verses_v2")

# Mapping of paragraph markers to their split markers
SPLIT_MARKER_MAP = {
    'ip': 'ip',  # Split intro paragraphs into multiple intro paragraphs
    'v': 'p',    # Split verses into paragraphs
    'm': 'p',    # Split m markers into paragraphs
    'p': 'p',    # Split long paragraphs into mulitple paragraphs
    'ef': 'ef',  # Split extended footnotes into multiple extended footnotes
}

SENTENCE_ENDINGS = ['.', '!', '?', '।']
WORD_BREAKS = [' ', ',', ';', ':', '-']
MAX_LENGTH = 200

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

def show_tokena(tokens, start=0, limit=20):
    "Display token structure with types, markers, and text"
    for idx,token in enumerate(tokens[start:start+limit], start):
        print(f"{idx if idx else 0:4d} | {token.type:25s} | {token.marker if token.marker else '':7} | {token.data if token.data else '':7s} | {token.text if token.text else ''}")


def get_split_marker(original_marker):
    """Get the marker to use for split text, default to 'p'"""
    return SPLIT_MARKER_MAP.get(original_marker, 'p')


def find_split_point(text, max_len=MAX_LENGTH, hard_max=MAX_LENGTH):
    """Find a sentence or word break near max_len"""
    if len(text) <= max_len:
        return None
    
    # First try to find sentence ending before hard_max
    for punct in SENTENCE_ENDINGS:
        split_pos = text.rfind(punct, 0, hard_max)
        if split_pos > 0:
            return split_pos + 1
    
    # If not found, look for word break before max_len
    for break_char in WORD_BREAKS:
        split_pos = text.rfind(break_char, 0, max_len)
        if split_pos > 0:
            return split_pos + 1
    
    return None  # Can't split

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


def split_long_sentence(sentence, max_len=MAX_LENGTH):
    "Split a long sentence at word breaks, preferring splits near the middle"
    if len(sentence) <= max_len: return [sentence]
    words = sentence.split()
    if len(words) == 1: return [sentence]
    target = len(sentence) / 2
    best_idx,best_diff = 0,float('inf')
    current_len = 0
    for i,word in enumerate(words[:-1]):
        current_len += len(word) + 1
        diff = abs(current_len - target)
        if diff < best_diff: best_idx,best_diff = i+1,diff
    left,right = ' '.join(words[:best_idx]),' '.join(words[best_idx:])
    return split_long_sentence(left, max_len) + split_long_sentence(right, max_len)

def split_ef_fp_text(tokens, max_len=MAX_LENGTH):
    "Split long TEXT after \\fp markers inside \\ef paragraphs into multiple \\ef...\\fp sequences"
    new_tokens = []
    in_ef = False
    for i,token in enumerate(tokens):
        if token.type == UsfmTokenType.PARAGRAPH:
            in_ef = (token.marker == 'ef')
            new_tokens.append(token)
        elif token.type == UsfmTokenType.CHARACTER and token.marker == 'fp' and in_ef:
            next_token = tokens[i+1] if i+1 < len(tokens) else None
            if next_token and next_token.type == UsfmTokenType.TEXT and len(next_token.text or '') > max_len:
                new_tokens.append(token)
            else:
                new_tokens.append(token)
        elif token.type == UsfmTokenType.TEXT and in_ef and i > 0:
            prev = tokens[i-1]
            if prev.type == UsfmTokenType.CHARACTER and prev.marker == 'fp' and len(token.text or '') > max_len:
                chunks = split_text_balanced(token.text, max_len)
                new_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=chunks[0]))
                for chunk in chunks[1:]:
                    new_tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker='ef'))
                    new_tokens.append(UsfmToken(UsfmTokenType.CHARACTER, marker='fp'))
                    new_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=chunk))
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return new_tokens


def split_text_balanced(text, max_len=MAX_LENGTH):
    "Split text into balanced groups of sentences"
    if len(text) <= max_len: return [text]
    sentences = split_into_sentences(text)
    if not sentences: return [text]
    all_sentences = []
    for s in sentences:
        if len(s) > max_len: all_sentences.extend(split_long_sentence(s, max_len))
        else: all_sentences.append(s)
    if not all_sentences: return [text]
    if all([len(s) <= max_len for s in all_sentences]): return all_sentences
    groups,current_group,current_len = [],[],0
    threshold = max_len * 0.95
    for i,sent in enumerate(all_sentences):
        sent_len = len(sent)
        if current_len + sent_len + (1 if current_group else 0) > threshold:
            if current_group: groups.append(' '.join(current_group))
            current_group,current_len = [sent],sent_len
        else:
            if i < len(all_sentences) - 1:
                next_sent = all_sentences[i + 1]
                next_len = len(next_sent)
                if current_len + sent_len + next_len + 2 > threshold:
                    combined_with_current = current_len + sent_len
                    if combined_with_current > threshold * 0.7 or next_len > threshold * 0.7:
                        if current_group: groups.append(' '.join(current_group))
                        current_group,current_len = [sent],sent_len
                        continue
            current_group.append(sent)
            current_len += sent_len + (1 if len(current_group) > 1 else 0)
    if current_group: groups.append(' '.join(current_group))
    return groups


def process_file(tokenizer, input_path, max_len, method='balanced', verbosity=0, tokens=None):
    "Process a single USFM file or tokens and splitting long paragraphs."
    output_path = input_path
    if tokens is None:
        with open(input_path, 'r', encoding='utf-8') as f: usfm_text = f.read()
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
        elif token.type == UsfmTokenType.TEXT and token.text and len(token.text) > max_len:
            # Determine split marker based on current paragraph
            split_marker = get_split_marker(current_para_marker)
            split_counter[split_marker] += 1
            if method == 'sentence':
                # text_chunks = split_into_sentences(token.text)
                sys.exit()
            elif method == 'optimal':
                #text_chunks = split_text_optimally(token.text, max_len=max_len)
                sys.exit()
            elif method == 'recursive':
                # text_chunks = split_text_recursively(token.text, max_len=max_len)
                sys.exit()
            elif method == 'balanced':
                text_chunks = split_text_balanced(token.text, max_len)
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

        # Create the list of lines in memory first.
        lines = []
        for i, token in enumerate(new_tokens):
            if i > 0 and token.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE]: lines.append('\n')
            usfm_str = token.to_usfm()
            if i < len(new_tokens) - 1 and new_tokens[i + 1].type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE]: usfm_str = usfm_str.rstrip(' ')
            lines.append(usfm_str)
        
        # Create a temporary file and write the lines
        temp_path = output_path.with_suffix('.tmp')
        temp_path.write_text(''.join(lines), encoding='utf-8')
        
        # Rename the temporary file to the required filename.
        temp_path.replace(output_path)

    return split_counter


def get_books_to_process(settings, project_dir, specified_books):

    sfm_suffix = Path(settings.file_name_suffix).suffix.lower()[1:]
    #print(f"suffix is {sfm_suffix}")

    # Find all SFM/USFM files
    sfm_files = [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm", sfm_suffix]]

    # Parse books argument
    if specified_books:
        book_list = expand_book_list(specified_books)  

        # Get book IDs for found files
        ids_of_books_found = [settings.get_book_id(sfm_file.name) for sfm_file in sfm_files] 
        return [sfm_file for sfm_file in sfm_files if settings.get_book_id(sfm_file.name) in book_list]

    # No books are specified or filtered,  return all of them.
    else :
        return sfm_files


def main():
    parser = argparse.ArgumentParser(description='Split long paragraphs in USFM files')
    parser.add_argument('project', help='Paratext project name')
    parser.add_argument('--max', type=int, default=MAX_LENGTH, help='Maximum paragraph length.')
    parser.add_argument("--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'")
    parser.add_argument('--methods', nargs='+', default=['balanced'], help='Methods used to split long paragraphs, must be one of sentence, optimal, recursive, balanced.')
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity level (e.g., -v, -vv, -vvv)")
    parser.add_argument('--show_from', type=int, help="Show the tokens found at given token number. Set limit to change how many to show.")
    parser.add_argument('--show_limit', default=25, help="Set the number of tokens to show, only has an effect when --show is used.")

    args = parser.parse_args()
    print(args)

    # Set verbosity level
    verbosity = args.verbose

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
    else :
        stylesheet = UsfmStylesheet("usfm.sty")
    tokenizer = UsfmTokenizer(stylesheet)

    # Parse project settings to get book IDs
    settings = FileParatextProjectSettingsParser(project_dir).parse()

    if args.show_from is not None:

        books_to_process = get_books_to_process(settings, project_dir, args.books)
        print(f"books_to_process are {books_to_process}")
        print(f"book_to_process is {books_to_process[0]}")
        # Only show tokens from one book.
        
        # Read and tokenize the file
        with open(books_to_process[0], 'r', encoding='utf-8') as f:
            usfm_text = f.read()
    
        tokens = list(tokenizer.tokenize(usfm_text))
        show_tokena(tokens, start=args.show_from, limit=args.show_limit)
        exit()

    # for method in args.methods:
    method=args.methods[0]

    output_dir = project_dir.parent / f"{project_dir.name}_split_{method}_{args.max}"
    
    # Copying the folder ensures that all necessary files are present.
    copy_folder(project_dir, output_dir)
    
    sfm_files = get_books_to_process(settings, output_dir, args.books)

    if sfm_files:
        if verbosity >= 1:
            print(f"Will process these books:\n{sfm_files}")

    # Process each file
        for sfm_file in sfm_files:
            sfm_file_out = output_dir / sfm_file.name
            if verbosity >= 1:
                print(f"Processing {sfm_file}")
                # Read and tokenize the file
                with open(sfm_file, 'r', encoding='utf-8') as f: usfm_text = f.read()
                tokens = list(tokenizer.tokenize(usfm_text))
                ef_fp_split_tokens = split_ef_fp_text(tokens, max_len=MAX_LENGTH)
                split_counter = process_file(tokenizer, sfm_file_out, args.max, method=method, verbosity=verbosity, tokens=ef_fp_split_tokens)
                if verbosity >= 2:
                    if len(split_counter) > 0:
                        print(f"Saved {sfm_file_out} after splitting lines. {split_counter}")
                    else:
                        print(f"No changes were needed to for {sfm_file_out}")        
    if verbosity >= 1:
        print(f"Done! Processed {len(sfm_files)} books to {output_dir}")

if __name__ == '__main__':
    main()

