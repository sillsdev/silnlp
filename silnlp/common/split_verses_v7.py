import argparse
import logging
import shutil
from collections import Counter
from pathlib import Path

from aiohttp import TraceDnsCacheHitParams
from machine.corpora import FileParatextProjectSettingsParser, UsfmStylesheet, UsfmToken, UsfmTokenizer, UsfmTokenType, UsfmParser, UsfmElementType

from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
from .paratext import get_project_dir

from .book_args import expand_book_list, get_sfm_files_to_process, get_epilog, add_books_argument

LOGGER = logging.getLogger(__package__ + ".split_verses_v7")

MAX_LENGTH = 200

# from .check_books import group_bible_books
VALID_CANONS = ["OT", "NT", "DT"]
VALID_BOOKS = OT_CANON + NT_CANON + DT_CANON

SENTENCE_ENDS = '.!?'
WORD_BREAKS = " ,;:-"
CLOSE_QUOTES = '"\'"'

ELEMENT_TO_TOKEN_TYPE = {
    UsfmElementType.PARA: UsfmTokenType.PARAGRAPH,
    UsfmElementType.NOTE: UsfmTokenType.NOTE,
    UsfmElementType.CHAR: UsfmTokenType.CHARACTER,
}

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


def split_text(text, max_len):
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


# def split_text_token(parser, max_len):
#     "Split long text into parts with the appropriate para/char markers"

#     chunks = split_text(parser.state.token.text, max_len)
#     if len(chunks) == 1: 
#         print(f"Warning: text chunk passed to split_text_token was not longer than {max_len}:\n{parser.state.token.text}")
#         return [parser.state.token]  # No change to this token.

#     elif len(chunks) > 1:
#         result = [UsfmToken(UsfmTokenType.TEXT, text=chunks[0])]
#         if len(parser.state.stack) == 3 and parser.state.stack[0].type == UsfmElementType.PARA and parser.state.stack[1].type == UsfmElementType.NOTE and parser.state.stack[2].type == UsfmElementType.CHAR:
#             for chunk in chunks[1:]:
#                 result.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker=parser.state.stack[0].marker))
#                 result.append(UsfmToken(UsfmTokenType.NOTE, marker=parser.state.stack[1].marker))
#                 result.append(UsfmToken(UsfmTokenType.CHARACTER, marker=parser.state.stack[2].marker))
#                 result.append(UsfmToken(UsfmTokenType.TEXT, text=chunk))
#         elif len(parser.state.stack) == 2 and parser.state.stack[0].type == UsfmElementType.PARA and parser.state.stack[1].type == UsfmElementType.CHAR:
#             for chunk in chunks[1:]:
#                 result.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker=parser.state.stack[0].marker))
#                 result.append(UsfmToken(UsfmTokenType.CHARACTER, marker=parser.state.stack[1].marker))
#                 result.append(UsfmToken(UsfmTokenType.TEXT, text=chunk))
#         elif len(parser.state.stack) == 1 and parser.state.stack[0].type == UsfmElementType.PARA:
#             for chunk in chunks[1:]:
#                 result.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker=parser.state.stack[0].marker))
#                 result.append(UsfmToken(UsfmTokenType.TEXT, text=chunk))
#         else:
#             print(f"Warning, don't know how to handle this parser.state.stack: {parser.state.stack} in split_text_token.")
#     return result


def split_text_token(parser, max_len):
    "Split long text into parts with the appropriate para/char markers"

    chunks = split_text(parser.state.token.text, max_len)
    if len(chunks) == 1: 
        print(f"Warning: text chunk passed to split_text_token was not longer than {max_len}:\n{parser.state.token.text}")
        return [parser.state.token]  # No change to this token.

    stack = parser.state.stack
    if not stack or stack[0].type != UsfmElementType.PARA:
        print(f"Warning: expected PARA as first stack element, got: {stack} with token: {parser.state.token} on line number {parser.state.token.line_number}")
        return [parser.state.token]

    result = [UsfmToken(UsfmTokenType.TEXT, text=chunks[0])]
    for chunk in chunks[1:]:
        for elem in stack:
            token_type = ELEMENT_TO_TOKEN_TYPE.get(elem.type)
            if token_type: result.append(UsfmToken(token_type, marker=elem.marker))
            else: print(f"Warning: unknown element type in stack: {elem.type}")
        result.append(UsfmToken(UsfmTokenType.TEXT, text=chunk))
    return result

def strip_eol_spaces(string):
    sample = string[:600]
    print(f"sample string is : {sample}")
    
    print(f"\nString is a {type(string)}. Items look like this with _ in place of spaces:")
    sample_view = sample.replace(' ','_').replace('\r', '\\r').replace('\n','\\n')
    print(sample_view)
    
    print(f"\nString with ' \\n' replaced by '\\n' is:")
    string_stripped = string.replace(' \n','\n')

    print(f"\nAfter removing eol_spaces string is a {type(string_stripped)}. Items look like this with _ in place of spaces:")
    new_sample_view = string_stripped[:600].replace(' ','_').replace('\r', '\\r').replace('\n','\\n')
    print(new_sample_view)
    print()
    print(string_stripped[:600])

    return string_stripped
    
    


def main():
    parser = argparse.ArgumentParser(description="Split long paragraphs in USFM files",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=get_epilog(),
    )
    parser.add_argument("project", type=str, help="Paratext project name - the files in this folder will be modified in place.")
    add_books_argument(parser)
    parser.add_argument("--max", type=int, default=MAX_LENGTH, help="Maximum paragraph length.")

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
    books = expand_book_list(args.books)
    sfm_files =  get_sfm_files_to_process(project_dir, books)

    output_dir = project_dir.parent / f"{project_dir.name}_split_{args.max}"
    
    # Copying the folder ensures that all necessary files are present.
    #shutil.copytree(project_dir, output_dir, dirs_exist_ok=True)

    # Test with one file:

    # Process each file
    for sfm_file in sfm_files:
        
        print(f"Processing sfm_file: {sfm_file}")
        output_tokens = []
        
        with open(sfm_file, 'r', encoding='utf-8') as f: usfm_text = f.read()
        parser = UsfmParser(usfm_text, stylesheet=settings.stylesheet, versification=settings.versification)
        
        while parser.process_token():
            if parser.state.token.type == UsfmTokenType.TEXT and parser.state.token.get_length() > MAX_LENGTH:
                output_tokens.extend(split_text_token(parser, max_len=args.max))
            else:
                output_tokens.append(parser.state.token)

        usfm_out = [token.to_usfm(include_newlines=True).replace('\r\n', '\n') for token in output_tokens]

        usfm_str = ''.join(usfm_out).replace(' \n','\n')
        usfm_stripped = usfm_str.replace(' \n','\n')
        
        output_sfm_file = output_dir / sfm_file.name
        with open(output_sfm_file, 'w', encoding='utf-8') as f: f.write(usfm_stripped)
        
        # with open(output_sfm_file, 'r', encoding='utf-8') as f: usfm_out_text = f.read()
        # out_parser = UsfmParser(usfm_out_text, stylesheet=settings.stylesheet, versification=settings.versification)
        # over_long_texts = dict()
        # while out_parser.process_token():
        #     token_len = parser.state.token.get_length()
        #     if out_parser.state.token.type == UsfmTokenType.TEXT and  token_len > MAX_LENGTH:
        #         over_long_texts[output_sfm_file] = {'lineno':parser.state.token.line_number, 'length':parser.state.token.get_length()}
        #         print(f"SFM output file {output_sfm_file} has the following long texts:")
        #         print(f"{over_long_texts}")
        #         exit()

        # print(f"SFM output file {output_sfm_file} has the following long texts:")
        # print(f"{over_long_texts}")
                    
    print(f"Done! Processed {len(sfm_files)} books in {output_dir}")


if __name__ == "__main__":
    main()



