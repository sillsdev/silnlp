import argparse
import logging
import shutil
import time
from pathlib import Path

from machine.corpora import FileParatextProjectSettingsParser, UsfmStylesheet, UsfmToken, UsfmTokenizer, UsfmTokenType, UsfmParser

from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
from .paratext import get_project_dir

# TODO refactor to use book_args
from .book_args import expand_book_list, get_sfm_files_to_process, get_epilog, add_books_argument

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


class ParagraphCollector:
    """Walks tokens via UsfmParser, collecting paragraphs and identifying split points."""
    
    def __init__(self, usfm_text, settings, max_len=200):
        self.parser = UsfmParser(usfm_text, stylesheet=settings.stylesheet, versification=settings.versification)
        self.max_len = max_len
        self.result_tokens = []                  # Output: modified token stream
        self.last_char_tags = []
        self.split_points = []
        self.state = self.parser.state
        self.current_para = []                   # Tokens in current paragraph
        self.current_text_len = 0                # Accumulated text length
        self.current_nonpara_marker = None       # Which CHAR marker we're in
        self.current_nonpara_len = 0             # Text length inside current CHAR marker
        self.open_nonpara_markers = []           # Track repeated markers for implicit splits
        self.exclude_markers = {'id', 'c', 'v'}  # Never split these
    
    @property
    def token(self): return self.state.token

    def process(self):
        "Main loop: iterate through all tokens, split as needed, return modified tokens."
        while self.parser.process_token():
            if self.token.type == UsfmTokenType.PARAGRAPH:
                self.on_paragraph_start()
            elif self.token.type == UsfmTokenType.TEXT:
                self.on_text()
            elif self.token.type in (UsfmTokenType.CHARACTER, UsfmTokenType.NOTE, UsfmTokenType.MILESTONE):
                self.on_nonpara_marker()
            elif self.token.type == UsfmTokenType.END:
                self.on_end_marker()
            else:
                self.current_para.append(self.token)
        
        self.flush_paragraph()
        return self.result_tokens
    

    def on_paragraph_start(self):
        self.flush_paragraph()
        self.current_para = [self.token]
        self.split_points = []
        self.current_text_len = 0
        self.current_nonpara_len = 0
        self.current_nonpara_marker = None
        self.open_nonpara_markers = []


    def on_text(self):
        """Accumulate text length; check if split needed."""
        self.current_para.append(self.token)
        text_len = len(self.token.text or '')
        if self.state.char_tags or self.state.note_tag:
            self.current_nonpara_len += text_len
            self.last_char_tags = [UsfmToken(type=UsfmTokenType.CHARACTER, marker=c.marker) for c in self.state.char_tags]
        else:
            self.current_text_len += text_len

            
    def on_nonpara_marker(self):
        "Check for repeated marker (implicit close = split point)."
        if self.token.marker in [e.marker for e in self.state.stack]:
            self.split_points.append((len(self.current_para), self.current_nonpara_len, list(self.state.char_tags)))
            self.current_nonpara_len = 0
        self.current_para.append(self.token)
        self.current_nonpara_marker = self.token.marker

    def on_end_marker(self):
        "Potential split point when returning to para level."
        self.current_para.append(self.token)
        self.split_points.append((len(self.current_para), self.current_text_len + self.current_nonpara_len, list(self.state.char_tags)))
        self.current_nonpara_len = 0
        self.current_nonpara_marker = None

    def flush_paragraph(self):
        "Apply splits to current_para, append to result_tokens."
        if not self.current_para: return
        
        para_marker = self.current_para[0]
        if para_marker.marker in self.exclude_markers:
            self.result_tokens.extend(self.current_para)
            return
        
        total_len = self.current_text_len + self.current_nonpara_len
        if total_len <= self.max_len:
            self.result_tokens.extend(self.current_para)
            return
        
        if self.split_points:
            parts = [(idx, length) for idx, length, _ in self.split_points]
            splits = optimal_grouping(parts, self.max_len)
            new_paras = split_paragraph_tokens(self.current_para, parts, splits)
            for j, new_para in enumerate(new_paras):
                text = get_paragraph_text(new_para)
                if len(text) > self.max_len:
                    char_tags = self.split_points[splits[j-1]][2] if j > 0 and j-1 < len(splits) else []
                    chunks = split_long_text(text, self.max_len)
                    for chunk in chunks:
                        self.result_tokens.append(UsfmToken(type=UsfmTokenType.PARAGRAPH, marker=para_marker.marker))
                        for ct in char_tags: self.result_tokens.append(UsfmToken(type=UsfmTokenType.CHARACTER, marker=ct.marker))
                        self.result_tokens.append(UsfmToken(type=UsfmTokenType.TEXT, text=chunk))
                else:
                    self.result_tokens.extend(new_para)
        else:
            text = get_paragraph_text(self.current_para)
            chunks = split_long_text(text, self.max_len)
            for chunk in chunks:
                self.result_tokens.append(UsfmToken(type=UsfmTokenType.PARAGRAPH, marker=para_marker.marker))
                self.result_tokens.append(UsfmToken(type=UsfmTokenType.TEXT, text=chunk))

        text = get_paragraph_text(self.current_para)
        self._emit_text_splits(para_marker, self.last_char_tags, text)

    def _emit_text_splits(self, para_marker, char_tags, text):
        chunks = split_long_text(text, self.max_len)
        for chunk in chunks:
            self.result_tokens.append(UsfmToken(type=UsfmTokenType.PARAGRAPH, marker=para_marker.marker))
            self.result_tokens.extend(char_tags)
            self.result_tokens.append(UsfmToken(type=UsfmTokenType.TEXT, text=chunk))


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
    shutil.copytree(project_dir, output_dir, dirs_exist_ok=True)

    for sfm_file in sfm_files:
        output_sfm_file = output_dir / sfm_file.name
        with open(sfm_file, 'r', encoding='utf-8') as f: usfm_text = f.read()       
        print(f"Processing input file {sfm_file} output to {output_sfm_file}")
        split_tokens = ParagraphCollector(usfm_text, settings, max_len=args.max).process()
        usfm_out = [token.to_usfm(include_newlines=True).replace('\r\n', '\n') for token in split_tokens]

        with open(output_sfm_file, 'w', encoding='utf-8') as f: f.write(''.join(usfm_out))
                    
    print(f"Done! Processed {len(sfm_files)} books in {output_dir}")


if __name__ == "__main__":
    main()



