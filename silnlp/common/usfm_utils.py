"""
Print out all paragraph and character markers for a book
To use set book, fpath, and out_path. fpath should be a path to a book in a Paratext project
"""
import argparse
from pathlib import Path
from machine.corpora import ParatextTextCorpus
from machine.corpora import FileParatextProjectSettingsParser, UsfmFileText, UsfmTokenizer, UsfmTokenType


def main() -> None:
    # parser = argparse.ArgumentParser(description="Extracts markers from Paratext projects")
    # parser.add_argument("projects", nargs="+", metavar="name", help="Paratext projects")
    # parser.add_argument(
    #     "--include", metavar="books", nargs="+", default=[], help="The books to include; e.g., 'NT', 'OT', 'GEN'"
    # )
    # parser.add_argument(
    #     "--exclude", metavar="books", nargs="+", default=[], help="The books to exclude; e.g., 'NT', 'OT', 'GEN'"
    # )

    # corpus = ParatextTextCorpus("data/WEB-PT")


    # this assumes fpath is a book in a Paratext project folder and that out_path is a file.
    book = "MAT"
    fpath = Path("S:/Paratext/projects/aArp_2024_07_03/41MATaArp.SFM")
    out_path = Path("F:/GitHub/temp/markers.txt")
    sentences_file = Path("F:/GitHub/temp/sentences.txt")
    
    settings = FileParatextProjectSettingsParser(fpath.parent).parse()
    file_text = UsfmFileText(
        settings.stylesheet,
        settings.encoding,
        book,
        fpath,
        settings.versification,
        include_markers=True,
        include_all_text=True,
        project=settings.name,
    )

    vrefs = []
    usfm_markers = []
    usfm_tokenizer = UsfmTokenizer(settings.stylesheet)
    with sentences_file.open("w", encoding=settings.encoding) as f:       
        for sent in file_text:
            f.write(f"{sent}\n")
            if len(sent.ref.path) > 0 and sent.ref.path[-1].name == "rem":
                continue

            vrefs.append(sent.ref)
            usfm_markers.append([])
            usfm_toks = usfm_tokenizer.tokenize(sent.text.strip())
            
            ignore_scope = None
            to_delete = ["fig"]
            for j, tok in enumerate(usfm_toks):
                if ignore_scope is not None:
                    if tok.type == UsfmTokenType.END and tok.marker[:-1] == ignore_scope.marker:
                        ignore_scope = None
                elif tok.type == UsfmTokenType.NOTE or (tok.type == UsfmTokenType.CHARACTER and tok.marker in to_delete):
                    ignore_scope = tok
                elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHARACTER, UsfmTokenType.END]:
                    usfm_markers[-1].append(tok.marker)

    with out_path.open("w", encoding=settings.encoding) as f:
        for ref, markers in zip(vrefs, usfm_markers):
            f.write(f"{ref} {markers}\n")



if __name__ == "__main__":
    main()
