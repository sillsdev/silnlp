from pathlib import Path

from machine.corpora import FileParatextProjectSettingsParser, UsfmFileText, UsfmTokenizer, UsfmTokenType

# Marker "type" is as defined by the UsfmTokenType given to tokens by the UsfmTokenizer,
# which mostly aligns with a marker's StyleType in the USFM stylesheet
CHARACTER_TYPE_EMBEDS = ["fig", "fm", "jmp", "rq", "va", "vp", "xt", "xtSee", "xtSeeAlso"]
PARAGRAPH_TYPE_EMBEDS = ["lit", "r", "rem"]
NON_NOTE_TYPE_EMBEDS = CHARACTER_TYPE_EMBEDS + PARAGRAPH_TYPE_EMBEDS


def main() -> None:
    """
    Print out all paragraph and character markers for a book
    To use set book, fpath, and out_path. fpath should be a path to a book in a Paratext project
    """

    book = "MAT"
    fpath = Path("")
    out_path = Path("")
    sentences_file = Path("")

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
            if len(sent.ref.path) > 0 and sent.ref.path[-1].name in PARAGRAPH_TYPE_EMBEDS:
                continue

            vrefs.append(sent.ref)
            usfm_markers.append([])
            usfm_toks = usfm_tokenizer.tokenize(sent.text.strip())

            ignore_scope = None
            for tok in usfm_toks:
                if ignore_scope is not None:
                    if tok.type == UsfmTokenType.END and tok.marker[:-1] == ignore_scope.marker:
                        ignore_scope = None
                elif tok.type == UsfmTokenType.NOTE or (
                    tok.type == UsfmTokenType.CHARACTER and tok.marker in CHARACTER_TYPE_EMBEDS
                ):
                    ignore_scope = tok
                elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHARACTER, UsfmTokenType.END]:
                    usfm_markers[-1].append(tok.marker)

    with out_path.open("w", encoding=settings.encoding) as f:
        for ref, markers in zip(vrefs, usfm_markers):
            f.write(f"{ref} {markers}\n")


if __name__ == "__main__":
    main()
