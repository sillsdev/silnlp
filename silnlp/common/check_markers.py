import argparse
from collections import Counter
from pathlib import Path
from typing import List, Counter

from machine.corpora import FileParatextProjectSettingsParser, UsfmStylesheet, UsfmToken, UsfmTokenizer, UsfmTokenType
#import machine.corpora.usfm_stylesheet as uss
#print(uss.__file__)

from .paratext import get_project_dir
from .split_verses_v3 import get_books_to_process

def check_end_markers(tokens, filepath, stylesheet):
    "Check that all CHARACTER/NOTE markers are properly closed before PARAGRAPH/CHAPTER/VERSE"
    open_markers = []
    for tok in tokens:
        if tok.type in [UsfmTokenType.CHARACTER, UsfmTokenType.NOTE]:
            tag = stylesheet.get_tag(tok.marker)
            if tag.end_marker: open_markers.append((tok.marker, tok.line_number, tag.end_marker))
        elif tok.type == UsfmTokenType.END:
            if not open_markers:
                print(f"{filepath}:{tok.line_number}: Unexpected end marker \\{tok.marker} with no matching opening")
                exit(1)
            expected_marker = open_markers[-1][0]
            if tok.marker != open_markers[-1][2]:
                print(f"{filepath}:{tok.line_number}: End marker \\{tok.marker} doesn't match opening \\{expected_marker}")
                exit(1)
            open_markers.pop()
        elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE]:
            if open_markers:
                marker,line,end = open_markers[-1]
                print(f"{filepath}:{line}: Missing end marker \\{end} for \\{marker}")
                exit(1)
    if open_markers:
        marker,line,end = open_markers[-1]
        print(f"{filepath}:{line}: Missing end marker \\{end} for \\{marker} at end of file")
        exit(1)


def get_token_data(settings, tokens: List[UsfmToken]):
    marker_count = Counter()
    results = dict()

    for token in tokens:
        marker_count.update([token.marker])
        if token.marker in results:
            continue
        else:
            results[token.marker] = (token, settings.stylesheet.get_tag(str(token.marker)).occurs_under)
        
    return results, marker_count


def get_types(settings, results):
    ou_types = dict()
    
    for marker, values in results.items():

        ou_styletype_count = Counter()
        ou_texttype_count = Counter()
        _, ou_markers = values
        
        if ou_markers:
            #print(f"values are split into {_} and {ou_markers}")
            for ou_marker in ou_markers:
                ou_styletype_count.update([settings.stylesheet.get_tag(ou_marker).style_type.name[:4]])
                ou_texttype_count.update([settings.stylesheet.get_tag(ou_marker).text_type.name])

            ou_types[marker] = (ou_styletype_count, ou_texttype_count)             
    return ou_types

def get_style_types(settings, results):
    ou_styletypes = dict()
    for marker, values in results.items():
        ou_type_count = Counter()
        _, ou_markers = values
        if ou_markers:
            #print(f"values are split into {_} and {ou_markers}")
            for ou_marker in ou_markers:
                ou_type_count.update([settings.stylesheet.get_tag(ou_marker).style_type.name[:4]])
            ou_styletypes[marker] = ou_type_count
    return ou_styletypes



def main():
    parser = argparse.ArgumentParser(description="Split long paragraphs in USFM files")
    parser.add_argument("project", help="Paratext project name")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
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

    # All of this is done already by settings = FileParatextProjectSettingsParser(project_dir).parse()
    # custom_sty_path = project_dir / "custom.sty"
    # if custom_sty_path.is_file():
    #     stylesheet = UsfmStylesheet("usfm.sty", custom_sty_path)
    # else:
    #     stylesheet = UsfmStylesheet("usfm.sty")
    # tokenizer = UsfmTokenizer(stylesheet)

    # Parse project settings to get book IDs
    settings = FileParatextProjectSettingsParser(project_dir).parse()
#    print(dir(settings))
#    print(settings.stylesheet)
    tokenizer = UsfmTokenizer(settings.stylesheet)

    # for marker in ['c','v']:
    #     print(f"Marker {marker}")
    #     print(f"style_type:   {settings.stylesheet.get_tag(marker).style_type}")
    #     print(f"text_type:    {settings.stylesheet.get_tag(marker).text_type}")
    #     print(f"rank:         {settings.stylesheet.get_tag(marker).rank}")
    #     print(f"occurs_under: {settings.stylesheet.get_tag(marker).occurs_under}")
    #     print(f"end_marker: {settings.stylesheet.get_tag(marker).end_marker}")
    #     print(f"Stylesheet.get_tag dir: {dir(settings.stylesheet.get_tag(marker))}")
    
    sfm_files = get_books_to_process(settings, project_dir, args.books)
    #print(f"Stylesheet dir: {dir(settings.stylesheet)}")
    
    # print()
    # print(dir(settings.stylesheet._tags))
    # print()
    # print(settings.stylesheet._tags)
    # print()
    # print(settings.stylesheet.get_tag('ft'))
    # print(dir(settings.stylesheet.get_tag('ft')))
    # print(settings.stylesheet.get_tag('ft').occurs_under)
    # exit()

    # Check each file
    for sfm_file in sfm_files:
        
        # Read and tokenize the file
        with open(sfm_file, "r", encoding="utf-8") as f: usfm_text = f.read()
        tokens = list(tokenizer.tokenize(usfm_text))
        results, marker_count = get_token_data(settings, tokens)
        # print("Results are:")
        # print(results)
        # print("Counts are:")
        # print(marker_count)
        # marker = 'ft'
        # print(f"Marker {marker} Occurs_under: {settings.stylesheet.get_tag(marker).occurs_under}")
        # print(f"Marker {marker} StyleType:    {settings.stylesheet.get_tag(marker).style_type}")

        #print(f"Here are the occurs_under types for each marker:")
        ou_types = get_types(settings, results)
        #print(ou_types, type(ou_types))
        #exit()
        
        for marker, values in ou_types.items():
            styletypes, texttypes = values
            #print(styletypes, type(styletypes))
            #print(texttypes, type(texttypes))
            style_type = settings.stylesheet.get_tag(marker).style_type.name if settings.stylesheet.get_tag(marker).style_type.name else ''
            end_marker = settings.stylesheet.get_tag(marker).end_marker if settings.stylesheet.get_tag(marker).end_marker else ''
                
            print(f"{marker:7s} | {end_marker[:4]:4s} | {style_type[:4]:4s} | {str(styletypes):50s} | {str(texttypes)}")

        # for token in tokens:
        #     if token.marker == 'ft':
        #         print(dir(token))
        #         print(token.marker, token.data, token.text, token.end_marker, token.get_length())
        #         print(dir(token.get_attribute('occursunder')))
        #         print(type(token.get_attribute('occursunder')))
        #         print(token.get_attribute('occursunder'))
        #         print(f"token.attributes = {token.attributes}")
        #         print(f"token.get_attribute('occursunder') = {token.get_attribute('occursunder')}")
        #         exit()
        #         print(f"ft token occursunder : token.occursunder")
                
        #check_end_markers(tokens, sfm_file, settings.stylesheet)

if __name__ == "__main__":
    main()
