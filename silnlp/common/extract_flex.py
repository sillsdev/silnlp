import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from typing import Optional

from pathlib2 import Path
from tqdm import tqdm


def extract_flex(
    xml: ET.ElementTree,
    name: str,
    output_folder: Path,
    only_languages: "Optional[set[str]]" = None,
    only_textids: "Optional[set[str]]" = None,
):
    print("Extracting text corpora from FLEx data...")
    lines = 0
    text_per_lang: dict[str, str] = {}
    text_ids_found = set()
    text_elements = xml.findall("interlinear-text")
    for text_element in tqdm(text_elements):
        text_id_elements = list(filter(lambda i: i.attrib["type"] == "title", text_element.findall("item")))
        if only_textids is not None and len(only_textids) > 0 and (len(text_id_elements) == 0 or not any(t.text in only_textids for t in text_id_elements)):
            continue
        if only_textids is not None and len(only_textids) > 0 and len(text_id_elements) > 0:
            text_ids_found.add(text_id_elements[0].text)
        phrase_elements = text_element.findall("paragraphs/paragraph/phrases/phrase")
        for phrase_element in phrase_elements:
            line_added = False
            items = phrase_element.findall("item")
            languages_added = set()
            for item in items:
                if item.attrib["type"] in ["txt", "gls"]:
                    language_code = item.attrib["lang"]
                    if only_languages is not None and len(only_languages) > 0 and language_code not in only_languages:
                        continue
                    line_added = True
                    languages_added.add(language_code)
                    if language_code not in text_per_lang:
                        text_per_lang[language_code] = "\n" * lines
                    text_per_lang[language_code] += (
                        item.text.strip().strip("\u200e").strip("'").strip() + "\n" if item.text is not None else "\n"
                    )
            if line_added:
                for language in set(text_per_lang.keys()) - languages_added:
                    text_per_lang[language] += "\n"
                lines += 1
    for language, text in text_per_lang.items():
        with open(output_folder / f"{language}-{name}.txt", "w") as f:
            f.write(text)
    if only_textids is not None and len(only_textids) > 0 and len(only_textids) != len(text_ids_found):
        print(f"Unable to find text id(s): {','.join(set(only_textids) - text_ids_found)}")

    print(f"Extracted {lines} lines of text")


def main():
    parser = ArgumentParser(description="Extracts text corpora from FLEx XML data")
    parser.add_argument("xml", nargs=1, help="FLEx xml data")
    parser.add_argument(
        "--output-folder",
        required=False,
        nargs=1,
        default="./",
        help="The folder in which to save the extracted texts (defaults to the current directory)",
    )
    parser.add_argument(
        "--only-langs", required=False, nargs="+", default=set(), help="Only extract data for these language codes"
    )
    parser.add_argument(
        "--text-ids", required=False, nargs="+", default=set(), help="Only extract data from texts with these text ids"
    )
    args = parser.parse_args()
    xml = None
    xml_path = Path(args.xml[0])
    print("Parsing xml file...")
    with open(xml_path, "r") as f:
        xml = ET.fromstring(f.read())
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        raise ValueError(f"Output directory {str(output_folder)} must exist")
    extract_flex(
        xml,
        xml_path.name.split(".")[0] if "." in xml_path.name else xml_path.name,
        output_folder,
        args.only_langs,
        args.text_ids,
    )


if __name__ == "__main__":
    main()
