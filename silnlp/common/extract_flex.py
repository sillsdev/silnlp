from argparse import ArgumentParser
from pathlib2 import Path
from typing import Optional
import xml.etree.ElementTree as ET
from tqdm import tqdm

def extract_flex(xml:ET.ElementTree, name:str, output_folder:Path, only_languages:'Optional[set[str]]'=None):
    print("Extracting text corpora from FLEx data...")
    all_phrase_elements = xml.findall('interlinear-text/paragraphs/paragraph/phrases/phrase')
    lines = 0
    print(only_languages)
    text_per_lang: dict[str,str] = {}
    for phrase_element in tqdm(all_phrase_elements):
        items = phrase_element.findall('item')
        languages_added = set()
        for item in items:
            if item.attrib['type'] in ['txt', 'gls']:
                language_code = item.attrib['lang']
                if only_languages is not None and language_code not in only_languages:
                    continue
                languages_added.add(language_code)
                if language_code not in text_per_lang:
                    text_per_lang[language_code] = "\n"*(lines-1)
                text_per_lang[language_code] += item.text.strip().strip("'").strip() + "\n" if item.text is not None else "\n"
                lines +=1
        for language in set(text_per_lang.keys()) - languages_added:
            text_per_lang[language] += "\n"
    for language, text in text_per_lang.items():
        with open(output_folder / f"{language}-{name}.txt", 'w') as f:
            f.write(text)
    print(f"Extracted {lines} lines of text")

def main():
    parser = ArgumentParser(description="Extracts text corpora from FLEx XML data")
    parser.add_argument("xml", nargs=1, help="FLEx xml data")
    parser.add_argument(    
        "--output-folder", required=False, nargs=1, default="./", help="The folder in which to save the extracted texts (defaults to the current directory)"
    )
    parser.add_argument(    
        "--only-langs", required=False, nargs="+", default=set(), help="Only extract data for these language codes"
    )
    args = parser.parse_args()
    xml = None
    xml_path = Path(args.xml[0])
    print("Parsing xml file...")
    with open(xml_path, 'r') as f:
        xml = ET.fromstring(f.read())
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        raise ValueError(f"Output directory {str(output_folder)} must exist")
    extract_flex(xml, xml_path.name.split('.')[0] if '.' in xml_path.name else xml_path.name, output_folder, args.only_langs)

if __name__ == "__main__":
    main()