import argparse
import yaml
from pathlib import Path
import regex as re
from ..common.environment import SIL_NLP_ENV

# A hardcoded list of major language ISO codes from the Flores-200 benchmark.
# This list can be modified as needed.
MAJOR_LANGS = {
    'en', 'eng', 'de', 'deu', 'es', 'spa', 'fr', 'fra', 'ru', 'rus', 'zh', 'zho',
    'ar', 'ara', 'hi', 'hin', 'ja', 'jpn', 'ko', 'kor', 'bn', 'ben', 'pt', 'por',
    'id', 'ind', 'it', 'ita', 'nl', 'nld', 'pl', 'pol', 'ro', 'ron', 'tr', 'tur',
    'vi', 'vie', 'th', 'tha', 'fa', 'fas', 'he', 'heb', 'uk', 'ukr', 'el', 'ell',
    'sv', 'swe', 'fi', 'fin', 'hu', 'hun', 'cs', 'ces', 'da', 'dan', 'bg', 'bul',
    'sr', 'srp', 'sk', 'slk', 'sl', 'slv', 'et', 'est', 'lv', 'lav', 'lt', 'lit',
    'is', 'isl', 'hr', 'hrv', 'bs', 'bos', 'mk', 'mkd', 'sq', 'sqi', 'ga', 'gle',
    'cy', 'cym', 'mt', 'mlt', 'ca', 'cat', 'eu', 'eus', 'gl', 'glg', 'gd', 'gla',
    'sco', 'wa', 'wln', 'fy', 'fry', 'li', 'lim', 'oc', 'oci', 'ast', 'be', 'bel',
    'ka', 'kat', 'am', 'amh', 'ti', 'tir', 'ha', 'hau', 'ig', 'ibo', 'yo', 'yor',
    'lg', 'lug', 'sw', 'swa', 'ln', 'lin', 'bm', 'bam', 'ff', 'ful', 'so', 'som',
    'om', 'orm', 'st', 'sot', 'ts', 'tso', 'xh', 'xho', 'zu', 'zul', 'ss', 'ssw',
    'af', 'afr', 'tn', 'tsn', 'kg', 'kon', 'mg', 'mlg', 'sn', 'sna', 'co', 'cos',
    'frp', 'fur', 'gag', 'gn', 'grn', 'hif', 'ilo', 'kbp', 'km', 'khm', 'ky', 'kir',
    'lmo', 'lus', 'ml', 'mal', 'mr', 'mar', 'my', 'mya', 'nb', 'nob', 'ne', 'nep',
    'nqo', 'ny', 'nya', 'or', 'ori', 'pap', 'pms', 'ps', 'pus', 'rm', 'roh', 'sat',
    'scn', 'sd', 'snd', 'si', 'sin', 'su', 'sun', 'tg', 'tgk', 'tk', 'tuk', 'ur',
    'urd', 'uz', 'uzb', 'vec', 'war', 'yi', 'yid', 'yue'
}

def extract_lang_code(corpus_name):
    """
    Extracts a 2 or 3 letter ISO code from a corpus name that follows the
    <iso>-<filename> format. Returns the ISO code or None if the format is invalid.
    """
    match = re.match(r'^[a-z]{2,3}-', corpus_name)
    if match:
        return match.group(0)[:-1]
    return None

def combine_config_files(root_folder: Path, output_filename: str = "config.yml"):
    """
    Finds and combines all config.yml files in subfolders with specific names.
    Re-sorts languages, de-duplicates entries, and sets a new aligner.
    """
    print(f"Searching for config.yml files in subfolders of: {root_folder}")
    output_file = root_folder / output_filename

    # Use sets to collect all unique corpora
    all_major_corpora = set()
    all_minor_corpora = set()
    
    # Initialize a base config with defaults
    global_config = {
        'data': {
            'aligner': 'eflomal',
            'corpus_pairs': [{
                'type': 'train',
                'src': [],
                'trg': [],
                'mapping': 'many_to_many',
                'test_size': 0,
                'val_size': 0
            }]
        }
    }
    
    found_first_config = False

    # Find all config.yml files in subdirectories
    for config_file in root_folder.rglob('**/config.yml'):
        parent_dir_name = config_file.parent.name
        
        # Check if the parent directory name starts with one of the specified prefixes
        if not (parent_dir_name.lower().startswith('align') or 
                parent_dir_name.lower().startswith('analyze') or 
                parent_dir_name.lower().startswith('analyse')):
            continue
        
        print(f"Found config file: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            if not found_first_config and 'data' in config and 'tokenize' in config['data']:
                global_config['data']['tokenize'] = config['data']['tokenize']
                found_first_config = True

            if 'data' in config and 'corpus_pairs' in config['data']:
                for pair in config['data']['corpus_pairs']:
                    src_list = set(pair.get('src', []))
                    trg_list = set(pair.get('trg', []))
                    
                    # Handle cases where 'src' or 'trg' are single strings, not lists
                    src_items = pair.get('src', [])
                    if isinstance(src_items, str):
                        src_items = [src_items]
                    src_list = set(src_items)
                    
                    trg_items = pair.get('trg', [])
                    if isinstance(trg_items, str):
                        trg_items = [trg_items]
                    trg_list = set(trg_items)

                    all_corpora = src_list.union(trg_list)
                    
                    for corpus in all_corpora:
                        lang_code = extract_lang_code(corpus)
                        if lang_code: # Only process if a valid lang_code was extracted
                            if lang_code in MAJOR_LANGS:
                                all_major_corpora.add(corpus)
                            else:
                                all_minor_corpora.add(corpus)
                        else:
                            print(f"Skipping invalid corpus name: {corpus} in {config_file}")

        except Exception as e:
            print(f"Error processing {config_file}: {e}")
    
    # Update the src corpora with all files.
    global_config['data']['corpus_pairs'][0]['src'] = sorted(list(all_major_corpora))
    global_config['data']['corpus_pairs'][0]['src'] += sorted(list(all_minor_corpora))
    
    # Update the trg corpora with only the minor language corpora.
    global_config['data']['corpus_pairs'][0]['trg'] = sorted(list(all_minor_corpora))

    # Write the combined config to a new file in the root folder
    output_path = root_folder / 'combined_config.yml'
    try:
        with open(output_path, 'w') as f:
            yaml.dump(global_config, f, sort_keys=False)
        print(f"\nSuccessfully wrote combined configuration to: {output_path}")
    except Exception as e:
        print(f"Failed to write the combined config file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine multiple config.yml files into one.'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='The root folder to search for config.yml files.'
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="config.yml",
        help="Output filename for the combined file. The default is config.yml.",
    )

    args = parser.parse_args()
    folder = Path(args.folder)

    if not folder.is_dir():
        folder = Path(SIL_NLP_ENV.mt_experiments_dir) / args.folder    
    
    if not folder.is_dir():
        print(f"Error: Couldn't find {args.folder} or {folder}.")
    else:
        combine_config_files(folder, args.output_filename)

