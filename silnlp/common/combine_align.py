import argparse
import yaml
from pathlib import Path
import regex as re
from ..common.environment import SIL_NLP_ENV

# List of keywords to exclude from filenames
EXCLUDED_KEYWORDS = ["XRI", "_AI", "train"]

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
    Filters out older files with dates and files with excluded keywords.                                                                     
    """
    print(f"Searching for config.yml files in subfolders of: {root_folder}")

    # Dictionary to hold corpus names, grouped by language code
    corpus_by_lang = {}
    
    # Initialize a base config with defaults
    global_config = {
        'data': {
            'aligner': 'eflomal',
            'corpus_pairs': [{
                'mapping': 'many_to_many',
                'src': [],
                'trg': [],
                'test_size': 0,
                'type': 'train',
                'val_size': 0
            }]
        }
    }
    tokenize_setting = None


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

            if tokenize_setting is None and 'data' in config and 'tokenize' in config['data']:
                tokenize_setting = config['data']['tokenize']

            if 'data' in config and 'corpus_pairs' in config['data']:
                for pair in config['data']['corpus_pairs']:
                    # Handle cases where 'src' or 'trg' are single strings, not lists
                    src_items = pair.get('src', [])
                    if isinstance(src_items, str):
                        src_items = [src_items]
                    
                    trg_items = pair.get('trg', [])
                    if isinstance(trg_items, str):
                        trg_items = [trg_items]
                    
                    all_corpora = set(src_items).union(set(trg_items))

                    for corpus in all_corpora:
                        # Filter out files with excluded keywords
                        if any(keyword.lower() in corpus.lower() for keyword in EXCLUDED_KEYWORDS):
                            print(f"Excluding file due to keyword: {corpus}")
                            continue

                        lang_code = extract_lang_code(corpus)
                        if lang_code:
                            if lang_code not in corpus_by_lang:
                                corpus_by_lang[lang_code] = {'dated': [], 'undated': []}

                            # Extract date from filename if present
                            date_match = re.search(r'_(\d{4}_\d{2}_\d{2})', corpus)
                            if date_match:
                                date_str = date_match.group(1)
                                                               
                                corpus_by_lang[lang_code]['dated'].append((date_str, corpus))
                            else:
                                corpus_by_lang[lang_code]['undated'].append(corpus)
                        else:
                            print(f"Skipping invalid corpus name: {corpus}")

        except Exception as e:
            print(f"Error processing {config_file}: {e}")
    
    # Filter for the most recent file for each language and include all undated files
    final_corpora = set()
    for lang_code, corpora_dict in corpus_by_lang.items():
        # Keep all undated files
        for corpus in corpora_dict['undated']:
            final_corpora.add(corpus)
        
        # Keep only the most recent dated file, if any exist
        if corpora_dict['dated']:
            corpora_dict['dated'].sort(key=lambda x: x[0], reverse=True)
                                                   
            final_corpora.add(corpora_dict['dated'][0][1])

    # Separate filtered corpora into major and minor languages
    major_corpora = set()
    minor_corpora = set()
    for corpus in final_corpora:
        lang_code = extract_lang_code(corpus)
        if lang_code and lang_code in MAJOR_LANGS:
            major_corpora.add(corpus)
        else:
            minor_corpora.add(corpus)

    # The new 'src' list is the sorted combination of major and minor languages
    # The new 'trg' list is the sorted list of minor languages
    global_config['data']['corpus_pairs'][0]['src'] = sorted(list(major_corpora)) + sorted(list(minor_corpora))
    global_config['data']['corpus_pairs'][0]['trg'] = sorted(list(minor_corpora))

    # Add tokenize setting if it was found
    if tokenize_setting is not None:
        global_config['data']['tokenize'] = tokenize_setting
    # Write the combined config to a new file in the root folder
    output_path = root_folder / output_filename
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