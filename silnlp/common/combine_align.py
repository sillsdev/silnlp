import argparse
import yaml
from pathlib import Path
import regex as re
from .environment import SIL_NLP_ENV

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

# List of keywords to exclude from filenames
EXCLUDED_KEYWORDS = ["XRI", "AI"]

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
    Also filters out older files with dates and files with excluded keywords.
    """
    print(f"Searching for config.yml files in subfolders of: {root_folder}")

    # Dictionary to hold corpus names, grouped by language code, with dates
    corpus_by_lang = {}
    
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
                            # Extract date from filename if present
                            date_match = re.search(r'_(\d{4}_\d{2}_\d{2})', corpus)
                            date_str = date_match.group(1) if date_match else "0000_00_00" # Use a default date for files without one
                            
                            if lang_code not in corpus_by_lang:
                                corpus_by_lang[lang_code] = []
                            corpus_by_lang[lang_code].append((date_str, corpus))
                        else:
                            print(f"Skipping invalid corpus name: {corpus}")

        except Exception as e:
            print(f"Error processing {config_file}: {e}")
    
    # Filter for the most recent file for each language
    final_corpora = set()
    for lang_code, corpus_list in corpus_by_lang.items():
        # Sort by date in descending order to get the most recent file first
        corpus_list.sort(key=lambda x: x[0], reverse=True)
        # Add the most recent file to the final set
        final_corpora.add(corpus_list[0][1])

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

    # Write the combined config to a new file in the root folder
    output_path = root_folder / 'combined_config.yml'
    try:
        with open(output_path, 'w') as f:
            yaml.dump(global_config, f, sort_keys=False)
        print(f"\nSuccessfully wrote combined configuration to: {output_path}")
    except Exception as e:
        print(f"Failed to write the combined config file: {e}")

def update_config(folder: Path):
    import sys
    import datetime
    config_path = folder / "config.yml"
    if not config_path.is_file():
        print(f"Error: config.yml not found in {folder}")
        sys.exit(1)

    # Backup config.yml
    today = datetime.date.today().strftime("%Y_%m_%d")
    backup_path = folder / f"config_{today}.yml"
    config_path.replace(backup_path)
    print(f"Backed up config.yml to {backup_path}")

    # Load config
    with open(backup_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Helper to find latest stem in scripture_dir
    def find_latest_stem(stem):
        # Match pattern: <prefix>_YYYY_MM_DD
        m = re.match(r"^(.*)_\d{4}_\d{2}_\d{2}$", stem)
        if not m:
            return None
        prefix = m.group(1)
        candidates = []
        for file in SIL_NLP_ENV.mt_scripture_dir.glob(f"{prefix}_????_??_??.*"):
            # Extract date from filename
            file_stem = file.stem
            m2 = re.match(rf"^{re.escape(prefix)}_(\d{{4}}_\d{{2}}_\d{{2}})$", file_stem)
            if m2:
                candidates.append((m2.group(1), file_stem))
        if not candidates:
            return None
        # Sort by date descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    updated = False
    pairs = config.get("data", {}).get("corpus_pairs", [])
    for pair in pairs:
        for key in ("src", "trg"):
            items = pair.get(key, [])
            if isinstance(items, str):
                items = [items]
            new_items = []
            for stem in items:
                m = re.match(r"^(.*)_\d{4}_\d{2}_\d{2}$", stem)
                if m:
                    latest_stem = find_latest_stem(stem)
                    if latest_stem and latest_stem != stem:
                        print(f"Updating {stem} -> {latest_stem}")
                        new_items.append(latest_stem)
                        updated = True
                    else:
                        new_items.append(stem)
                else:
                    new_items.append(stem)
            pair[key] = new_items if len(new_items) > 1 else (new_items[0] if new_items else [])

    if updated:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"Updated config.yml written to {config_path}")
    else:
        print("No updates made to config.yml. Restoring original.")
        backup_path.replace(config_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine multiple config.yml files into one or update config.yml with latest file stems.'
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--update-config',
        action='store_true',
        help='Update config.yml in the given folder with latest file stems.'
    )
    group.add_argument(
        '--output_filename',
        type=str,
        default=None,
        help='Output filename for the combined file. The default is config.yml.'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='The root folder to search for config.yml files or to update config.'
    )

    args = parser.parse_args()
    folder = Path(args.folder)

    if not folder.is_dir():
        folder = Path(SIL_NLP_ENV.mt_experiments_dir) / args.folder    

    if not folder.is_dir():
        print(f"Error: Couldn't find {args.folder} or {folder}.")
    elif args.update_config:
        update_config(folder)
    else:
        output_filename = args.output_filename or "config.yml"
        combine_config_files(folder, output_filename)
