import argparse
import sys
from pathlib import Path

import regex as re
import yaml

from ..common.environment import SIL_NLP_ENV

# List of keywords to exclude from filenames
EXCLUDED_KEYWORDS = ["XRI", "_AI", "train"]

# A hardcoded list of major language ISO codes from the Flores-200 benchmark.
# This list can be modified as needed.
MAJOR_LANGS = {
"af",
"afr",
"am",
"amh",
"ar",
"ara",
"ast",
"bam",
"be",
"bel",
"ben",
"bg",
"bm",
"bn",
"bos",
"bs",
"bul",
"ca",
"cat",
"ces",
"co",
"cos",
"cs",
"cy",
"cym",
"da",
"dan",
"de",
"deu",
"el",
"ell",
"en",
"eng",
"es",
"est",
"et",
"eu",
"eus",
"fa",
"fas",
"ff",
"fi",
"fin",
"fr",
"fra",
"frp",
"fry",
"ful",
"fur",
"fy",
"ga",
"gag",
"gd",
"gl",
"gla",
"gle",
"glg",
"gn",
"grn",
"ha",
"hau",
"he",
"heb",
"hi",
"hif",
"hin",
"hr",
"hrv",
"hu",
"hun",
"ibo",
"id",
"ig",
"ilo",
"ind",
"is",
"isl",
"it",
"ita",
"ja",
"jpn",
"ka",
"kat",
"kbp",
"kg",
"khm",
"kir",
"km",
"ko",
"kon",
"kor",
"ky",
"lav",
"lg",
"li",
"lim",
"lin",
"lit",
"lmo",
"ln",
"lt",
"lug",
"lus",
"lv",
"mal",
"mar",
"mg",
"mk",
"mkd",
"ml",
"mlg",
"mlt",
"mr",
"mt",
"my",
"mya",
"nb",
"ne",
"nep",
"nl",
"nld",
"nob",
"nqo",
"ny",
"nya",
"oc",
"oci",
"om",
"or",
"ori",
"orm",
"pap",
"pl",
"pms",
"pol",
"por",
"ps",
"pt",
"pus",
"rm",
"ro",
"roh",
"ron",
"ru",
"rus",
"sat",
"scn",
"sco",
"sd",
"si",
"sin",
"sk",
"sl",
"slk",
"slv",
"sn",
"sna",
"snd",
"so",
"som",
"sot",
"spa",
"sq",
"sqi",
"sr",
"srp",
"ss",
"ssw",
"st",
"su",
"sun",
"sv",
"sw",
"swa",
"swe",
"tg",
"tgk",
"th",
"tha",
"ti",
"tir",
"tk",
"tn",
"tr",
"ts",
"tsn",
"tso",
"tuk",
"tur",
"uk",
"ukr",
"ur",
"urd",
"uz",
"uzb",
"vec",
"vi",
"vie",
"wa",
"war",
"wln",
"xh",
"xho",
"yi",
"yid",
"yo",
"yor",
"yue",
"zh",
"zho",
"zu",
"zul",
}


def extract_lang_code(corpus_name):
    """
    Extracts a 2 or 3 letter ISO code from a corpus name that follows the
    <iso>-<filename> format. Returns the ISO code or None if the format is invalid.
    """
    match = re.match(r"^[a-z]{2,3}-", corpus_name)
    if match:
        return match.group(0)[:-1]
    return None


def is_alignment_config(config_file: Path) -> bool:
    """
    Returns True if the YAML file contains a top-level 'data' key
    with an 'aligner' key inside it.
    """
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return (
            isinstance(config, dict)
            and "data" in config
            and isinstance(config["data"], dict)
            and "aligner" in config["data"]
        )
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return False


def find_config_files(root_folder: Path) -> list:
    """
    Finds all config.yml files in subfolders that are likely to be alignment configs.
    Returns a list of Path objects for these files.
    """
    print(f"Searching for config.yml files in subfolders of: {root_folder}")
    alignment_configs = []
    for config_file in root_folder.rglob("**/config.yml"):
        if is_alignment_config(config_file):
            alignment_configs.append(config_file)

    if not alignment_configs:
        print("No alignment config files found.")
    else:
        print(f"Found {len(alignment_configs)} alignment config files:")

    return alignment_configs


def combine_config_files(root_folder: Path) -> dict:
    """
    Combines alignment config files into one.
    Re-sorts languages, de-duplicates entries, and sets a new aligner.
    Filters out older files with dates and files with excluded keywords.
    Returns the config_data.
    """

    # Dictionary to hold corpus names, grouped by language code
    corpus_by_lang = {}

    # Initialize a base config with defaults
    global_config = {
        "data": {
            "aligner": "eflomal",
            "corpus_pairs": [
                {"mapping": "many_to_many", "src": [], "trg": [], "test_size": 0, "type": "train", "val_size": 0}
            ],
        }
    }
    tokenize_setting = None

    config_files = find_config_files(root_folder)

    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            if tokenize_setting is None and "data" in config and "tokenize" in config["data"]:
                tokenize_setting = config["data"]["tokenize"]

            if "data" in config and "corpus_pairs" in config["data"]:
                for pair in config["data"]["corpus_pairs"]:
                    # Handle cases where 'src' or 'trg' are single strings, not lists
                    src_items = pair.get("src", [])
                    if isinstance(src_items, str):
                        src_items = [src_items]

                    trg_items = pair.get("trg", [])
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
                                corpus_by_lang[lang_code] = {"dated": [], "undated": []}

                            # Extract date from filename if present
                            date_match = re.search(r"_(\d{4}_\d{2}_\d{2})", corpus)
                            if date_match:
                                date_str = date_match.group(1)

                                corpus_by_lang[lang_code]["dated"].append((date_str, corpus))
                            else:
                                corpus_by_lang[lang_code]["undated"].append(corpus)
                        else:
                            print(f"Skipping invalid corpus name: {corpus}")

        except Exception as e:
            print(f"Error processing {config_file}: {e}")

    # Filter for the most recent file for each language and include all undated files
    final_corpora = set()
    for lang_code, corpora_dict in corpus_by_lang.items():
        # Keep all undated files
        for corpus in corpora_dict["undated"]:
            final_corpora.add(corpus)

        # Keep only the most recent dated file, if any exist
        if corpora_dict["dated"]:
            corpora_dict["dated"].sort(key=lambda x: x[0], reverse=True)

            final_corpora.add(corpora_dict["dated"][0][1])

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
    global_config["data"]["corpus_pairs"][0]["src"] = sorted(list(major_corpora)) + sorted(list(minor_corpora))
    global_config["data"]["corpus_pairs"][0]["trg"] = sorted(list(minor_corpora))

    # Add tokenize setting if it was found
    if tokenize_setting is not None:
        global_config["data"]["tokenize"] = tokenize_setting

    return global_config


def write_config_file(output_path: Path, config_data: dict):

    # Write the combined config file.
    try:
        with open(output_path, "w") as f:
            yaml.dump(config_data, f, sort_keys=False)
        print(f"\nSuccessfully wrote combined configuration to: {output_path}")
    except Exception as e:
        print(f"Failed to write the combined config file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple config.yml files into one.")
    parser.add_argument("folder", type=str, help="The root folder to search for config.yml files.")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="config.yml",
        help="Output filename for the combined file. The default is config.yml.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Will overwrite an existing config output file.")

    args = parser.parse_args()
    folder = Path(args.folder)

    # If the provided folder is not a directory, assume it's a relative path within the experiments directory
    if not folder.is_dir():
        folder = Path(SIL_NLP_ENV.mt_experiments_dir) / args.folder

    if not folder.is_dir():
        print(f"Error: Couldn't find {args.folder} or {folder}.")
        sys.exit(1)

    else:
        # Place the new config file in a subfolder named 'Align'. Create that output folder if it doesn't exist.
        # Don't overwrite an existing config.yml file unless the --overwrite flag is used.

        output_folder = folder / "Align"
        output_file = output_folder / args.output_filename

        # 1. Ensure the output folder exists
        if not output_folder.is_dir():
            print(f"Creating output folder: {output_folder}")
            output_folder.mkdir(parents=True, exist_ok=True)

        # 2. Check if output file exists
        if output_file.is_file() and not args.overwrite:
            print(f"Warning: {output_file} already exists. Use the --overwrite flag to overwrite the file.")
            sys.exit(2)

        # 3. If allowed to write, get config_data
        config_data = combine_config_files(folder)

        # 4. Inform user about config_data and writing outcome
        if not config_data or not config_data.get("data", {}).get("corpus_pairs", [{}])[0].get("src"):
            print("No config data found. No output file will be written.")
            sys.exit(3)

        if output_file.is_file() and args.overwrite:
            print(f"Will overwrite existing config file: {output_file}")
        else:
            print(f"Writing new config file to: {output_file}")

        write_config_file(output_file, config_data)
