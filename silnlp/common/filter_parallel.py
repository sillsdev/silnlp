"""
Adapted from: www.aclweb.org/anthology/D19-1430: Exploiting Monolingual Data at Scale for Neural Machine Translation
              by  Lijun Wu, Yiren Wang, Yingce Xia, Tao Qin, Jianhuang Lai, Tei-Yan Liu
"""

import re
import os
import sys
import argparse
from string import punctuation
import yaml
import json
import urllib3
import time
from typing import Dict, List
from tqdm import tqdm

from ..common.utils import get_git_revision_hash, merge_dict

_DEFAULT_FILTER_CONFIG: dict = {
    "filter": {
        "dup_toggle": True,
        "src_trg_same_toggle": True,
        "sentence_word_num_toggle": True,
        "sentence_words_ratio_toggle": True,
        "specific_punct_toggle": True,
        "characs_toggle": True,
        "special_char_toggle": True,
        "punctuation_toggle": True,
        "html_toggle": True,
        "characs_sum_toggle": True,
        "latin_toggle": False,
        "scripts_toggle": True,
        "min_tok": 4,
        "max_tok": 200,
        "src_trg_words_ratio": 1.8,
        "max_words_per_sent": 150,
        "avg_word_len_lb": 3.0,
        "avg_word_len_ub": 20,
        "specific_punct_limit": 5,
        "min_punct_threshold": 3,
        "punct_max_num": 12,
        "src_trg_punct_ratio": 3,
        "punct_text_ratio": 0.5,
        "src_trg_char_ratio": 3,
        "latin_ratio": 0.25,
        "valid_scripts": "latin inherited common",
        "script_error_threshold": 1,
        "exclude_symbols": "So",
    },
}


def load_config(config_file_name: str) -> dict:
    config = _DEFAULT_FILTER_CONFIG.copy()

    if config_file_name is None:
        config_path = os.path.join(".", "config.yml")
    else:
        config_path = os.path.join(".", config_file_name)

    if not os.path.isfile(config_path):
        print(f"Warning: config file {config_path} not found; using defaults")
        return config

    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)


def write_config(config_file_name: str, config: dict):
    with open(config_file_name, "w") as file:
        yaml.dump(config, file)


def show_config(config: dict):
    print(json.dumps(config, indent=2))


all_lines = set()


# Duplicated sentences remove
def dup_check(src: str, trg: str) -> bool:
    this_line = src + '<averyunlikelytoken-xyzzy>' + trg
    if this_line in all_lines:
        return True
    all_lines.add(this_line)
    return False


# Same source and target sentence remove
def src_trg_same_check(src: str, trg: str) -> bool:
    return src == trg


# Sentence words number remove
def sentence_word_num_check(src: str, trg: str, min_tok: int, max_tok: int) -> bool:
    def check_word_num(sent):
        segs = sent.strip().split()
        if len(segs) < min_tok or len(segs) > max_tok:
            return True
        return False

    return check_word_num(src) or check_word_num(trg)


# Sentence pair words ratio exceeded remove
def sentence_words_ratio_check(src: str, trg: str, src_trg_words_ratio: float) -> bool:
    m_x = len(src.split())
    m_y = len(trg.split())

    return m_x / m_y > src_trg_words_ratio or m_y / m_x > src_trg_words_ratio


# Specific punctuation number exceeded sentence remove
def specific_punct_check(src: str, trg: str, specific_punct_limit: int) -> bool:
    def hot_fix_filter(sent):
        if sent.count("/") > specific_punct_limit:
            return True
        if sent.count("|") > specific_punct_limit:
            return True
        if sent.count("-") > specific_punct_limit:
            return True
        if len(re.findall("[\d\-\|/]", sent)) / len(sent) > 0.5:
            return True
        return False

    return hot_fix_filter(src) or hot_fix_filter(trg)


# Characters condition remove
def characs_check(src: str, trg: str, max_words_per_sent: int, avg_word_len_lb: float, avg_word_len_ub: float) -> bool:
    def filter_by_len(sent):
        segs = sent.split()
        for x in segs:
            if len(x) > max_words_per_sent:
                return True
        m_char = sum([len(x) for x in segs])
        m_word = len(segs)
        ratio = m_char * 1. / (m_word + 1e-9)
        if ratio > avg_word_len_ub or ratio < avg_word_len_lb:
            return True
        return False

    return filter_by_len(src) or filter_by_len(trg)


# Punctuation condition remove
punctuation_set = set(punctuation)


def punctuation_check(src: str,
                      trg: str,
                      min_punct_threshold: int,
                      punct_max_num: int,
                      src_trg_punct_ratio: float,
                      punct_text_ratio: float) -> bool:
    count_func = lambda l1, l2: sum([1 for x in l1 if x in l2])

    m_punct_src = count_func(src, set(punctuation_set))
    m_punct_trg = count_func(trg, set(punctuation_set))

    if m_punct_src < min_punct_threshold or m_punct_trg < min_punct_threshold:
        return False

    if m_punct_src > punct_max_num or m_punct_trg > punct_max_num:
        return True
    if m_punct_src / (len(src) + 1e-9) > punct_text_ratio or \
            m_punct_trg / (len(trg) + 1e-9) > punct_text_ratio:
        return True
    if m_punct_src / (m_punct_trg + 1e-9) > src_trg_punct_ratio or \
            m_punct_trg / (m_punct_src + 1e-9) > src_trg_punct_ratio:
        return True

    return False


# Html address or tags contained sentence remove
def html_check(src: str, trg: str) -> bool:
    def filter_by_html(sent):
        detector = re.compile('<.*?>')
        html_tag = re.findall(detector, sent)
        if html_tag or 'https://' in sent or 'http://' in sent:
            return True
        return False

    return filter_by_html(src) or filter_by_html(trg)


# Special chars (hard to print)
def special_char_check(src: str, trg: str) -> bool:
    return r"\x" in src or r"\x" in trg


# Optional: Src/trg chars ratio exceeded remove
def characs_sum_check(src: str, trg: str, src_trg_char_ratio: float) -> bool:
    segs_src = src.split()
    m_char_src = sum([len(x) for x in segs_src])

    segs_trg = trg.split()
    m_char_trg = sum([len(y) for y in segs_trg])

    return m_char_src / m_char_trg > src_trg_char_ratio or m_char_trg / m_char_src > src_trg_char_ratio


# Optional: Remove sentences with too many non-Latin characters
def latin_check(src: str, trg: str, latin_ratio: float) -> bool:
    def count_latin(sent):
        if len(re.findall("[^a-zA-Z]", sent)) / len(sent) > latin_ratio:
            return False
        return True

    return count_latin(src) or count_latin(trg)


#
# Script checking code
#

script_dict: Dict = {}
scripts_url = "http://www.unicode.org/Public/UNIDATA/Scripts.txt"


def add_single(hex_str: str, script_name: str, gen_cat: str):
    s = script_name.lower()
    g = gen_cat.lower()
    code_val = int(hex_str, 16)
    script_dict[code_val] = [s, g]


def add_range(start_hex_str: str, end_hex_str: str, script_name: str, gen_cat: str):
    s = script_name.lower()
    g = gen_cat.lower()
    start_code_val = int(start_hex_str, 16)
    end_code_val = int(end_hex_str, 16)
    for i in range(start_code_val, end_code_val+1):
        script_dict[i] = [s, g]


def load_script_dict(valid_scripts: List[str]):
    unicode_re = re.compile(r"^([0-9A-F]{4,5})(..([0-9A-F]{4,5})){0,1}\s+;\s+([\w]+)\s+#\s+([\S]{2,2})\s+")
    http = urllib3.PoolManager()

    start_time = time.time()
    print(f'Loading scripts data from {scripts_url}')
    r = http.request('GET', scripts_url, preload_content=False)

    while True:
        data = r.readline()
        if not data:
            break
        line = data.decode('utf-8').strip()
        if line != "" and not line.startswith('#'):
            parts = unicode_re.match(line)
            if parts.group(4).lower() in valid_scripts:
                if parts.group(3) is None:
                    add_single(parts.group(1), parts.group(4), parts.group(5))
                else:
                    add_range(parts.group(1), parts.group(3), parts.group(4), parts.group(5))
    r.release_conn()
    end_time = time.time()
    print(f'Loaded in {(end_time - start_time):.4f} seconds')


#    print("\n".join(f"{k}\t{v}" for k, v in script_dict.items()))

symbol_cats = {'so'}


def script_check(src: str, trg: str, valid_src_scripts: List[str], valid_trg_scripts: List[str],
                 exclude_symbols: List[str], threshold: int) -> bool:

    def check_line(line: str, scripts: List[str], exc_symbols: List[str], limit: int) -> bool:
        error_count = 0

        for c in line:
            char_code = ord(c)
            if char_code not in script_dict:
                error_count += 1
            else:
                script = script_dict[char_code][0]
                genl_cat = script_dict[char_code][1]

                if script not in scripts:
                    error_count += 1
                elif genl_cat in exc_symbols:
                    error_count += 1

        if error_count >= limit:
            return False
        return True

    return not check_line(src, valid_src_scripts, exclude_symbols, threshold) or \
           not check_line(trg, valid_trg_scripts, exclude_symbols, threshold)


def log_error(log_flag: bool, logfile, label: str, src: str, trg: str):
    if log_flag:
        logfile.write(f"{label}\t{src}\t{trg}\n")


def main() -> None:
    def ratio_string(count: int, total: int) -> str:
        return f'{count: >10} ({(100 * count / total):6.2f}%)'

    def print_counters(out_file):
        out_file.write(f'  {original_line_count:>10}\tOriginal sentences\n')
        out_file.write(f'- {ratio_string(count_duplicates, original_line_count)}\tduplicate src/trg sentence pairs\n')
        out_file.write(f'- {ratio_string(count_src_trg_same, original_line_count)}\tsame src/trg sentence\n')
        out_file.write(f'- {ratio_string(count_word_num, original_line_count)}\t'
                       f'word count < {min_tok} or > {max_tok}\n')
        out_file.write(f'- {ratio_string(count_words_ratio, original_line_count)}\t'
                       f'exceeded src/trg word ratio ({src_trg_words_ratio})\n')
        out_file.write(f'- {ratio_string(count_specific_punc, original_line_count)}\t'
                       f'exceeded specific punct. limit ({specific_punct_limit})\n')
        out_file.write(f'- {ratio_string(count_characs, original_line_count)}\t'
                       f'exceeded max words ({max_words_per_sent}) or avg word length bounds '
                       f'({avg_word_len_lb}/{avg_word_len_ub})\n')
        out_file.write(f'- {ratio_string(count_special_char, original_line_count)}\tspecial characters\n')
        out_file.write(f'- {ratio_string(count_punctuation, original_line_count)}\t'
                       f'max. punct ({punct_max_num}), src/trg punct. ratio ({src_trg_punct_ratio}),'
                       f' text/punct ratio ({punct_text_ratio})\n')
        out_file.write(f'- {ratio_string(count_html, original_line_count)}\tHTML\n')
        out_file.write(f'- {ratio_string(count_characs_sum, original_line_count)}\t'
                       f'src/trg character ratio ({src_trg_char_ratio})\n')
        out_file.write(f'- {ratio_string(count_latin, original_line_count)}\tLatin ratio ({latin_ratio})\n')
        out_file.write(f'- {ratio_string(count_script, original_line_count)}\t'
                       f'>={script_error_threshold} chars not in valid script list '
                       f'(src: {valid_src_scripts}; trg: {valid_trg_scripts})\n')
        out_file.write(f'= {ratio_string(final_line_count, original_line_count)}\tRemaining sentences\n')

    parser = argparse.ArgumentParser(description="Filtering for noisy parallel corpora")
    parser.add_argument('src', type=str, help='source file')
    parser.add_argument('trg', type=str, help='target file')
    parser.add_argument('--config', type=str, default='config.yml', help='config file')
    parser.add_argument('--errors', default=False, action="store_true", help="log errors")
    parser.add_argument('--max_lines', type=int, default=-1, help='max lines to process')
    args = parser.parse_args()

    rev_hash = get_git_revision_hash()

    config = load_config(args.config)
    filter_config = config.get("filter")
    write_config(f"effective_config-{rev_hash}.yml", filter_config)
    max_lines = args.max_lines

    # Initialize counters
    original_line_count = final_line_count = 0
    count_duplicates = count_src_trg_same = count_word_num = count_words_ratio = count_specific_punc = 0
    count_characs = count_special_char = count_punctuation = count_html = count_characs_sum = count_latin = 0
    count_script = 0

    # Initialize toggles
    dup_toggle = filter_config.get("dup_toggle")
    src_trg_same_toggle = filter_config.get("src_trg_same_toggle")
    sentence_word_num_toggle = filter_config.get("sentence_word_num_toggle")
    sentence_words_ratio_toggle = filter_config.get("sentence_words_ratio_toggle")
    specific_punct_toggle = filter_config.get("specific_punct_toggle")
    characs_toggle = filter_config.get("characs_toggle")
    special_char_toggle = filter_config.get("special_char_toggle")
    punctuation_toggle = filter_config.get("punctuation_toggle")
    html_toggle = filter_config.get("html_toggle")
    characs_sum_toggle = filter_config.get("characs_sum_toggle")
    latin_toggle = filter_config.get("latin_toggle")
    scripts_toggle = filter_config.get("scripts_toggle")

    # Initialize settings
    min_tok = filter_config.get('min_tok')
    max_tok = filter_config.get('max_tok')
    src_trg_words_ratio = filter_config.get('src_trg_words_ratio')
    specific_punct_limit = filter_config.get('specific_punct_limit')
    max_words_per_sent = filter_config.get('max_words_per_sent')
    avg_word_len_lb = filter_config.get('avg_word_len_lb')
    avg_word_len_ub = filter_config.get('avg_word_len_ub')
    min_punct_threshold = filter_config.get('min_punct_threshold')
    punct_max_num = filter_config.get('punct_max_num')
    src_trg_punct_ratio = filter_config.get('src_trg_punct_ratio')
    punct_text_ratio = filter_config.get('punct_text_ratio')
    src_trg_char_ratio = filter_config.get('src_trg_char_ratio')
    latin_ratio = filter_config.get('latin_ratio', 0.25)
    if filter_config.get('valid_src_scripts') and filter_config.get('valid_trg_scripts'):
        valid_src_scripts = [s.lower() for s in filter_config.get('valid_src_scripts').split()]
        valid_trg_scripts = [s.lower() for s in filter_config.get('valid_trg_scripts').split()]
    else:
        valid_src_scripts = [s.lower() for s in filter_config.get('valid_scripts').split()]
        valid_trg_scripts = valid_src_scripts
    script_error_threshold = filter_config.get('script_error_threshold')
    exclude_symbols = [s.lower() for s in filter_config.get('exclude_symbols').split()]

    # Initial setup (if needed)
    if scripts_toggle:
        load_script_dict(list(set(valid_src_scripts) | set(valid_trg_scripts)))

    src_out_file_name = f"{os.path.splitext(args.src)[0]}_clean{os.path.splitext(args.src)[1]}"
    src_err_file_name = f"{os.path.splitext(args.src)[0]}_errors{os.path.splitext(args.src)[1]}"
    trg_out_file_name = f"{os.path.splitext(args.trg)[0]}_clean{os.path.splitext(args.trg)[1]}"
    with open(args.src, "r", encoding="utf-8") as src_in, \
            open(args.trg, "r", encoding="utf-8") as trg_in, \
            open(src_out_file_name, "w", encoding="utf-8") as src_out, \
            open(trg_out_file_name, "w", encoding="utf-8") as trg_out, \
            open(src_err_file_name, "w", encoding="utf-8") as error_log:
        for src_line, trg_line in tqdm(zip(src_in, trg_in)):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            original_line_count += 1

            if dup_toggle and dup_check(src_line, trg_line):
                count_duplicates += 1
                log_error(args.errors, error_log, "dup_check", src_line, trg_line)
            elif src_trg_same_toggle and src_trg_same_check(src_line, trg_line):
                count_src_trg_same += 1
                log_error(args.errors, error_log, "src_trg_same_check", src_line, trg_line)
            elif sentence_word_num_toggle and sentence_word_num_check(src_line, trg_line, min_tok, max_tok):
                count_word_num += 1
                log_error(args.errors, error_log, "sentence_word_num_check", src_line, trg_line)
            elif sentence_words_ratio_toggle and sentence_words_ratio_check(src_line, trg_line, src_trg_words_ratio):
                count_words_ratio += 1
                log_error(args.errors, error_log, "sentence_words_ratio_check", src_line, trg_line)
            elif specific_punct_toggle and specific_punct_check(src_line, trg_line, specific_punct_limit):
                count_specific_punc += 1
                log_error(args.errors, error_log, "specific_punct_check", src_line, trg_line)
            elif characs_toggle and characs_check(src_line, trg_line, max_words_per_sent, avg_word_len_lb,
                                                  avg_word_len_ub):
                count_characs += 1
                log_error(args.errors, error_log, "characs_check", src_line, trg_line)
            elif special_char_toggle and special_char_check(src_line, trg_line):
                count_special_char += 1
                log_error(args.errors, error_log, "special_char_check", src_line, trg_line)
            elif punctuation_toggle and punctuation_check(src_line, trg_line, min_punct_threshold, punct_max_num,
                                                          src_trg_punct_ratio, punct_text_ratio):
                count_punctuation += 1
                log_error(args.errors, error_log, "punctuation_check", src_line, trg_line)
            elif html_toggle and html_check(src_line, trg_line):
                count_html += 1
                log_error(args.errors, error_log, "html_check", src_line, trg_line)
            elif characs_sum_toggle and characs_sum_check(src_line, trg_line, src_trg_char_ratio):
                count_characs_sum += 1
                log_error(args.errors, error_log, "characs_sum_check", src_line, trg_line)
            elif latin_toggle and latin_check(src_line, trg_line, latin_ratio):
                count_latin += 1
                log_error(args.errors, error_log, "latin_check", src_line, trg_line)
            elif scripts_toggle and script_check(src_line, trg_line,
                                                 valid_src_scripts, valid_trg_scripts, exclude_symbols,
                                                 script_error_threshold):
                count_script += 1
                log_error(args.errors, error_log, "script_check", src_line, trg_line)
            else:
                print(src_line, file=src_out)
                print(trg_line, file=trg_out)
                final_line_count += 1

            if max_lines != -1 and original_line_count >= max_lines:
                break

    print_counters(sys.stdout)
    with open("log.out", "w", encoding="utf-8") as f:
        print_counters(f)


if __name__ == "__main__":
    main()
