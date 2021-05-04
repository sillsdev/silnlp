"""
Adapted from: www.aclweb.org/anthology/D19-1430: Exploiting Monolingual Data at Scale for Neural Machine Translation
              by  Lijun Wu, Yiren Wang, Yingce Xia, Tao Qin, Jianhuang Lai, Tei-Yan Liu
"""

import re
import os
import argparse
from string import punctuation
import yaml
import json
import urllib3
import time
from typing import Dict, List

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
        "src_trg_words_ratio": 2.0,
        "max_words_per_sent": 150,
        "avg_word_len_lb": 2.8,
        "avg_word_len_ub": 20,
        "specific_punct_limit": 5,
        "min_punct_threshold": 3,
        "punct_max_num": 12,
        "src_trg_punct_ratio": 3,
        "punct_text_ratio": 0.5,
        "src_trg_char_ratio": 3,
        "latin_ratio": 0.25,
        "valid_scripts": "latin common",
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

    with config_path.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)


def show_config(config: dict):
    print(json.dumps(config, indent=2))


all_lines = set()


# Duplicated sentences remove
def dup_check(src: str, trg: str) -> bool:
    this_line = src + "<averyunlikelytoken-xyzzy>" + trg
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
        ratio = m_char * 1.0 / (m_word + 1e-9)
        if ratio > avg_word_len_ub or ratio < avg_word_len_lb:
            return True
        return False

    return filter_by_len(src) or filter_by_len(trg)


# Punctuation condition remove
punctuation_set = set(punctuation)


def punctuation_check(
    src: str,
    trg: str,
    min_punct_threshold: int,
    punct_max_num: int,
    src_trg_punct_ratio: float,
    punct_text_ratio: float,
) -> bool:
    count_func = lambda l1, l2: sum([1 for x in l1 if x in l2])

    m_punct_src = count_func(src, set(punctuation_set))
    m_punct_trg = count_func(trg, set(punctuation_set))

    if m_punct_src < min_punct_threshold or m_punct_trg < min_punct_threshold:
        return False

    if m_punct_src > punct_max_num or m_punct_trg > punct_max_num:
        return True
    if m_punct_src / (len(src) + 1e-9) > punct_text_ratio or m_punct_trg / (len(trg) + 1e-9) > punct_text_ratio:
        return True
    if (
        m_punct_src / (m_punct_trg + 1e-9) > src_trg_punct_ratio
        or m_punct_trg / (m_punct_src + 1e-9) > src_trg_punct_ratio
    ):
        return True

    return False


# Html address or tags contained sentence remove
def html_check(src: str, trg: str) -> bool:
    def filter_by_html(sent):
        detector = re.compile("<.*?>")
        html_tag = re.findall(detector, sent)
        if html_tag or "https://" in sent or "http://" in sent:
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


def add_single(hex_str: str, script_name: str):
    s = script_name.lower()
    code_val = int(hex_str, 16)
    if s in script_dict.keys():
        script_dict[s].add(code_val)
    else:
        script_dict[s] = {code_val}


def add_range(start_hex_str: str, end_hex_str: str, script_name: str):
    s = script_name.lower()
    start_code_val = int(start_hex_str, 16)
    end_code_val = int(end_hex_str, 16)
    if s in script_dict.keys():
        script_dict[s].update(set(range(start_code_val, end_code_val + 1)))
    else:
        script_dict[s] = set(range(start_code_val, end_code_val + 1))


def load_script_dict():
    unicode_re = re.compile(r"^([0-9A-F]{4,5})(..([0-9A-F]{4,5})){0,1}\s+;\s+([\w]+)\s+#")
    http = urllib3.PoolManager()

    start_time = time.time()
    print(f"Loading scripts data from {scripts_url}")
    r = http.request("GET", scripts_url, preload_content=False)

    while True:
        data = r.readline()
        if not data:
            break
        line = data.decode("utf-8").strip()
        if line != "" and not line.startswith("#"):
            parts = unicode_re.match(line)
            if parts.group(3) is None:
                add_single(parts.group(1), parts.group(4))
            else:
                add_range(parts.group(1), parts.group(3), parts.group(4))
    r.release_conn()
    end_time = time.time()
    print(f"Loaded in {(end_time-start_time):.4f} seconds")


#    print("\n".join(f"{k}\t{v}" for k, v in script_dict.items()))


def script_check(src: str, trg: str, valid_scripts: List[str]) -> bool:
    def check_line(line: str, scripts: List[str]) -> bool:
        for c in line:
            char_code = ord(c)

            match = False
            for s in scripts:
                if char_code in script_dict[s]:
                    match = True
                    break

            # Character c wasn't part of one of the valid scripts
            if not match:
                return False

        # Successfully checked all the characters in the line
        return True

    return not check_line(src, valid_scripts) or not check_line(trg, valid_scripts)


def log_error(log_flag: bool, logfile, label: str, src: str, trg: str):
    if log_flag:
        logfile.write(f"{label}\t{src}\t{trg}\n")


def main() -> None:
    def ratio_string(count: int, total: int) -> str:
        return f"{count: >10} ({(100*count/total):6.2f}%)"

    parser = argparse.ArgumentParser(description="Filtering for noisy parallel corpora")
    parser.add_argument("src", type=str, help="source file")
    parser.add_argument("trg", type=str, help="target file")
    parser.add_argument("--config", type=str, default="config.yml", help="config file")
    parser.add_argument("--errors", default=False, action="store_true", help="log errors")
    parser.add_argument("--max_lines", type=int, default=-1, help="max lines to process")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    config = load_config(args.config)
    filter_config = config.get("filter")
    show_config(filter_config)
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
    min_tok = filter_config.get("min_tok")
    max_tok = filter_config.get("max_tok")
    src_trg_words_ratio = filter_config.get("src_trg_words_ratio")
    specific_punct_limit = filter_config.get("specific_punct_limit")
    max_words_per_sent = filter_config.get("max_words_per_sent")
    avg_word_len_lb = filter_config.get("avg_word_len_lb")
    avg_word_len_ub = filter_config.get("avg_word_len_ub")
    min_punct_threshold = filter_config.get("min_punct_threshold")
    punct_max_num = filter_config.get("punct_max_num")
    src_trg_punct_ratio = filter_config.get("src_trg_punct_ratio")
    punct_text_ratio = filter_config.get("punct_text_ratio")
    src_trg_char_ratio = filter_config.get("src_trg_char_ratio")
    latin_ratio = filter_config.get("latin_ratio", 0.25)
    valid_scripts = [s.lower() for s in filter_config.get("valid_scripts").split()]

    # Initial setup (if needed)
    if scripts_toggle:
        load_script_dict()

    with open(args.src, "r", encoding="utf-8") as src_in, open(args.trg, "r", encoding="utf-8") as trg_in, open(
        f"{args.src}.clean", "w", encoding="utf-8"
    ) as src_out, open(f"{args.trg}.clean", "w", encoding="utf-8") as trg_out, open(
        f"{args.src}.errors", "w", encoding="utf-8"
    ) as error_log:
        for src_line, trg_line in zip(src_in, trg_in):
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
            elif characs_toggle and characs_check(
                src_line, trg_line, max_words_per_sent, avg_word_len_lb, avg_word_len_ub
            ):
                count_characs += 1
                log_error(args.errors, error_log, "characs_check", src_line, trg_line)
            elif special_char_toggle and special_char_check(src_line, trg_line):
                count_special_char += 1
                log_error(args.errors, error_log, "special_char_check", src_line, trg_line)
            elif punctuation_toggle and punctuation_check(
                src_line, trg_line, min_punct_threshold, punct_max_num, src_trg_punct_ratio, punct_text_ratio
            ):
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
            elif scripts_toggle and script_check(src_line, trg_line, valid_scripts):
                count_script += 1
                log_error(args.errors, error_log, "script_check", src_line, trg_line)
            else:
                print(src_line, file=src_out)
                print(trg_line, file=trg_out)
                final_line_count += 1

            if max_lines != -1 and original_line_count >= max_lines:
                break

    print(f"  {original_line_count:>10}\tOriginal sentences")
    print(f"- {ratio_string(count_duplicates,original_line_count)}\tduplicate src/trg sentence pairs")
    print(f"- {ratio_string(count_src_trg_same,original_line_count)}\tsame src/trg sentence")
    print(f"- {ratio_string(count_word_num,original_line_count)}\tword count < {min_tok} or > {max_tok}")
    print(
        f"- {ratio_string(count_words_ratio,original_line_count)}\texceeded src/trg word ratio ({src_trg_words_ratio})"
    )
    print(
        f"- {ratio_string(count_specific_punc,original_line_count)}\texceeded specific punct. limit ({specific_punct_limit})"
    )
    print(
        f"- {ratio_string(count_characs,original_line_count)}\texceeded max words ({max_words_per_sent}) or avg word length bounds ({avg_word_len_lb}/{avg_word_len_ub})"
    )
    print(f"- {ratio_string(count_special_char,original_line_count)}\tspecial characters")
    print(
        f"- {ratio_string(count_punctuation,original_line_count)}\tmax. punct ({punct_max_num}), src/trg punct. ratio ({src_trg_punct_ratio}), text/punct ratio ({punct_text_ratio})"
    )
    print(f"- {ratio_string(count_html,original_line_count)}\tHTML")
    print(f"- {ratio_string(count_characs_sum,original_line_count)}\tsrc/trg character ratio ({src_trg_char_ratio})")
    print(f"- {ratio_string(count_latin,original_line_count)}\tLatin ratio ({latin_ratio})")
    print(f"- {ratio_string(count_script,original_line_count)}\tInvalid scripts ({valid_scripts})")
    print(f"= {ratio_string(final_line_count,original_line_count)}\tRemaining sentences")


if __name__ == "__main__":
    main()