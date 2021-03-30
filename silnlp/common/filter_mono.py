"""
Original source: www.aclweb.org/anthology/D19-1430: Exploiting Monolingual Data at Scale for Neural Machine Translation
                 Lijun Wu, Yiren Wang, Yingce Xia, Tao Qin, Jianhuang Lai, Tei-Yan Liu

Usage:
"python preprocess_mono.py monolingual_file"

Monolingual data filtering hard rules, including:
1. duplicated sentences remove
3. sentences with '/', '|', '-' > 5
4. sentences with digital numbers/characters > 0.5
5. sentences contains word composed by more than 40 characters
6. sentences with average characters for word > 20 or <4
7. sentences with punctuations > 15
8. sentences with punctuations/characters > 0.5
10. sentences with html address and html tags
11. optional: non english characters > 0.25
"""
import time
import re
import argparse
from string import punctuation

parser = argparse.ArgumentParser()
parser.add_argument("src", help="source file")
parser.add_argument(
    "--soft_html",
    action="store_true",
    default=False,
    help="whether to use soft version only to remove html tag, not the sentence",
)
parser.add_argument(
    "--check_non_latin", action="store_true", default=False, help="Check for too many non-Latin characters"
)
args = parser.parse_args()
f1 = args.src

# Checking flags
check_non_latin = args.check_non_latin

# Checking settings
min_tok = 4
max_top = 200
max_words_per_sent = 150
avg_word_len_lb = 3
avg_word_len_ub = 20
min_punct_threshold = 3
punc_max_num = 10
latin_ratio = 0.3

# Duplicated sentences remove
def dup_remove(x_in):
    all_lines = [x.strip() for x in x_in]
    x_out = set(all_lines)  # make as set

    print("After removing duplicated sentences, %i pairs sentences" % len(x_out))
    return x_out


def sentence_word_num_remove(x_in):
    def check_word_num(sent):
        segs = sent.strip().split()
        if len(segs) < min_tok or len(segs) > max_top:
            return False
        return True

    x_out = []

    for x in x_in:
        if check_word_num(x):
            x_out.append(x.strip())

    print("After removing sentences with too few or too many words, %i sentences remain" % len(x_out))
    return x_out


# Specific punctuation number exceeded sentence remove
def specific_punc_remove(x_in):
    def hot_fix_filter(sent):
        sent = sent.strip()
        if sent.count("/") > 5:
            return False
        if sent.count("|") > 5:
            return False
        if sent.count("-") > 5:
            return False
        if len(re.findall("[\d\-\|/]", sent)) / len(sent) > 0.5:
            return False
        return True

    x_out = []

    for x in x_in:
        if hot_fix_filter(x):
            x_out.append(x.strip())

    print("After removing sentences with too many specific punctuations, %i sentences remain" % len(x_out))
    return x_out


# Characters condition remove
def characs_remove(x_in):
    def filter_by_len(sent):
        segs = sent.strip().split()
        for x in segs:
            if len(x) > max_words_per_sent:
                return False
        m_char = sum([len(x) for x in segs])
        m_word = len(segs)
        ratio = m_char * 1.0 / (m_word + 1e-9)
        if ratio > avg_word_len_ub or ratio < avg_word_len_lb:
            return False
        return True

    x_out = []

    for x in x_in:
        if filter_by_len(x):
            x_out.append(x.strip())

    print("After removing sentence with characters condition, %i sentences remain" % len(x_out))
    return x_out


# Punctuation condition remove
def punctuation_remove(x_in):
    x_out = []

    count_func = lambda l1, l2: sum([1 for x in l1 if x in l2])

    punctuation_set = set(punctuation)
    for x in x_in:
        m_punc_x = count_func(x.strip(), set(punctuation_set))
        if m_punc_x / (len(x.strip()) + 1e-9) > 0.5 or m_punc_x > punc_max_num:
            continue
        x_out.append(x.strip())

    print("After removing sentences with too much punctuations, %i sentences remain" % len(x_out))
    return x_out


# Html address or tags contained sentence remove
def html_remove(x_in):
    x_out = []

    def filter_by_html(sentence):
        sen = sentence.strip()
        detector = re.compile("<.*?>")
        html_tag = re.findall(detector, sen)
        if html_tag or "https://" in sen or "http://" in sen:
            return False
        return True

    def soft_filter_by_html(sent):
        sent = sent.strip()
        detector = re.compile("<.*?>")
        sent = re.sub(detector, "", sent)
        sent = re.sub("https?:\/\/.*[ \r\n]", "", x, flags=re.MULTILINE)
        return sent

    for x in x_in:
        if args.soft_html:
            x_out.append(soft_filter_by_html(x))
        else:
            if filter_by_html(x):
                x_out.append(x.strip())

    print("After removing sentences with html address or tags, %i sentences remain" % len(x_out))
    return x_out


# From Teacher Xia, special chars (hard to print)
def special_char_remove(x_in):
    x_out = []

    for x in x_in:
        if r"\x" in x:
            continue
        x_out.append(x.strip())

    print("After removing sentences with special characters, %i sentences remain" % len(x_out))
    return x_out


# Optional: Latin letter contained sentence remove
def latin_remove(x_in):
    def count_latin(sent):
        if len(re.findall("[^a-zA-Z]", sent)) / len(sent) > latin_ratio:
            return False
        return True

    x_out = []
    for x in x_in:
        if count_latin(x.strip()):
            x_out.append(x.strip())

    print("After removing sentences with too many non-Latin characters, %i sentences remain" % len(x_out))
    return x_out


filter_1 = []

fr_1 = open(f1, "r", encoding="utf8")

f1_all_lines = fr_1.readlines()

print(f"Starting with {len(f1_all_lines)} sentences")
start = time.time()
filter_1 = dup_remove(f1_all_lines)
end = time.time()
print(f"Elapsed time: {((end-start)/60):.2f} minutes")
start = time.time()
filter_1 = sentence_word_num_remove(filter_1)
end = time.time()
print(f"Elapsed time: {((end-start)/60):.2f} minutes")
start = time.time()
filter_1 = specific_punc_remove(filter_1)
end = time.time()
print(f"Elapsed time: {((end-start)/60):.2f} minutes")
start = time.time()
filter_1 = characs_remove(filter_1)
end = time.time()
print(f"Elapsed time: {((end-start)/60):.2f} minutes")
start = time.time()
filter_1 = special_char_remove(filter_1)
end = time.time()
print(f"Elapsed time: {((end-start)/60):.2f} minutes")
start = time.time()
filter_1 = punctuation_remove(filter_1)
end = time.time()
print(f"Elapsed time: {((end-start)/60):.2f} minutes")
start = time.time()
filter_1 = html_remove(filter_1)
end = time.time()
print(f"Elapsed time: {((end-start)/60):.2f} minutes")
start = time.time()
if check_non_latin:
    filter_1 = latin_remove(filter_1)
    end = time.time()
    print(f"Elapsed time: {((end - start) / 60):.2f} minutes")

fr_1.close()


fw_1 = open(f1 + ".clean", "w", encoding="utf8")

print("After all filtering rules, %i sentences remain" % len(filter_1))

for x in filter_1:
    print(x, file=fw_1)

fw_1.close()
