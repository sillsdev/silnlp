"""
Original source: www.aclweb.org/anthology/D19-1430: Exploiting Monolingual Data at Scale for Neural Machine Translation
                 Lijun Wu, Yiren Wang, Yingce Xia, Tao Qin, Jianhuang Lai, Tei-Yan Liu

Usage:
"python preprocess.py source_file target_file"

Parallel data filtering hard rules, including:
1. duplicated sentences
2. same source and target sentences
3. sentences with '/', '|', '-' > 5
4. sentences with digital numbers/characters > 0.5
5. sentences contains word composed by more than 40 characters
6. sentences with average characters for word > 20 or <4
7. sentences with punctuations > 15
8. sentences with punctuations/characters > 0.5
9. src punctuations/tgt punctuations > 5 or 1/5
10. sentences with html address and html tags
11. optional: non english characters > 0.25
12. optional: src characters / tgt characters > 3 or 1/3
"""
import sys
import re
import argparse
from string import punctuation

parser = argparse.ArgumentParser()
parser.add_argument("src", help="source file")
parser.add_argument("tgt", help="target file")
parser.add_argument("--soft_html", action="store_true", default=False, help="only remove html tag, not the sentence")
args = parser.parse_args()
f1 = args.src
f2 = args.tgt

min_tok = 4
max_top = 200
src_tgt_words_ratio = 1.8
max_words_per_sent = 150
avg_word_len_lb = 3
avg_word_len_ub = 20
min_punct_threshold = 3
punc_max_num = 12
punc_ratio = 3
src_tgt_char_ratio = 3
lattin_ratio = 0.25

# Duplicated sentences remove
def dup_remove(x_in, y_in):
    tok = "lijun_wu"
    # all_lines = [x.strip() for x in x_in]
    all_lines = []
    for idx, (x, y) in enumerate(zip(x_in, y_in)):
        all_lines.append(x.strip() + tok + y.strip())  # [src+tok+tgt]
    all_lines = set(all_lines)  # make as set

    x_out = []
    y_out = []
    for sent in all_lines:
        segs = sent.split(tok)
        x_out.append(segs[0])
        y_out.append(segs[1])
    assert len(x_out) == len(y_out)
    print("After removing duplicated sentences, %i pairs remain" % len(x_out))
    return x_out, y_out


# Same source and target sentence remove
def src_tgt_same_remove(x_in, y_in):
    x_out = []
    y_out = []
    for (x, y) in zip(x_in, y_in):
        if x.strip() == y.strip():
            continue
        x_out.append(x.strip())
        y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing same source and target sentence, %i pairs remain" % len(x_out))
    return x_out, y_out


# Sentence words number remove
def sentence_word_num_remove(x_in, y_in):
    def check_word_num(sent):
        segs = sent.strip().split()
        if len(segs) < min_tok or len(segs) > max_top:
            return False
        return True

    x_out = []
    y_out = []

    with open("dump.tsv", "w", encoding="utf-8") as f:
        for (x, y) in zip(x_in, y_in):
            if check_word_num(x) and check_word_num(y):
                x_out.append(x.strip())
                y_out.append(y.strip())
            else:
                f.write(f"{x}\t{y}\n")

    assert len(x_out) == len(y_out)
    print("After removing sentences with too few or too many words, %i pairs remain" % len(x_out))
    return x_out, y_out


# Sentence pair words ratio exceeded remove
def sentence_words_ratio_remove(x_in, y_in):
    x_out = []
    y_out = []

    for (x, y) in zip(x_in, y_in):
        m_x = len(x.strip().split())
        m_y = len(y.strip().split())

        if m_x / m_y > src_tgt_words_ratio or m_y / m_x > src_tgt_words_ratio:
            continue
        x_out.append(x.strip())
        y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing sentence pair exceeds length ratio, %i pairs remain" % len(x_out))
    return x_out, y_out


# Specific punctuation number exceeded sentence remove
def specific_punc_remove(x_in, y_in):
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
    y_out = []

    for (x, y) in zip(x_in, y_in):
        if hot_fix_filter(x) and hot_fix_filter(y):
            x_out.append(x.strip())
            y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing sentences with too many specific punctuations, %i pairs remain" % len(x_out))
    return x_out, y_out


# Characters condition remove
def characs_remove(x_in, y_in):
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
    y_out = []

    for (x, y) in zip(x_in, y_in):
        if filter_by_len(x) and filter_by_len(y):
            x_out.append(x.strip())
            y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing sentence with characters condition, %i pairs remain" % len(x_out))
    return x_out, y_out


# Punctuation condition remove
def punctuation_remove(x_in, y_in):
    x_out = []
    y_out = []

    count_func = lambda l1, l2: sum([1 for x in l1 if x in l2])

    punctuation_set = set(punctuation)
    punct_density = 0
    punct_max_limit = 0
    punct_ratio = 0
    for (x, y) in zip(x_in, y_in):
        m_punc_x = count_func(x.strip(), set(punctuation_set))
        m_punc_y = count_func(y.strip(), set(punctuation_set))
        if m_punc_x >= min_punct_threshold and m_punc_y >= min_punct_threshold:
            if m_punc_x / (len(x.strip()) + 1e-9) > 0.5 or m_punc_y / (len(y.strip()) + 1e-9) > 0.5:
                punct_density += 1
                continue
            elif m_punc_x > punc_max_num or m_punc_y > punc_max_num:
                punct_max_limit += 1
                continue
            elif m_punc_x / (m_punc_y + 1e-9) > punc_ratio or m_punc_y / (m_punc_x + 1e-9) > punc_ratio:
                punct_ratio += 1
                continue
        x_out.append(x.strip())
        y_out.append(y.strip())
    assert len(x_out) == len(y_out)
    print(f"After removing sentences with too much punctuation, {len(x_out)} pairs remain")
    print(f"-- Punctuation-related sentence removal counts:")
    print(f"---- Exceeds punct. density: {punct_density}")
    print(f"---- Exceeds max. punct. limit ({punc_max_num}): {punct_max_limit}")
    print(f"---- Src/target punctuation ratio ({punc_ratio}): {punct_ratio}")
    return x_out, y_out


"""      if m_punc_x / (len(x.strip()) + 1e-9) > 0.5 or \
         m_punc_y / (len(y.strip()) + 1e-9) > 0.5 or \
         m_punc_x > punc_max_num or \
         m_punc_y > punc_max_num or \
         m_punc_x / (m_punc_y + 1e-9) > punc_ratio or \
         m_punc_y/ (m_punc_x + 1e-9) > punc_ratio:
        continue
"""


# Html address or tags contained sentence remove
def html_remove(x_in, y_in):
    x_out = []
    y_out = []

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

    for (x, y) in zip(x_in, y_in):
        if args.soft_html:
            x_out.append(soft_filter_by_html(x))
            y_out.append(soft_filter_by_html(y))
        else:
            if filter_by_html(x) or filter_by_html(y):
                x_out.append(x.strip())
                y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing sentences with html address or tags, %i pairs remain" % len(x_out))
    return x_out, y_out


# From Teacher Xia, special chars (hard to print)
def special_char_remove(x_in, y_in):
    x_out = []
    y_out = []

    for (x, y) in zip(x_in, y_in):
        if r"\x" in x or r"\x" in y:
            continue
        x_out.append(x.strip())
        y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing sentences with special characters, %i pairs remain" % len(x_out))
    return x_out, y_out


# Optional: Src/tgt chars ratio exceeded remove
def characs_sum_remove(x_in, y_in):
    x_out = []
    y_out = []

    for (x, y) in zip(x_in, y_in):
        segs_x = x.strip().split()
        m_char_x = sum([len(x) for x in segs_x])

        segs_y = y.strip().split()
        m_char_y = sum([len(y) for y in segs_y])

        if m_char_x / m_char_y > src_tgt_char_ratio or m_char_y / m_char_x > src_tgt_char_ratio:
            continue
        x_out.append(x.strip())
        y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing sentence with characters ratio condition, %i pairs remain" % len(x_out))
    return x_out, y_out


# Optional: Latin letter contained sentence remove
def latin_remove(x_in, y_in):
    def count_latin(sent):
        if len(re.findall("[^a-zA-Z]", sent)) / len(sent) > lattin_ratio:
            return False
        return True

    x_out = []
    y_out = []
    for (x, y) in zip(x_in, y_in):
        if count_latin(x.strip()) and count_latin(y.strip()):
            x_out.append(x.strip())
            y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print("After removing sentences with too many latin characters, %i pairs remain" % len(x_out))
    return x_out, y_out


filter_1 = []
filter_2 = []

fr_1 = open(f1, "r", encoding="utf8")
fr_2 = open(f2, "r", encoding="utf8")

f1_all_lines = fr_1.readlines()
f2_all_lines = fr_2.readlines()

filter_1, filter_2 = dup_remove(f1_all_lines, f2_all_lines)
filter_1, filter_2 = src_tgt_same_remove(filter_1, filter_2)
filter_1, filter_2 = sentence_word_num_remove(filter_1, filter_2)
filter_1, filter_2 = sentence_words_ratio_remove(filter_1, filter_2)
filter_1, filter_2 = specific_punc_remove(filter_1, filter_2)
filter_1, filter_2 = characs_remove(filter_1, filter_2)
filter_1, filter_2 = special_char_remove(filter_1, filter_2)
filter_1, filter_2 = punctuation_remove(filter_1, filter_2)
filter_1, filter_2 = html_remove(filter_1, filter_2)
filter_1, filter_2 = characs_sum_remove(filter_1, filter_2)
# filter_1, filter_2 = latin_remove(filter_1, filter_2)

fr_1.close()
fr_2.close()


fw_1 = open(f1 + ".clean", "w", encoding="utf8")
fw_2 = open(f2 + ".clean", "w", encoding="utf8")

assert len(filter_1) == len(filter_2)
print("After all filtering rules, remain %i pairs" % len(filter_1))

for x in filter_1:
    print(x, file=fw_1)

for y in filter_2:
    print(y, file=fw_2)

fw_1.close()
fw_2.close()
