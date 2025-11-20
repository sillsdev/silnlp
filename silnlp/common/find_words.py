import argparse
from collections import Counter
import csv
import unicodedata

from ..common.environment import SIL_NLP_ENV

# Normalize combined characters for Devanagari (Ref: https://docs.python.org/3/howto/unicode.html#comparing-strings)
def NFD(s):
    return unicodedata.normalize('NFD', s)

def main():
    parser = argparse.ArgumentParser(description="Counts lexicon entries")
    parser.add_argument("experiment", help="Experiment folder from path S:\\Alignment\\experiments\\")
    parser.add_argument("--word_list", help="File containing words to find", default="unmatched_src_words.txt")
    args = parser.parse_args()

    # Set up path and files
    lex_path = SIL_NLP_ENV.align_experiments_dir / args.experiment
    word_filename = SIL_NLP_ENV.align_experiments_dir / args.experiment / args.word_list
    vref_filename = SIL_NLP_ENV.align_experiments_dir / args.experiment/ "refs.txt"

    # Get count of each word in the file
    with (lex_path / "src_words.txt").open("r", encoding="utf8") as src_wd_file:
        src_word_counts = []
        for entry in src_wd_file:
            entry = list(entry.split('\t'))
            if len(entry) > 1:
                    entry[1] = int(entry[1].strip())
                    src_word_counts.append(entry)
            else:
                print("Error: word counts are missing. Please run count_words.py with the --count flag set.")
                return 1

    # Extract list of words
    src_word_dict = dict(list(src_word_counts))
    with(word_filename).open("r", encoding = "utf8") as word_file:
        words = []
        for word in word_file:
            words.append(word.rstrip('\n'))
    # Check for words and word count in each verse; write to output file.
    with (lex_path / "src.txt").open("r", encoding = "utf8") as src_data_file:
            with(vref_filename).open("r", encoding = "utf8") as ref_file:
                word_list = list(enumerate(words))
                result = []
                seen_words = []
                for verse in zip(ref_file, src_data_file):
                    word_text = []
                    word_num = []
                    word_count = 0
                    for word in word_list:
                        #if NFD(NFD(word[1])) in NFD(NFD(verse[1])):
                        #if word[1] in verse[1]: # (to find all instances; not just first)
                        if word[1] in verse[1] and word[1] not in seen_words:
                            for entry in src_word_counts:
                                 if entry[0] == word[1]:
                                      word_count += entry[1]
                            seen_words.append(word[1])
                            word_text.append(word[1])
                            word_num.append(src_word_dict[word[1]])
                    result.append([verse[0].rstrip('\n'), word_count, word_num, word_text])
    with (lex_path / "unmatched_word_verses.txt").open("w", encoding = "utf8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(['Reference','Novelty Score','Word Counts','Words'])
        for line in result:
            writer.writerow([line[0], line[1], line[2], *line[3]])
    #print(result)


if __name__ == '__main__':
    main()
