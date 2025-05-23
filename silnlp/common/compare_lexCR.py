import argparse
from collections import Counter
import numpy
import re
from typing import List

from ..common.environment import SIL_NLP_ENV
from machine.tokenization import LatinWordTokenizer

# Latin Tokenizer from machine library
#def get_all_words(src_file: str) -> List:
#    words = []
#    tokenizer = LatinWordTokenizer()
#    with open(src_file, "r", encoding = "utf8") as src_data_file:
#        for line in src_data_file:
#            line_words = tokenizer.tokenize(line)
#            for word in line_words:
#                word = word.strip().strip("\'\"\\;,:.!?()-[]").lower()
#                if word != "" and not word.isnumeric():
#                    words.append(word)
#    return words

# Naive whitespace-based script-agnostic word splitter
def get_all_words(src_file: str) -> List:
    words = []
    pattern = re.compile(r",(?=\S)")  # Look for commas with no following space
    with open(src_file, "r", encoding = "utf8") as src_data_file:     
        for line in src_data_file:
            for word in line.split(" "):
                word = word.strip().strip("\'\"\\;,:.!?()-[]0123456789").lower()
                finder = pattern.search(word)
                if finder:             # Add space after commas as needed
                    word = word[:finder.span()[1]]+" "+word[finder.span()[1]:]
                if word != "":
                    words.append(word)  
    return words

def find_unique(words1: List, words2: List) -> List:
    unique_words = []
    for word in words1:
        if word not in words2:
            unique_words.append(word)
    return unique_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Compares unique words in two corpora")
    parser.add_argument("exp1", help="First experiment folder from path S:\\Alignment\\experiments\\")
    parser.add_argument("exp2", help="Second experiment folder from path S:\\Alignment\\experiments\\")
    parser.add_argument("--stats", help="True or False: Output word count and number of renderings for common words", 
                        action='store_true')
    parser.add_argument("--src", help="If set, only the source side of the two experiment lexicons is compared", 
                        action='store_true')
    parser.add_argument("--trg", help="If set, only the target side of the two experiment lexicons is compared", 
                        action='store_true')
    args = parser.parse_args()

    # If not explicitly limited, compare both source and target lexicons
    if args.src == False and args.trg == False:
        args.src = True
        args.trg = True

    lex_path1 = SIL_NLP_ENV.align_experiments_dir / args.exp1
    lex_path2 = SIL_NLP_ENV.align_experiments_dir / args.exp2

    # Compare source words and write results to files
    if args.src == True:
        src_file1 = lex_path1 / "src.txt"
        src_file2 = lex_path2 / "src.txt"

        # Find all words and unique words on source side
        src_words1 = get_all_words(src_file1)
        unique_src_words1 = numpy.unique(numpy.array(src_words1))
        src_words2 = get_all_words(src_file2)
        unique_src_words2 = numpy.unique(numpy.array(src_words2))
        src1_only_words = find_unique(unique_src_words1,unique_src_words2)
        src2_only_words = find_unique(unique_src_words2,unique_src_words1)
        src1_word_counter = Counter(src_words1).most_common()
        src2_word_counter = Counter(src_words2).most_common()

        # Write unique source words to files
        src_words_file1 = lex_path1 / "src_words.txt"
        src_words_file2 = lex_path2 / "src_words.txt"
        with open(src_words_file1, "w", encoding="utf8") as output_file:
            for word in unique_src_words1:
                output_file.writelines(word+'\n')
        with open(src_words_file2, "w", encoding="utf8") as output_file:
            for word in unique_src_words2:
                output_file.writelines(word+'\n')

        # Re-write src_words files with counts
        with (lex_path1 / "src_words.txt").open("w", encoding = "utf8") as output_file:  
            for entry in src1_word_counter:
                output_file.writelines(entry[0] + '\t' + str(entry[1]) + '\n')
        with (lex_path2 / "src_words.txt").open("w", encoding = "utf8") as output_file:  
            for entry in src2_word_counter:
                output_file.writelines(entry[0] + '\t' + str(entry[1]) + '\n')

        # Write source words missing from the alternate source file
        #with (lex_path1 / "unmatched_src_words.txt").open("w", encoding="utf8") as output_file:
        #    output_file.writelines(f'src.txt words not found in {src_file2}\n')
        #    for word in src1_only_words:
        #        output_file.writelines(word+'\n')
        #with (lex_path2 / "unmatched_src_words.txt").open("w", encoding="utf8") as output_file:
        #    output_file.writelines(f'src.txt words not found in {src_file1}\n')
        #    for word in src2_only_words:
        #        output_file.writelines(word+'\n')


        # Rewrite of above section to include counts in the output file: 
        with (lex_path1 / "unmatched_src_words.txt").open("w", encoding="utf8") as output_file:
            output_file.writelines(f'src.txt words not found in {src_file2}\n')
            for entry in src1_word_counter:
                if entry[0] in src1_only_words:
                    output_file.writelines(entry[0] + '\t' + str(entry[1]) + '\n')
        with (lex_path2 / "unmatched_src_words.txt").open("w", encoding="utf8") as output_file:
            output_file.writelines(f'src.txt words not found in {src_file1}\n')
            for entry in src2_word_counter:
                if entry[0] in src2_only_words:
                    output_file.writelines(entry[0] + '\t' + str(entry[1]) + '\n')

    # Compare target words and write results to files
    if args.trg == True:
        trg_file1 = lex_path1 / "trg.txt"
        trg_file2 = lex_path2 / "trg.txt"

        # Find all words and unique words on target side
        trg_words1 = get_all_words(trg_file1)
        unique_trg_words1 = numpy.unique(numpy.array(trg_words1))
        trg_words2 = get_all_words(trg_file2)
        unique_trg_words2 = numpy.unique(numpy.array(trg_words2))
        trg1_only_words = find_unique(unique_trg_words1,unique_trg_words2)
        trg2_only_words = find_unique(unique_trg_words2,unique_trg_words1)

        # Write unique target words to files
        trg_words_file1 = lex_path1 / "trg_words.txt"
        trg_words_file2 = lex_path2 / "trg_words.txt"
        with open(trg_words_file1, "w", encoding="utf8") as output_file:
            for word in unique_trg_words1:
                output_file.writelines(word+'\n')
        with open(trg_words_file2, "w", encoding="utf8") as output_file:
            for word in unique_trg_words2:
                output_file.writelines(word+'\n')

        # Write target words missing from the alternate target file
        with (lex_path1 / "unmatched_trg_words.txt").open("w", encoding="utf8") as output_file:
            output_file.writelines(f'trg.txt words not found in {trg_file2}\n')
            for word in trg1_only_words:
                output_file.writelines(word+'\n')
        with (lex_path2 / "unmatched_trg_words.txt").open("w", encoding="utf8") as output_file:
            output_file.writelines(f'trg.txt words not found in {trg_file1}\n')
            for word in trg2_only_words:
                output_file.writelines(word+'\n')
    
    # Write the lex coverage stats
    with (lex_path1 / "lex_coverage.txt").open("a", encoding="utf8") as output_file:
        if args.src == True:
            output_file.writelines(f'Unique words in src.txt: {len(unique_src_words1)}\n')
            output_file.writelines(
                f'Words also found in {src_words_file2}: {len(unique_src_words1)-len(src1_only_words)}\n')
            output_file.writelines(f'Words missing from {src_words_file2}: {len(src1_only_words)}\n')
        if args.trg == True:
            output_file.writelines(f'Unique words in trg.txt: {len(unique_trg_words1)}\n')
            output_file.writelines(
                f'Words also found in {trg_words_file2}: {len(unique_trg_words1)-len(trg1_only_words)}\n')
            output_file.writelines(f'Words missing from {trg_words_file2}: {len(trg1_only_words)}\n')

    with (lex_path2 / "lex_coverage.txt").open("a", encoding="utf8") as output_file:
        if args.src == True:
            output_file.writelines(f'Unique words in src.txt: {len(unique_src_words2)}\n')
            output_file.writelines(
                f'Words also found in {src_words_file1}: {len(unique_src_words2)-len(src2_only_words)}\n')
            output_file.writelines(f'Words missing from {src_words_file1}: {len(src2_only_words)}\n')
        if args.trg == True:
            output_file.writelines(f'Unique words in trg.txt: {len(unique_trg_words2)}\n')
            output_file.writelines(
                f'Words also found in {trg_words_file1}: {len(unique_trg_words2)-len(trg2_only_words)}\n')
            output_file.writelines(f'Words missing from {trg_words_file1}: {len(trg2_only_words)}\n')

    # Output stats if requested
    if args.stats == True:
        if args.src == True:
            print(f'Unique words in src.txt: {len(unique_src_words1)}')
            print(f'Words also found in {src_words_file2}: {len(unique_src_words1)-len(src1_only_words)}')
            print(f'Words missing from {src_words_file2}: {len(src1_only_words)}')
        if args.trg == True:
            print(f'Unique words in {trg_words_file1}: {len(unique_trg_words1)}')
            print(f'Words also found in {trg_words_file2}: {len(unique_trg_words1)-len(trg1_only_words)}')
            print(f'Words missing from {trg_words_file2}: {len(trg1_only_words)}')

    
if __name__ == "__main__":
    main()
