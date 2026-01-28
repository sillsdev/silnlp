import argparse
from collections import Counter
import csv
import numpy
import pandas as pd
import re

from ..common.environment import SIL_NLP_ENV 

def is_word(word: str) -> bool:
    return any(char.isalpha() for char in word)

def main() -> None:
    parser = argparse.ArgumentParser(description="Counts lexicon entries")
    parser.add_argument("experiment", help="Experiment folder from path S:\\Alignment\\experiments\\")
    parser.add_argument("--aligner", help="Aligner: eflomal, fast-align, hmm", default="eflomal")
    parser.add_argument("--num", help="Number of most common words to include", type=int, default=100)
    parser.add_argument("--stats", help="Print word count and number of renderings for common words", 
                        action='store_true')
    parser.add_argument("--count", help="Include count in src word files", action='store_true')
    args = parser.parse_args()

    # Set up path and lex files
    lex_path = SIL_NLP_ENV.align_experiments_dir / args.experiment
    lex_txt_file = "lexicon."+args.aligner+".txt"
    new_lex_txt_file = "lexicon."+args.aligner+"_clean.txt"

    # Get source and target iso codes
    with (lex_path / "config.yml").open("r", encoding="utf8") as conf:
        for line in conf:
            if "src" in line.split(" ")[0]:
                src_iso = line.split(" ")[1].split("-")[0]
            elif "trg" in line.split(" ")[0]:
                trg_iso = line.split(" ")[1].split("-")[0]
    # TODO: error or use a default if it fails to get both iso codes
    
    # Look for commas with no following whitespace
    pattern = re.compile(r",(?=\S)")  

    # Pull all the separate words from the source data. Take most common and all unique.
    src_words = []
    with (lex_path / "src.txt").open("r", encoding = "utf8") as src_data_file:     
        for line in src_data_file:
            for word in line.split(" "):
                word = word.strip().strip("\'\"\\;,:.!?()-[]").lower()
                # Add space after commas as needed
                finder = pattern.search(word)
                if finder:
                    word = word[:finder.span()[1]]+" "+word[finder.span()[1]:]
                if word != "" and not word.isnumeric():
                    src_words.append(word)  
    src_data_word_counter = Counter(src_words).most_common(args.num)
    if args.count:
       src_word_counter = Counter(src_words).most_common()
    unique_src_words = numpy.unique(numpy.array(src_words))

    # Pull all the separate words from the target data. Take all unique.
    trg_words = [] 
    with (lex_path / "trg.txt").open("r", encoding = "utf8") as trg_data_file:     
        for line in trg_data_file:
            for word in line.split(" "):
                word = word.strip().strip("\'\"\\;,:.!?()-[]").lower()
                # Add space after commas as needed
                finder = pattern.search(word)
                if finder:
                    word = word[:finder.span()[1]]+" "+word[finder.span()[1]:]
                if word != "" and not word.isnumeric():
                    trg_words.append(word)  
    unique_trg_words = numpy.unique(numpy.array(trg_words))
    
    # Prep lexicon file for pandas csv reader (escape quotes)
    with (lex_path / lex_txt_file).open("r", encoding="utf8") as lexicon:
        with (lex_path / new_lex_txt_file).open("w", encoding="utf8") as new_lex:
            for line in lexicon.readlines():
                line = line.replace("'","\\'").replace("\"","\\\"")
                if is_word(line.split("\t")[0]):
                    new_lex.write(line)

    # Read the lexicon into a dataframe after escaping out quotes.
    # Find the most most diverse src words (most lexicon entries).
    lex_df = pd.read_csv(lex_path / new_lex_txt_file, sep = '\t')
    lex_df.columns = [src_iso, trg_iso, "percent"]
    lex_word_counter = Counter(lex_df[src_iso]).most_common(args.num)

    # Find all the renderings for the most diverse words.
    diverse_wd={}
    diverse_wd_renderings = 0
    for entry in lex_word_counter:
        diverse_wd_renderings += entry[1]
        word = entry[0]
        diverse_wd[word] = []
        for index, trg_word in enumerate(lex_df[trg_iso]):
            if word == lex_df[src_iso][index]:
                diverse_wd[word].append(lex_df[trg_iso][index])

    # Find all the renderings for the most common words.
    common_wd={}                # Dictionary of most common src words and trg renderings
    common_wd_instances = 0     # Instances of most common src words
    common_wd_renderings = 0    # Cumulative trg renderings for most common src words
    for entry in src_data_word_counter:
        common_wd_instances += entry[1]
        word = entry[0]
        common_wd[word] = []
        for index, trg_word in enumerate(lex_df[trg_iso]):
            if word == lex_df[src_iso][index]:
                common_wd[word].append(lex_df[trg_iso][index])
    for renderings in common_wd.values():
        common_wd_renderings += len(renderings)

    # Write the dictionary of renderings for the most common words to a .csv file in the experiment directory.
    with open(f"{lex_path}\\trg_renderings.csv", "w", encoding="utf8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([f'{src_iso} Word', f'{trg_iso} Words'])
        for src_wd in common_wd:
            writer.writerow([src_wd, *common_wd[src_wd]])

    with (lex_path / "src_words.txt").open("w", encoding = "utf8") as output_file:  
            if args.count:
                for entry in src_word_counter:
                    output_file.writelines(entry[0] + '\t' + str(entry[1]) + '\n')
            else:
                for word in unique_src_words:
                    output_file.writelines(word + '\n')

    # Optionally, output a few stats
    if args.stats:
        print(f"\nSource Data: {len(src_words)} words and {len(unique_src_words)} unique words.")
        print(f"Target Data: {len(trg_words)} words and {len(unique_trg_words)} unique words.")
        print(f"\nThe {args.num} most common source words in the dataset appear an average of "
              f"{round((common_wd_instances)/args.num)} times per word.")
        print(f"The {args.num} most common source words in the dataset have an average of "
              f"{round((common_wd_renderings)/args.num)} target renderings per word.")
        print(f"The {args.num} most diverse source words in the dataset have an average of "
              f"{round((diverse_wd_renderings)/args.num)} renderings per word.\n")

    
    # Print the "score" and write to file
    score = round(100*(1/((common_wd_renderings)/args.num)))
    print(f"Internal Consistency score is {score}.\n  Score of 100 indicates that the top source words "
          f"each average 1 target rendering.\n  Score = 100*1/(average trg_words per most common src_word)")

    with (lex_path / "lex_stats.csv").open("w", encoding="utf8") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(['#_Src_wds','#_trg_wds','Unique_src','Unique_trg','Num_top_wds','Avg_inst',
                         'avg_trg_renderings','avg_diverse_renderings'])
        writer.writerow([len(src_words),len(trg_words),len(unique_src_words),len(unique_trg_words),args.num,
                         round((common_wd_instances)/args.num),round((common_wd_renderings)/args.num),
                         round((diverse_wd_renderings)/args.num)]) 


if __name__ == "__main__":
    main()

