import argparse
from pathlib import Path
from .environment import SIL_NLP_ENV
from machine.scripture import ALL_BOOK_IDS, book_id_to_number, is_nt, is_ot

OT_CANON = [book for book in ALL_BOOK_IDS if is_ot(book_id_to_number(book))]
NT_CANON = [book for book in ALL_BOOK_IDS if is_nt(book_id_to_number(book))]
DT_CANON = [
    "TOB",
    "JDT",
    "ESG",
    "WIS",
    "SIR",
    "BAR",
    "LJE",
    "S3Y",
    "SUS",
    "BEL",
    "1MA",
    "2MA",
    "3MA",
    "4MA",
    "1ES",
    "2ES",
    "MAN",
    "PS2",
    "ODA",
    "PSS",
    "EZA",
    "JUB",
    "ENO",
]

counts = {'GEN': 1533, 'EXO': 1213, 'LEV': 859, 'NUM': 1289, 'DEU': 959, 'JOS': 658, 'JDG': 618, 'RUT': 85, '1SA': 811, '2SA': 695, '1KI': 817, '2KI': 719, '1CH': 943, '2CH': 822, 'EZR': 280, 'NEH': 405, 'EST': 167, 'JOB': 1070, 'PSA': 2527, 
'PRO': 915, 'ECC': 222, 'SNG': 117, 'ISA': 1291, 'JER': 1364, 'LAM': 154, 'EZK': 1273, 'DAN': 357, 'HOS': 197, 'JOL': 73, 'AMO': 146, 'OBA': 21, 'JON': 48, 'MIC': 105, 'NAM': 47, 'HAB': 56, 'ZEP': 53, 'HAG': 38, 'ZEC': 211, 'MAL': 55, 'MAT': 1071, 'MRK': 678, 'LUK': 1151, 'JHN': 879, 'ACT': 1006, 'ROM': 433, '1CO': 437, '2CO': 256, 'GAL': 149, 'EPH': 155, 'PHP': 104, 'COL': 95, '1TH': 89, '2TH': 47, '1TI': 113, '2TI': 83, 'TIT': 46, 'PHM': 25, 'HEB': 303, 'JAS': 108, '1PE': 105, '2PE': 61, '1JN': 105, '2JN': 13, '3JN': 15, 'JUD': 25, 'REV': 405, 'TOB': 248, 'JDT': 340, 'ESG': 267, 'WIS': 435, 'SIR': 1401, 'BAR': 141, 'LJE': 72, 'S3Y': 67, 'SUS': 64, 'BEL': 42, '1MA': 924, '2MA': 555, '3MA': 228, '4MA': 482, '1ES': 434, '2ES': 944, 'MAN': 15, 'PS2': 7, 'ODA': 275, 'PSS': 293, 'EZA': 715, 'JUB': 1217, 'ENO': 1563}


def get_lines(file) -> list[str]:
    with open(file, 'r', encoding='utf-8') as f:
        return [line for line in f.readlines()]


def count_verses_from_vref(vref_path):
    "Count total verses per book from vref.txt"
    counts = {}
    for line in get_lines(vref_path):
        book = line.split()[0]
        counts[book] = counts.get(book, 0) + 1
    return counts

def count_verses_from_extracted(extract_path, vref_path):
    "Count non-empty verses per book from extracted file"
    counts = {}
    vrefs = get_lines(vref_path)
    lines = get_lines(extract_path)
    
    for vref, line in zip(vrefs, lines):
        if len(line) > 3:
            book = vref.split()[0]
            counts[book] = counts.get(book, 0) + 1
    return counts

def check_complete_from_extracted(extract_path, vref_path, fudge=0.99):
    "Check which books are complete from extracted file"
    complete_counts = count_verses_from_vref(vref_path)
    actual_counts = count_verses_from_extracted(extract_path, vref_path)
    
    result = []
    ot_complete = sum(complete_counts.get(b, 0) for b in OT_CANON)
    ot_actual = sum(actual_counts.get(b, 0) for b in OT_CANON)
    nt_complete = sum(complete_counts.get(b, 0) for b in NT_CANON)
    nt_actual = sum(actual_counts.get(b, 0) for b in NT_CANON)
    
    if ot_actual >= ot_complete * fudge: result.append('OT')
    else: result.extend([b for b in OT_CANON if actual_counts.get(b, 0) >= complete_counts.get(b, 0) * fudge])
    
    if nt_actual >= nt_complete * fudge: result.append('NT')
    else: result.extend([b for b in NT_CANON if actual_counts.get(b, 0) >= complete_counts.get(b, 0) * fudge])
    
    return ';'.join(result)

def main() -> None:
    parser = argparse.ArgumentParser( prog="list_books", description="List the complete books from a scripture file.")
    parser.add_argument("filename", type=str, help="The scripture file name.")
    args = parser.parse_args()

    vref_path = SIL_NLP_ENV.assets_dir / "vref.txt"
    scripture_dir = SIL_NLP_ENV.mt_scripture_dir

    filename = args.filename

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    file = scripture_dir / filename
    if not file.is_file():
        print(f"Couldn't find file: {file}.")
        raise RuntimeError
    else:
#        counts = count_verses_from_vref(vref_path)
#        print(counts)
#        actual_counts = count_verses_from_extracted(file, vref_path)
#        print(actual_counts)
#        exit()
        print(f"Found file: {file}")
        print(f"Complete books are:\n{check_complete_from_extracted(file,vref_path)}")

if __name__ == "__main__":
    main()