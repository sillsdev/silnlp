import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import List

from lxml import etree
from machine.scripture import book_number_to_id, get_chapters

from .. import sfm
from ..common.paratext import get_book_path, get_project_dir
from ..common.translator import collect_segments, get_stylesheet
from ..sfm import usfm
from .collect_verse_counts import DT_canon, NT_canon, OT_canon

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)
valid_books.extend(DT_canon)

LOGGER = logging.getLogger(__package__ + ".translate")


def get_sfm_files(project_dir):
    return [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]]


def parse_book_mp(project_dir):
    ''' The main function of parse_book is to count the number of verses in each book of a Paratext project..
        This function reads the files and returns a list of strings with the number of verses or a
        string with an error message.
        The function only reads utf-8 files. Any non UTF-8 files will throw an error.
    '''
    results = list()

    sfm_files = get_sfm_files(project_dir)
    print(f"{project_dir}")
    out_dir = Path("E:/Work/Corpora/Checks")
    books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]
    books_to_check = [book for book in valid_books if book in books_found]
    stylesheet = get_stylesheet(project_dir)

    for book in books_to_check:
        book_path = get_book_path(project_dir, book)

        if not book_path.is_file():
            results.append(f"Can't find file {book_path}")
        else:
            out_file = book_path.with_suffix(".txt")
            print(f"out_file is {out_file}.")
            exit()

            if out_file.exists():
                continue
            else:
                with open(out_file, 'w', encoding="utf-8-sig") as f_out:
                    with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
                        try:
                            doc: List[sfm.Element] = list(
                                usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False)
                            )
                        except Exception as e:
                            f_out.write(f"Couldn't parse book {book}\tError is: {e}\n")
                            #results.append(f"Couldn't parse book {book}\n Error is: {e}")

                        book = ""
                        for elem in doc:
                            if elem.name == "id":
                                book = str(elem[0]).strip()[:3]
                                break
                        if book == "":
                            f_out.write(f"The file {book_path} doesn't contain an id marker.\n")
                            #results.append(f"The file {book_path} doesn't contain an id marker.")

                        segments = collect_segments(book, doc)
                        vrefs = [s.ref for s in segments]
                        f_out.write(f"{book} contains {len(vrefs)} verses.\n")
                        #results.append(f"{book} contains {len(vrefs)} verses.")

def main() -> None:

    parser = argparse.ArgumentParser(
        prog="check_books",
        description="Checks sfm files for a project with the same parser as translate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
   
    projects_dir = get_project_dir("")
    projects = [project for project in projects_dir.glob("*") if project.is_dir()][:200]
    print(f"Found {len(projects)} folders in {projects_dir}")

    no_of_cpu = 20
    print(f"Number of processors: {mp.cpu_count()} using {no_of_cpu}")
    pool = mp.Pool(no_of_cpu)
    
    #Keep a dictionary of the results
    all_results = dict()
    filecount = 0
    sys.stdout.flush()
    
    # Iterate over projects with multiple processors.
    results = pool.map(parse_book_mp, [project for project in projects[:20]])
    
    pool.close()
    #print(results)
    #exit()

    # with open("E:\Work\Corpora\PT_project_errors.txt", "w", encoding="utf-8",) as error_file:
    #     for result in results:
    #         for project_dir in result:
    #             print(f"{project_dir}")
                
    #         # print("This is a single result:")
    #         # print(result)
    #         # print(f"result is a :{type(result)}")        
    #         # print(result.keys(), result.values)

    #             for line in result[project_dir]:
    #                 #print(f"{project_dir}\t{line}\n")
    #                 error_file.write(f"{project_dir}\t{line}\n")


if __name__ == "__main__":
    main()
