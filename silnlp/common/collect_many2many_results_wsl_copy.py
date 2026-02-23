import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import yaml
import openpyxl

from silnlp.common.environment import SIL_NLP_ENV

LOGGER = logging.getLogger(__package__ + ".collect_many2many_results")

def extract_target_language(folder_name):
    """Extract target language from folder name (everything before the last underscore)."""
    parts = folder_name.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else folder_name

def process_folder(folder_path):
    results = []
    
    # Iterate through subdirectories
    for subfolder in folder_path.iterdir():
        if not subfolder.is_dir() or subfolder.name.lower() in ['align', 'alignment', 'alignments', 'analyze', 'analyse']:
            continue
            
        config_file = subfolder / "config.yml"
        if not config_file.exists():
            continue
            
        LOGGER.info(f"Processing folder: {subfolder.name}")
        
        # Read config.yml
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        data_config = config.get("data", {})
        corpus_pair = data_config.get("corpus_pairs", [{}])[0]
        
        target_language = extract_target_language(subfolder.name)
        src_list = corpus_pair.get("src", [])
        src1 = src_list[0] if len(src_list) > 0 else ""
        src2 = src_list[1] if len(src_list) > 1 else ""
        trg = corpus_pair.get("trg", "")
        corpus_books = corpus_pair.get("corpus_books", "")
        test_books_str = corpus_pair.get("test_books", "")
        mapping = corpus_pair.get("mapping", "")
        
        test_books_list = [b.strip() for b in test_books_str.split(";") if b.strip()]
        scores_file = subfolder / "scores-5000.csv"
        
        if scores_file.exists():
            # Read scores-5000.csv
            scores_df = pd.read_csv(scores_file)
            # Standardize column names if needed (handle case sensitivity)
            cols_map = {col: col for col in scores_df.columns}
            bleu_col = next((c for c in scores_df.columns if c.lower() == "bleu"), None)
            chrf_col = next((c for c in scores_df.columns if c.lower() == "chrf3++"), None)
            book_col = next((c for c in scores_df.columns if c.lower() == "book"), None)
            
            for index, row in scores_df.iterrows():
                book = str(row[book_col]) if book_col else ""
                # Skip 'ALL' or empty books if we only want specific books, 
                # but usually scores-*.csv has rows per book.
                
                results.append({
                    "Target_language": target_language,
                    "Mapping": mapping,
                    "Source 1": src1,
                    "Source 2": src2,
                    "Target": trg,
                    "corpus_books": corpus_books,
                    "test_books": test_books_str,
                    "Book": book,
                    "BLEU": row[bleu_col] if bleu_col else None,
                    "chrF3++": row[chrf_col] if chrf_col else None
                })
        else:
            # No scores file, add placeholder rows for experiment
            results.append({
                "Target_language": target_language,
                "Mapping": mapping,
                "Source 1": src1,
                "Source 2": src2,
                "Target": trg,
                "corpus_books": corpus_books,
                "test_books": test_books_str,
                "Book": None,
                "BLEU": None,
                "chrF3++": None
            })
                
    return results

def main():
    parser = argparse.ArgumentParser(description="Collect results from many-to-many NLLB experiments.")
    parser.add_argument("folder", help="Name of the folder in MT experiments directory.")
    
    args = parser.parse_args()
    
    target_folder = Path(args.folder)
    if not target_folder.is_absolute():
        target_folder = SIL_NLP_ENV.mt_experiments_dir / args.folder
        
    if not target_folder.is_dir():
        LOGGER.error(f"Directory not found: {target_folder}")
        return 1
        
    LOGGER.info(f"Scanning directory: {target_folder}")
    
    results_data = process_folder(target_folder)
    
    if not results_data:
        LOGGER.warning("No experiment results found.")
        return 0
        
    df = pd.DataFrame(results_data)
    
    output_file = target_folder / f"{args.folder}-results.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        
    # Formatting
    wb = openpyxl.load_workbook(output_file)
    ws = wb['Results']
    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter # Get the column name
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        ws.column_dimensions[column].width = adjusted_width
    wb.save(output_file)
    
    LOGGER.info(f"Results saved to: {output_file}")
    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
