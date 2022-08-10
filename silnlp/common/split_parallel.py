import os
import argparse
import yaml
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

from ..common.utils import get_git_revision_hash, get_mt_exp_dir, merge_dict, set_seed
from ..common.corpus import load_corpus, write_corpus

_DEFAULT_SPLIT_CONFIG: dict = {
    "split": {
    },
}


def load_config(exp_dir: str, config_file_name: str) -> dict:
    config = _DEFAULT_SPLIT_CONFIG.copy()

    if config_file_name is None:
        config_path = os.path.join(exp_dir, "config.yml")
    else:
        config_path = os.path.join(exp_dir, config_file_name)

    if not os.path.isfile(config_path):
        print(f"Warning: config file {config_path} not found; using defaults")
        return config

    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)


def write_config(exp_dir: str, config_file_name: str, config: dict):
    with open(os.path.join(exp_dir, config_file_name), "w", encoding="utf-8") as file:
        yaml.dump(config, file)


def show_config(config: dict):
    print(json.dumps(config, indent=2))


def keep_lines(src, trg):
    src = src.strip()
    trg = trg.strip()

    if src == '' or trg == '' or src == trg:
        return False
    else:
        return True
        

def main() -> None:
    parser = argparse.ArgumentParser(description="Splitting a parallel corpus")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument('--config', type=str, default='config.yml', help='config file')
    args = parser.parse_args()

    rev_hash = get_git_revision_hash()

    exp_name: str = args.experiment
    exp_dir = get_mt_exp_dir(exp_name)
    config = load_config(exp_dir, args.config)
    split_config = config.get('split')
    write_config(exp_dir, f'effective_config-{rev_hash}.split.yml', split_config)

    set_seed(split_config["seed"])

    # Create the initial data frame
    corpus = pd.DataFrame(columns=['SRC', 'TRG'])
    print(f'Loading corpus - src: {split_config.get("src")}, trg: {split_config.get("trg")}')
    src_lines = list(load_corpus(Path(os.path.join(exp_dir, split_config.get('src')))))
    trg_lines = list(load_corpus(Path(os.path.join(exp_dir, split_config.get('trg')))))
    
    # Remove lines where one or other of the lines are blank, or both lines are identical.
    filtered_src = list()
    filtered_trg = list()
    
    for src, trg in zip(src_lines, trg_lines):
        if keep_lines(src, trg):
            filtered_src.append(src)
            filtered_trg.append(trg)
                   
    corpus['SRC'] = filtered_src
    corpus['TRG'] = filtered_trg

    remainder = None

    splits = split_config.get('splits')
    for split in splits:
        if split.get('type') == 'test':
            print(f'Creating test set - src: {split.get("src")}, trg: {split.get("trg")}, size: {split.get("size")}')
            remainder, test = train_test_split(corpus, test_size=split.get('size'))
            write_corpus(Path(os.path.join(exp_dir, split.get('src'))), test['SRC'])
            write_corpus(Path(os.path.join(exp_dir, split.get('trg'))), test['TRG'])
            break

    for split in splits:
        if split.get('type') == 'val':
            print(f'Creating val set - src: {split.get("src")}, trg: {split.get("trg")}, size: {split.get("size")}')
            remainder, val = train_test_split(remainder, test_size=split.get('size'))
            write_corpus(Path(os.path.join(exp_dir, split.get('src'))), val['SRC'])
            write_corpus(Path(os.path.join(exp_dir, split.get('trg'))), val['TRG'])
            break

    for split in splits:
        if split.get('type') == 'train':
            train_size = split.get('size')
            print(f'Creating train set - src: {split.get("src")}, trg: {split.get("trg")}, size: {train_size}')
            if isinstance(train_size, int):
                _, train = train_test_split(remainder if remainder is not None else corpus, test_size=train_size)
            elif isinstance(train_size, str) and train_size == 'all':
                train = remainder if remainder is not None else corpus
            else:
                print(f'Invalid training split size: {train_size}')
                continue
            write_corpus(Path(os.path.join(exp_dir, split.get('src'))), train['SRC'])
            write_corpus(Path(os.path.join(exp_dir, split.get('trg'))), train['TRG'])


if __name__ == "__main__":
    main()
