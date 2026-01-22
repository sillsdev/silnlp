from pathlib import Path
import yaml
def count_ranges(fpath):
    "Count lines containing only <range>"
    return sum(1 for line in Path(fpath).read_text(encoding='utf-8').splitlines() if line.strip() == '<range>')
def process_config(cfg_path, base_dir=Path('M:/MT/scripture')):
    "Count ranges in all corpus files from config"
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    results = {}
    for pair in cfg['data']['corpus_pairs']:
        for src in (pair['src'] if isinstance(pair['src'], list) else [pair['src']]):
            fpath = base_dir / f"{src}.txt"
            if fpath.exists(): results[src] = count_ranges(fpath)
        trg = pair['trg']
        fpath = base_dir / f"{trg}.txt"
        if fpath.exists(): results[trg] = count_ranges(fpath)
    return results


def main():
    return

if __name__ == "__main__":
    main()