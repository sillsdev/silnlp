"""Create NMT experiment folders from a production onboarding request.

Parses the onboarding.log written by silnlp.common.onboard_project in a
MT/experiments/_OnboardingRequests/<request> folder, selects the reference
projects whose alignment stats pass the thresholds, and creates
<Country>/<Language>/<experiment> folders containing config.yml and
translate_config.yml. See create_onboarding_experiments_plan.md for details.
"""

import argparse
import itertools
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from machine.scripture import ALL_BOOK_IDS, book_id_to_number, is_nt, is_ot

from .environment import SilNlpEnv
from .iso_info import ALT_ISO, NLLB_TAG_FROM_ISO

LOGGER = logging.getLogger(__package__ + ".create_onboarding_experiments")

OT_CANON = [book for book in ALL_BOOK_IDS if is_ot(book_id_to_number(book))]
NT_CANON = [book for book in ALL_BOOK_IDS if is_nt(book_id_to_number(book))]

MAIN_PROJECT_RE = re.compile(r"Processing onboarding request for main project '([^']+)'")
EXTRACT_RE = re.compile(r"Extracted corpus file: .*[/\\]([^/\\]+)\.txt\s*$")
VERSES_RE = re.compile(r"# of Verses: (\d+)")
# "beteween" is how onboard_project's analyze step spells it in the log.
ALIGN_RE = re.compile(r"Computing alignment beteween (\S+) and (\S+) using")
STATS_RE = re.compile(
    r"(?P<main>\S+) -> (?P<ref>\S+) stats - count: (?P<count>\d+),"
    r".*?parallel count: (?P<parallel>\d+) alignment: (?P<alignment>[\d.]+),"
    r".*?source script: (?P<src_script>[^,]+),"
    r".*?target script: (?P<trg_script>[^,]+),"
)

CHECKPOINT = 5000
SEED = 111
MODEL = "facebook/nllb-200-distilled-1.3B"
BOOK_COMPLETENESS_THRESHOLD = 0.98
MAX_UNPROMPTED_MIXED = 3

EXPERIMENT_ARGS = [
    "-m",
    "silnlp.nmt.experiment",
    "--save-checkpoints",
    "--save-confidences",
    "--clearml-queue",
    "jobs_urgent",
    "--clearml-tag",
    "eitl",
    "--preprocess",
    "--stats",
    "--train",
    "--test",
    "--translate",
]


@dataclass
class Candidate:
    """A reference project aligned against the main project in the log."""

    name: str
    stem: str  # extract file stem, e.g. en-NIV11R
    iso: str  # iso prefix exactly as it appears in the filename
    count: int
    parallel: int
    alignment: float
    script: str


@dataclass
class MainProject:
    name: str
    stem: str
    iso: str
    verses: Optional[int]
    script: Optional[str]


@dataclass
class Experiment:
    sources: List[Candidate]
    folder: Path
    config: dict
    translate_config: dict


def parse_log(log_path: Path) -> Tuple[MainProject, List[Candidate]]:
    main_name: Optional[str] = None
    stems: Dict[str, str] = {}  # project name -> extract stem
    verses: Dict[str, int] = {}  # extract stem -> # of Verses
    stats: List[dict] = []
    last_stem: Optional[str] = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = MAIN_PROJECT_RE.search(line)
        if m is not None and main_name is None:
            main_name = m.group(1)
            continue
        m = EXTRACT_RE.search(line)
        if m is not None:
            last_stem = m.group(1)
            stems[stem_to_project(last_stem)] = last_stem
            continue
        m = VERSES_RE.search(line)
        if m is not None and last_stem is not None:
            verses[last_stem] = int(m.group(1))
            last_stem = None
            continue
        m = ALIGN_RE.search(line)
        if m is not None:
            for stem in m.groups():
                stems[stem_to_project(stem)] = stem
            continue
        m = STATS_RE.search(line)
        if m is not None:
            stats.append(m.groupdict())

    if main_name is None:
        raise ValueError(f"No 'Processing onboarding request for main project' line found in {log_path}.")
    main_stem = stems.get(main_name)
    if main_stem is None:
        raise ValueError(f"No extract file or alignment line found for main project '{main_name}' in {log_path}.")

    candidates: Dict[str, Candidate] = {}
    main_script: Optional[str] = None
    for entry in stats:
        if entry["main"] != main_name:
            continue
        main_script = entry["src_script"].strip()
        ref_name = entry["ref"]
        ref_stem = stems.get(ref_name)
        if ref_stem is None:
            LOGGER.warning(f"No extract stem found for aligned project '{ref_name}'. Skipping it.")
            continue
        candidates[ref_name] = Candidate(
            name=ref_name,
            stem=ref_stem,
            iso=stem_to_iso(ref_stem),
            count=int(entry["count"]),
            parallel=int(entry["parallel"]),
            alignment=float(entry["alignment"]),
            script=entry["trg_script"].strip(),
        )

    main = MainProject(
        name=main_name,
        stem=main_stem,
        iso=stem_to_iso(main_stem),
        verses=verses.get(main_stem),
        script=main_script,
    )
    return main, list(candidates.values())


def stem_to_iso(stem: str) -> str:
    return stem.split("-", 1)[0]


def stem_to_project(stem: str) -> str:
    return stem.split("-", 1)[1] if "-" in stem else stem


def to_iso3(iso: str) -> Optional[str]:
    if len(iso) == 3:
        return iso
    return ALT_ISO.get_alternative(iso)


def nllb_tag(iso: str, script: str) -> str:
    iso3 = to_iso3(iso)
    if iso3 is None:
        raise ValueError(f"Cannot resolve iso code '{iso}' to a 3-letter code.")
    return NLLB_TAG_FROM_ISO.get(iso3, f"{iso3}_{script}")


def lookup_language(iso: str, assets_dir: Path) -> Tuple[str, str]:
    """Return (language name, country) for an iso code from languageFamilies.json."""
    iso3 = to_iso3(iso)
    if iso3 is None:
        raise ValueError(f"Cannot resolve iso code '{iso}' to a 3-letter code.")
    with open(assets_dir / "languageFamilies.json", "r", encoding="utf-8") as file:
        entries = json.load(file)
    for entry in entries:
        if entry.get("isoCode") == iso3:
            return entry["language"], entry["langCountry"]
    raise ValueError(f"Iso code '{iso3}' not found in languageFamilies.json; cannot determine language and country.")


def folder_name(name: str) -> str:
    name = name.replace(",", "").replace("-", " ")
    return "_".join(word.capitalize() for word in name.split())


def load_verse_counts(request_dir: Path, experiments_dir: Path) -> pd.DataFrame:
    """Load verse_counts.csv from the request folder, extended with any rows only in the global file."""
    frames = []
    for path in [request_dir / "verse_counts.csv", experiments_dir / "verse_counts" / "verse_counts.csv"]:
        if path.is_file():
            frames.append(pd.read_csv(path, index_col="file"))
    if not frames:
        raise FileNotFoundError(
            f"No verse_counts.csv found in {request_dir} or {experiments_dir / 'verse_counts'};"
            " required for --books complete."
        )
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="first")]


def resolve_corpus_books(books_arg: str, stems: Sequence[str], verse_counts: Optional[pd.DataFrame]) -> str:
    if books_arg.lower() != "complete":
        return books_arg
    if verse_counts is None:
        raise ValueError("verse counts are required for --books complete")
    if "complete" not in verse_counts.index:
        raise ValueError("No 'complete' row found in verse_counts.csv; cannot apply the completeness rule.")
    for stem in stems:
        if stem not in verse_counts.index:
            raise ValueError(f"No verse counts found for '{stem}'; cannot apply the completeness rule.")

    books = []
    for book in OT_CANON + NT_CANON:
        if book not in verse_counts.columns:
            continue
        complete_count = verse_counts.at["complete", book]
        if pd.isna(complete_count) or complete_count <= 0:
            continue
        threshold = BOOK_COMPLETENESS_THRESHOLD * complete_count
        counts = [verse_counts.at[stem, book] for stem in stems]
        if all(not pd.isna(count) and count >= threshold for count in counts):
            books.append(book)

    for canon, token in [(OT_CANON, "OT"), (NT_CANON, "NT")]:
        if all(book in books for book in canon):
            books = [book for book in books if book not in canon] + [token]
    return ";".join(books)


def build_config(sources: List[Candidate], main: MainProject, corpus_books: str) -> dict:
    lang_codes: Dict[str, str] = {}
    for source in sources:
        lang_codes.setdefault(source.iso, nllb_tag(source.iso, source.script))
    lang_codes.setdefault(main.iso, nllb_tag(main.iso, main.script or ""))
    src_stems = [source.stem for source in sources]
    return {
        "data": {
            "corpus_pairs": [
                {
                    "corpus_books": corpus_books,
                    "mapping": "mixed_src",
                    "src": src_stems[0] if len(src_stems) == 1 else src_stems,
                    "trg": main.stem,
                    "type": "train,test",
                }
            ],
            "lang_codes": lang_codes,
            "seed": SEED,
        },
        "model": MODEL,
    }


def build_translate_config(sources: List[Candidate], translate_books: str) -> dict:
    return {
        "translate": [
            {"books": translate_books, "src_project": source.name, "checkpoint": CHECKPOINT} for source in sources
        ],
        "postprocess": [{"paragraph_behavior": "place"}],
    }


def find_existing(lang_dir: Path, prefix: str, config: dict) -> Tuple[Optional[Path], int]:
    """Return (folder with an identical config or None, next free index) for prefix_<n> folders."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_index = 0
    for folder in lang_dir.iterdir() if lang_dir.is_dir() else []:
        m = pattern.match(folder.name)
        if m is None or not folder.is_dir():
            continue
        max_index = max(max_index, int(m.group(1)))
        config_path = folder / "config.yml"
        if config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    if yaml.safe_load(file) == config:
                        return folder, max_index
            except yaml.YAMLError:
                LOGGER.warning(f"Could not parse {config_path}; ignoring it for the identical-config check.")
    return None, max_index + 1


def select_mixed_pairs(pairs: List[List[Candidate]], dry_run: bool) -> List[List[Candidate]]:
    if len(pairs) <= MAX_UNPROMPTED_MIXED:
        return pairs
    print(f"\n{len(pairs)} mixed-source experiments pass the thresholds:")
    for i, pair in enumerate(pairs, start=1):
        names = " + ".join(f"{source.name} ({source.alignment:.4f})" for source in pair)
        print(f"  {i}. {names}")
    if dry_run:
        print("Dry run: all listed pairs are included in the report below.")
        return pairs
    reply = input("Enter the numbers to create (e.g. 1,3), 'all' or 'none': ").strip().lower()
    if reply in ("", "none"):
        return []
    if reply == "all":
        return pairs
    chosen = []
    for token in re.split(r"[,\s]+", reply):
        if token.isdigit() and 1 <= int(token) <= len(pairs):
            chosen.append(pairs[int(token) - 1])
        else:
            LOGGER.warning(f"Ignoring invalid selection '{token}'.")
    return chosen


def write_yaml(path: Path, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(content, file, sort_keys=False, default_flow_style=False, allow_unicode=True)


def submit_experiments(experiments: List[Experiment], experiments_dir: Path, submit: Optional[bool]) -> None:
    """Print the run command for each experiment and optionally execute them.

    submit: True runs without asking, None asks first, False only prints the commands.
    """
    names = [experiment.folder.relative_to(experiments_dir).as_posix() for experiment in experiments]
    print("\nTo run the experiments:")
    for name in names:
        print(f"  poetry run python {' '.join(EXPERIMENT_ARGS)} {name}")
    if submit is None:
        try:
            reply = input(f"\nRun {len(names)} experiment(s) now? [y/N]: ").strip().lower()
        except EOFError:
            reply = ""
        submit = reply in ("y", "yes")
    if not submit:
        return

    failures = []
    for name in names:
        print(f"\nRunning experiment {name}")
        result = subprocess.run([sys.executable] + EXPERIMENT_ARGS + [name])
        if result.returncode != 0:
            failures.append(name)
            print(f"Experiment {name} exited with code {result.returncode}.")
    if failures:
        print(f"\n{len(failures)} of {len(names)} experiment(s) failed: {', '.join(failures)}")
    else:
        print(f"\nAll {len(names)} experiment(s) completed.")


def resolve_request_dir(request: str, experiments_dir: Path) -> Path:
    requests_dir = experiments_dir / "_OnboardingRequests"
    for name in [request, f"{request}_Request"]:
        candidate = requests_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No request folder '{request}' or '{request}_Request' found in {requests_dir}.")


def run(
    request_dir: Path,
    experiments_dir: Path,
    assets_dir: Path,
    books: str,
    translate_books: str,
    min_parallel: int,
    min_alignment: float,
    dry_run: bool = False,
    submit: Optional[bool] = False,
) -> List[Experiment]:
    log_path = request_dir / "onboarding.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"No onboarding.log found in {request_dir}.")
    main, candidates = parse_log(log_path)

    language, country = lookup_language(main.iso, assets_dir)
    lang_dir = experiments_dir / folder_name(country) / folder_name(language)
    print(f"Main project: {main.name} ({main.stem}), language: {language} [{main.iso}], country: {country}")
    print(f"Experiment location: {lang_dir}")

    candidates.sort(key=lambda c: c.alignment, reverse=True)
    passing = [c for c in candidates if c.parallel >= min_parallel and c.alignment >= min_alignment]
    print(f"\n{'Reference':<24} {'iso':<5} {'count':>7} {'parallel':>9} {'alignment':>10}  result")
    for c in candidates:
        result = "pass" if c in passing else "fail"
        print(f"{c.name:<24} {c.iso:<5} {c.count:>7} {c.parallel:>9} {c.alignment:>10.4f}  {result}")
    if not passing:
        print(f"\nNo references passed the thresholds (parallel >= {min_parallel}, alignment >= {min_alignment}).")
        return []

    verse_counts = None
    if books.lower() == "complete":
        verse_counts = load_verse_counts(request_dir, experiments_dir)

    mixed = select_mixed_pairs([list(pair) for pair in itertools.combinations(passing, 2)], dry_run)

    experiments: List[Experiment] = []
    existing_experiments: List[Experiment] = []
    print()
    for sources in [[c] for c in passing] + mixed:
        label = " + ".join(source.name for source in sources)
        try:
            corpus_books = resolve_corpus_books(books, [s.stem for s in sources] + [main.stem], verse_counts)
        except ValueError as e:
            print(f"Skipped {label}: {e}")
            continue
        if not corpus_books:
            print(f"Skipped {label}: no book meets the {BOOK_COMPLETENESS_THRESHOLD:.0%} completeness rule.")
            continue
        config = build_config(sources, main, corpus_books)
        prefix = "_".join([source.name for source in sources] + [main.iso])
        existing, index = find_existing(lang_dir, prefix, config)
        if existing is not None:
            print(f"Skipped {label}: {existing} already contains an identical config.yml.")
            existing_experiments.append(
                Experiment(
                    sources=sources,
                    folder=existing,
                    config=config,
                    translate_config=build_translate_config(sources, translate_books),
                )
            )
            continue
        folder = lang_dir / f"{prefix}_{index}"
        experiment = Experiment(
            sources=sources,
            folder=folder,
            config=config,
            translate_config=build_translate_config(sources, translate_books),
        )
        experiments.append(experiment)
        if dry_run:
            print(f"Would create {folder} (corpus_books: {corpus_books})")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            write_yaml(folder / "config.yml", experiment.config)
            write_yaml(folder / "translate_config.yml", experiment.translate_config)
            print(f"Created {folder} (corpus_books: {corpus_books})")
    if experiments or existing_experiments:
        # Existing folders with an identical config are offered too: their creation was
        # skipped, but the experiments themselves may not have been run yet.
        # In a dry run nothing new exists to execute, so only print the commands.
        submit_experiments(experiments + existing_experiments, experiments_dir, submit=False if dry_run else submit)
    return experiments


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Create experiment folders from an onboarding request.")
    parser.add_argument("request", help="Request folder name in MT/experiments/_OnboardingRequests")
    parser.add_argument("--min-parallel", type=int, default=2000, help="Minimum parallel verse count (default 2000)")
    parser.add_argument("--min-alignment", type=float, default=0.2, help="Minimum alignment score (default 0.2)")
    parser.add_argument(
        "--books",
        default="complete",
        help="Semicolon-separated corpus_books list, or 'complete' (default) to derive it from verse_counts.csv",
    )
    parser.add_argument(
        "--translate-books",
        required=True,
        help="Book or semicolon-separated list of books for translate_config.yml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without creating folders or files")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run each created experiment (silnlp.nmt.experiment) without asking first",
    )
    args = parser.parse_args()

    environment = SilNlpEnv.create_standard_environment()
    experiments_dir = Path(environment.mt_experiments_dir)
    run(
        request_dir=resolve_request_dir(args.request, experiments_dir),
        experiments_dir=experiments_dir,
        assets_dir=Path(environment.assets_dir),
        books=args.books,
        translate_books=args.translate_books,
        min_parallel=args.min_parallel,
        min_alignment=args.min_alignment,
        dry_run=args.dry_run,
        submit=True if args.run else None,
    )


if __name__ == "__main__":
    main()
