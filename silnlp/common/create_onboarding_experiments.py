"""Create NMT experiment folders from a production onboarding request or analyze run.

Parses the corpus-stats.csv written by silnlp.common.analyze — found either in a
MT/experiments/_OnboardingRequests/<request> folder (top level or alignments/) or in
any experiment folder (e.g. PNG/Taupota/Align) — falling back to scraping the
onboarding.log for older request folders without one. It then selects the reference
projects whose alignment stats pass the thresholds, offers the top experiments for
selection, and creates <Country>/<Language>/<experiment> folders containing config.yml
and translate_config.yml. See create_onboarding_experiments_plan.md for details.
"""

import argparse
import itertools
import json
import logging
import re
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from machine.scripture import ALL_BOOK_IDS, book_id_to_number, book_number_to_id, get_chapters, is_nt, is_ot

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
TOP_EXPERIMENTS = 20

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


def stem_matches(stem: str, target: str) -> bool:
    """Case-insensitive match of a target against a stem, its iso prefix, or its project name."""
    return target.lower() in (stem.lower(), stem_to_iso(stem).lower(), stem_to_project(stem).lower())


def parse_corpus_stats(stats_path: Path, target: Optional[str] = None) -> Tuple[MainProject, List[Candidate]]:
    """Parse a corpus-stats.csv written by silnlp.common.analyze.

    The main project is the stem appearing in every row on one side: trg for alignment
    runs (e.g. <Country>/<Language>/Align), src for onboarding analyze runs. `target`
    (an iso code, project name or stem) overrides the detection; it may match either
    side per row, and rows not involving it are ignored.
    """
    df = pd.read_csv(stats_path)
    if df.empty:
        raise ValueError(f"{stats_path} contains no rows.")

    oriented = []  # (main stem, main script, ref stem, ref script, row)
    if target is not None:
        skipped = 0
        for _, row in df.iterrows():
            if stem_matches(row["src_project"], target):
                oriented.append((row["src_project"], row["src_script"], row["trg_project"], row["trg_script"], row))
            elif stem_matches(row["trg_project"], target):
                oriented.append((row["trg_project"], row["trg_script"], row["src_project"], row["src_script"], row))
            else:
                skipped += 1
        if not oriented:
            raise ValueError(f"'{target}' does not match any project in {stats_path}.")
        main_stems = sorted({entry[0] for entry in oriented})
        if len(main_stems) > 1:
            raise ValueError(
                f"'{target}' matches more than one project in {stats_path} ({', '.join(main_stems)});"
                " use the project name to disambiguate."
            )
        if skipped:
            LOGGER.warning(f"Ignoring {skipped} row(s) in {stats_path.name} that do not involve '{target}'.")
    elif df["trg_project"].nunique() == 1 and df["src_project"].nunique() > 1:
        for _, row in df.iterrows():
            oriented.append((row["trg_project"], row["trg_script"], row["src_project"], row["src_script"], row))
    elif df["src_project"].nunique() == 1 and df["trg_project"].nunique() > 1:
        for _, row in df.iterrows():
            oriented.append((row["src_project"], row["src_script"], row["trg_project"], row["trg_script"], row))
    else:
        raise ValueError(f"Cannot determine the main project in {stats_path}; specify it with --target.")

    candidates: Dict[str, Candidate] = {}
    for _, _, ref_stem, ref_script, row in oriented:
        candidates[stem_to_project(ref_stem)] = Candidate(
            name=stem_to_project(ref_stem),
            stem=ref_stem,
            iso=stem_to_iso(ref_stem),
            count=int(row["count"]),
            parallel=int(row["parallel"]),
            alignment=float(row["align_score"]),
            script=str(ref_script).strip(),
        )

    main_stem, main_script = oriented[0][0], oriented[0][1]
    main = MainProject(
        name=stem_to_project(main_stem),
        stem=main_stem,
        iso=stem_to_iso(main_stem),
        verses=None,
        script=str(main_script).strip(),
    )
    return main, list(candidates.values())


def parse_log_main_name(log_path: Path) -> Optional[str]:
    """Return the main project name from onboarding.log, if the log names one."""
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = MAIN_PROJECT_RE.search(line)
        if m is not None:
            return m.group(1)
    return None


def stem_to_iso(stem: str) -> str:
    return stem.split("-", 1)[0]


def stem_to_project(stem: str) -> str:
    return stem.split("-", 1)[1] if "-" in stem else stem


def parse_book_list(value: str) -> str:
    """Normalise and validate a book list from the command line.

    Any selection get_chapters accepts is kept verbatim (books, OT/NT, ranges like GEN-DEU,
    chapter selections like "MAT 1-4", subtractions like "NT;-REV"). A plain book list may
    also be separated by commas or spaces instead of semicolons. Raises ValueError with
    get_chapters' message when the value is not a valid book list.
    """
    verbatim = value.strip().upper()
    candidates = [verbatim, ";".join(token for token in re.split(r"[;,\s]+", verbatim) if token)]
    error: Optional[ValueError] = None
    for candidate in candidates:
        try:
            if get_chapters(candidate):
                return candidate
        except ValueError as e:
            error = error or e
    raise ValueError(f"'{value}' is not a valid book list{f': {error}' if error else '.'}")


def compact_canons(books: List[str]) -> str:
    """Replace a full testament's books with its OT/NT token."""
    for canon, token in [(OT_CANON, "OT"), (NT_CANON, "NT")]:
        if all(book in books for book in canon):
            books = [book for book in books if book not in canon] + [token]
    return ";".join(books)


def to_iso3(iso: str) -> Optional[str]:
    if len(iso) == 3:
        return iso
    return ALT_ISO.get_alternative(iso)


def nllb_tag(iso: str, script: str) -> str:
    iso3 = to_iso3(iso)
    if iso3 is None:
        raise ValueError(f"Cannot resolve iso code '{iso}' to a 3-letter code.")
    return NLLB_TAG_FROM_ISO.get(iso3, f"{iso3}_{script}")


def load_language_entries(assets_dir: Path) -> List[dict]:
    with open(assets_dir / "languageFamilies.json", "r", encoding="utf-8") as file:
        return json.load(file)


def lookup_language(iso: str, entries: List[dict]) -> Tuple[str, str]:
    """Return (language name, country) for an iso code from the languageFamilies.json entries."""
    iso3 = to_iso3(iso)
    if iso3 is None:
        raise ValueError(f"Cannot resolve iso code '{iso}' to a 3-letter code.")
    for entry in entries:
        if entry.get("isoCode") == iso3:
            return entry["language"], entry["langCountry"]
    raise ValueError(f"Iso code '{iso3}' not found in languageFamilies.json; cannot determine language and country.")


def synthesize_trg_iso(iso3: str, real_isos: AbstractSet[str]) -> str:
    """Derive a code that is neither a real iso code nor in NLLB by mutating the last two letters."""
    for last_two in itertools.product(string.ascii_lowercase, repeat=2):
        candidate = iso3[0] + "".join(last_two)
        if candidate != iso3 and candidate not in real_isos and candidate not in NLLB_TAG_FROM_ISO:
            return candidate
    raise ValueError(f"Could not synthesize a target iso code from '{iso3}'.")


def find_prior_rename(scripture_dir: Optional[Path], main: MainProject, real_isos: AbstractSet[str]) -> Optional[str]:
    """Return the synthetic iso of an extract file renamed by a previous run, if one exists.

    The renamed file on disk is the durable record of the code chosen earlier, so re-runs
    reuse it instead of deriving a possibly different code from the current iso tables.
    """
    if scripture_dir is None or not scripture_dir.is_dir():
        return None
    for path in sorted(scripture_dir.glob(f"*-{stem_to_project(main.stem)}.txt")):
        iso = stem_to_iso(path.stem)
        if iso != main.iso and iso not in real_isos and iso not in NLLB_TAG_FROM_ISO:
            return iso
    return None


def execute_rename(scripture_dir: Path, terms_dir: Optional[Path], old_stem: str, new_stem: str) -> None:
    """Rename the target extract file (and its terms renderings files) to the synthetic stem."""
    old_path = scripture_dir / f"{old_stem}.txt"
    old_path.rename(scripture_dir / f"{new_stem}.txt")
    print(f"Renamed {old_path.name} to {new_stem}.txt in {scripture_dir}")
    if terms_dir is not None and terms_dir.is_dir():
        for path in sorted(terms_dir.glob(f"{old_stem}-*-renderings.txt")):
            target = f"{new_stem}{path.name[len(old_stem):]}"
            path.rename(terms_dir / target)
            print(f"Renamed {path.name} to {target} in {terms_dir}")


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
            " required for --training-books complete."
        )
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="first")]


def resolve_corpus_books(
    books_arg: str,
    stems: Sequence[str],
    verse_counts: Optional[pd.DataFrame],
    exclude: AbstractSet[int] = frozenset(),
) -> Tuple[str, List[str]]:
    """Resolve the corpus_books list, excluding the books in `exclude` (canon book numbers).

    An explicit books_arg is kept verbatim (it may use any get_chapters syntax) with
    subtraction selections appended for the excluded books it covers.
    Returns (corpus_books string, ids of the books that were excluded from it).
    """
    if books_arg.lower() != "complete":
        selection = get_chapters(books_arg)  # book number -> chapter list ([] = whole book)
        overlap = sorted(number for number in selection if number in exclude)
        removed = [book_number_to_id(number) for number in overlap]
        if not removed:
            return books_arg, []
        if len(overlap) == len(selection):
            return "", removed
        subtractions = [
            f"-{book_number_to_id(number)}{','.join(str(chapter) for chapter in selection[number])}"
            for number in overlap
        ]
        corpus_books = ";".join([books_arg] + subtractions)
        get_chapters(corpus_books)  # never emit a selection the downstream parser rejects
        return corpus_books, removed

    if verse_counts is None:
        raise ValueError("verse counts are required for --training-books complete")
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

    removed = [book for book in books if book_id_to_number(book) in exclude]
    return compact_canons([book for book in books if book_id_to_number(book) not in exclude]), removed


def build_config(
    sources: List[Candidate], main: MainProject, corpus_books: str, test_variant: Optional[str] = None
) -> dict:
    lang_codes: Dict[str, str] = {}
    for source in sources:
        lang_codes.setdefault(source.iso, nllb_tag(source.iso, source.script))
    lang_codes.setdefault(main.iso, nllb_tag(main.iso, main.script or ""))
    src_stems = [source.stem for source in sources]
    corpus_pair: Dict[str, object] = {
        "corpus_books": corpus_books,
        "mapping": "mixed_src",
        "src": src_stems[0] if len(src_stems) == 1 else src_stems,
        "trg": main.stem,
        "type": "train" if test_variant == "notest" else "train,test",
    }
    if test_variant == "test100":
        corpus_pair["test_size"] = 100
    return {
        "data": {
            "corpus_pairs": [corpus_pair],
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


def select_experiments(
    singles: List[List[Candidate]], mixed: List[List[Candidate]], dry_run: bool
) -> List[List[Candidate]]:
    """Show the top possible experiments (singles first) and ask which to create.

    Under dry_run the list is only displayed and every displayed experiment is returned.
    """
    total = len(singles) + len(mixed)
    displayed = (singles + mixed)[:TOP_EXPERIMENTS]
    shown = f"top {len(displayed)} of {total}" if total > len(displayed) else f"{total}"
    print(f"\nPossible experiments ({shown}, single sources first):")
    for i, sources in enumerate(displayed, start=1):
        names = " + ".join(f"{source.name} ({source.alignment:.4f})" for source in sources)
        print(f"  {i:>2}. {names}")
    if dry_run:
        print("Dry run: all listed experiments are included in the report below.")
        return displayed
    try:
        reply = input("Enter the numbers to create (e.g. 1,3), 'all' or 'none': ").strip().lower()
    except EOFError:
        reply = ""
    if reply in ("", "none"):
        return []
    if reply == "all":
        return displayed
    chosen = []
    for token in re.split(r"[,\s]+", reply):
        if token.isdigit() and 1 <= int(token) <= len(displayed):
            chosen.append(displayed[int(token) - 1])
        else:
            LOGGER.warning(f"Ignoring invalid selection '{token}'.")
    return chosen


def write_yaml(path: Path, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(content, file, sort_keys=False, default_flow_style=False, allow_unicode=True)


def submit_experiments(
    experiments: List[Experiment], experiments_dir: Path, submit: Optional[bool], no_test: bool = False
) -> None:
    """Print the run command for each experiment and optionally execute them.

    submit: True runs without asking, None asks first, False only prints the commands.
    no_test: drop the --test stage (the experiments have no test set).
    """
    experiment_args = [arg for arg in EXPERIMENT_ARGS if not (no_test and arg == "--test")]
    names = [experiment.folder.relative_to(experiments_dir).as_posix() for experiment in experiments]
    print("\nTo run the experiments:")
    for name in names:
        print(f"  poetry run python {' '.join(experiment_args)} {name}")
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
        result = subprocess.run([sys.executable] + experiment_args + [name])
        if result.returncode != 0:
            failures.append(name)
            print(f"Experiment {name} exited with code {result.returncode}.")
    if failures:
        print(f"\n{len(failures)} of {len(names)} experiment(s) failed: {', '.join(failures)}")
    else:
        print(f"\nAll {len(names)} experiment(s) completed.")


def resolve_request_dir(request: str, experiments_dir: Path) -> Path:
    requests_dir = experiments_dir / "_OnboardingRequests"
    for candidate in [requests_dir / request, requests_dir / f"{request}_Request", experiments_dir / request]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No request folder '{request}' or '{request}_Request' found in {requests_dir},"
        f" and '{request}' is not a folder in {experiments_dir}."
    )


def run(
    request_dir: Path,
    experiments_dir: Path,
    assets_dir: Path,
    training_books: str,
    translate_books: str,
    min_parallel: int,
    min_alignment: float,
    scripture_dir: Optional[Path] = None,
    terms_dir: Optional[Path] = None,
    test_variant: Optional[str] = None,
    target: Optional[str] = None,
    dry_run: bool = False,
    submit: Optional[bool] = False,
) -> List[Experiment]:
    if test_variant not in (None, "notest", "test100"):
        raise ValueError(f"Unknown test_variant '{test_variant}'; expected None, 'notest' or 'test100'.")
    log_path = request_dir / "onboarding.log"
    stats_paths = [request_dir / "corpus-stats.csv", request_dir / "alignments" / "corpus-stats.csv"]
    stats_path = next((path for path in stats_paths if path.is_file()), None)
    lang_dir_override: Optional[Path] = None
    if stats_path is not None:
        # The CSV is the stable machine-readable artifact, so it is preferred over scraping
        # the log; an explicit --target wins over the log's main-project line as the hint
        # for which side of the CSV is the target.
        main_hint = target or (parse_log_main_name(log_path) if log_path.is_file() else None)
        main, candidates = parse_corpus_stats(stats_path, target=main_hint)
        # A stats folder inside the experiments tree (e.g. PNG/Taupota/Align) creates its
        # experiments next to itself, keeping whatever country/language naming already
        # exists — but never inside _OnboardingRequests, where the derived location applies.
        request_parents = request_dir.resolve().parents
        if experiments_dir.resolve() in request_parents and (
            (experiments_dir / "_OnboardingRequests").resolve() not in request_parents
        ):
            lang_dir_override = request_dir.parent
    elif log_path.is_file():
        main, candidates = parse_log(log_path)
        if target is not None and not stem_matches(main.stem, target):
            raise ValueError(f"--target '{target}' does not match the main project '{main.stem}' in {log_path}.")
    else:
        raise FileNotFoundError(f"No corpus-stats.csv or onboarding.log found in {request_dir}.")

    language_entries = load_language_entries(assets_dir)
    language, country = lookup_language(main.iso, language_entries)
    lang_dir = lang_dir_override or experiments_dir / folder_name(country) / folder_name(language)
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

    # The src and trg isos of a corpus pair must differ. When a passing reference shares the
    # main project's iso, switch the main project to a synthetic code (not a real iso, not in
    # NLLB) and rename its extract file to match. The rename is deferred until an experiment
    # is actually created; a run that creates nothing leaves MT/scripture untouched. A rename
    # done by a previous run (recorded by the file on disk) is reused even when the current
    # thresholds no longer surface the clash, so the configs always match the file that exists.
    counts_stem = main.stem  # verse_counts.csv is keyed by the original stem
    real_isos = {entry["isoCode"] for entry in language_entries}
    prior_iso = find_prior_rename(scripture_dir, main, real_isos)
    clashing = [c for c in passing if to_iso3(c.iso) == to_iso3(main.iso)]
    pending_rename: Optional[Tuple[str, str]] = None  # (old stem, new stem), executed on first creation
    if clashing or prior_iso is not None:
        synthetic = prior_iso or synthesize_trg_iso(to_iso3(main.iso) or main.iso, real_isos)
        if clashing:
            names = ", ".join(c.name for c in clashing)
            print(
                f"\n{names} share{'s' if len(clashing) == 1 else ''} iso code '{main.iso}' with the target"
                f" project; using synthetic target code '{synthetic}' instead."
            )
        else:
            print(f"\nUsing synthetic target code '{synthetic}' from the previously renamed extract file.")
        new_stem = f"{synthetic}-{stem_to_project(main.stem)}"
        if scripture_dir is None:
            if not dry_run:
                raise ValueError("No scripture directory available to rename the target extract file in.")
            print(f"Would rename {main.stem}.txt to {new_stem}.txt in the MT scripture folder.")
        else:
            old_exists = (scripture_dir / f"{main.stem}.txt").is_file()
            new_exists = (scripture_dir / f"{new_stem}.txt").is_file()
            if old_exists and new_exists:
                raise RuntimeError(
                    f"Both {main.stem}.txt and {new_stem}.txt exist in {scripture_dir} (the original was"
                    " probably re-extracted after an earlier rename); remove the stale one before re-running."
                )
            if old_exists:
                pending_rename = (main.stem, new_stem)
                if dry_run:
                    print(
                        f"Would rename {main.stem}.txt to {new_stem}.txt in {scripture_dir}"
                        " (and matching terms renderings files)."
                    )
            elif not new_exists and not dry_run:
                raise FileNotFoundError(f"Neither {main.stem}.txt nor {new_stem}.txt found in {scripture_dir}.")
        main.iso, main.stem = synthetic, new_stem

    verse_counts = None
    if training_books.lower() == "complete":
        verse_counts = load_verse_counts(request_dir, experiments_dir)
    translate_set = frozenset(get_chapters(translate_books))

    chosen = select_experiments(
        [[c] for c in passing], [list(pair) for pair in itertools.combinations(passing, 2)], dry_run
    )

    experiments: List[Experiment] = []
    existing_experiments: List[Experiment] = []
    warned_removals: set = set()
    print()
    for sources in chosen:
        label = " + ".join(source.name for source in sources)
        try:
            corpus_books, removed = resolve_corpus_books(
                training_books, [s.stem for s in sources] + [counts_stem], verse_counts, exclude=translate_set
            )
        except ValueError as e:
            print(f"Skipped {label}: {e}")
            continue
        if removed and tuple(removed) not in warned_removals:
            warned_removals.add(tuple(removed))
            print(f"Warning: excluded the books being translated from corpus_books: {';'.join(removed)}")
        if not corpus_books:
            print(
                f"Skipped {label}: no training books remain after the {BOOK_COMPLETENESS_THRESHOLD:.0%}"
                " completeness rule and the translate-book exclusion."
            )
            continue
        config = build_config(sources, main, corpus_books, test_variant)
        prefix = "_".join([source.name for source in sources] + [main.iso] + ([test_variant] if test_variant else []))
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
            if pending_rename is not None:
                assert scripture_dir is not None
                execute_rename(scripture_dir, terms_dir, *pending_rename)
                pending_rename = None
            folder.mkdir(parents=True, exist_ok=True)
            write_yaml(folder / "config.yml", experiment.config)
            write_yaml(folder / "translate_config.yml", experiment.translate_config)
            print(f"Created {folder} (corpus_books: {corpus_books})")
    if experiments or existing_experiments:
        # Existing folders with an identical config are offered too: their creation was
        # skipped, but the experiments themselves may not have been run yet.
        # In a dry run nothing new exists to execute, so only print the commands.
        submit_experiments(
            experiments + existing_experiments,
            experiments_dir,
            submit=False if dry_run else submit,
            no_test=test_variant == "notest",
        )
    return experiments


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Create experiment folders from an onboarding request.")
    parser.add_argument(
        "request",
        help="Request folder name in MT/experiments/_OnboardingRequests, or a folder relative to"
        " MT/experiments containing a corpus-stats.csv from an analyze run (e.g. PNG/Taupota/Align)",
    )
    parser.add_argument("--min-parallel", type=int, default=2000, help="Minimum parallel verse count (default 2000)")
    parser.add_argument("--min-alignment", type=float, default=0.2, help="Minimum alignment score (default 0.2)")
    parser.add_argument(
        "--target",
        help="Iso code or project name of the target language, overriding its detection from"
        " corpus-stats.csv (it may appear in either column) or onboarding.log",
    )
    book_list_syntax = (
        "separated by commas, semicolons or spaces, including silnlp selections like ranges and"
        ' subtractions (quote semicolons and spaces: "GEN;RUT", "GEN RUT", "NT;-REV", "GEN-DEU")'
    )
    parser.add_argument(
        "--training-books",
        default="complete",
        help=f"Corpus_books list {book_list_syntax}, or 'complete' (default) to derive it from verse_counts.csv",
    )
    parser.add_argument(
        "--translate-books",
        required=True,
        help=f"Book or list of books for translate_config.yml, {book_list_syntax}",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--no-test",
        action="store_true",
        help="Train-only experiments (type: train, no test set); folder names gain _notest",
    )
    test_group.add_argument(
        "--test100",
        action="store_true",
        help="Use a 100-verse test set (test_size: 100); folder names gain _test100",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without creating folders or files")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run each created experiment (silnlp.nmt.experiment) without asking first",
    )
    args = parser.parse_args()
    try:
        if args.training_books.strip().lower() != "complete":
            args.training_books = parse_book_list(args.training_books)
        args.translate_books = parse_book_list(args.translate_books)
    except ValueError as e:
        parser.error(str(e))

    environment = SilNlpEnv.create_standard_environment()
    experiments_dir = Path(environment.mt_experiments_dir)
    run(
        request_dir=resolve_request_dir(args.request, experiments_dir),
        experiments_dir=experiments_dir,
        assets_dir=Path(environment.assets_dir),
        training_books=args.training_books,
        translate_books=args.translate_books,
        min_parallel=args.min_parallel,
        min_alignment=args.min_alignment,
        scripture_dir=Path(environment.mt_scripture_dir),
        terms_dir=Path(environment.mt_terms_dir),
        test_variant="notest" if args.no_test else "test100" if args.test100 else None,
        target=args.target,
        dry_run=args.dry_run,
        submit=True if args.run else None,
    )


if __name__ == "__main__":
    main()
