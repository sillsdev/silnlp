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
import shutil
import string
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from machine.corpora import FileParatextProjectSettingsParser
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
        # Keep the first row that reports a real script; a no-parallel-data row logs 'None'/'nan'.
        main_script = main_script or clean_script(entry["src_script"])
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
    """Case-insensitive match of a target against a stem, its iso prefix, or its project name.

    Iso codes match across their 2- and 3-letter forms (e.g. 'fra' matches an 'fr-' stem).
    """
    target = target.lower()
    if target in (stem.lower(), stem_to_project(stem).lower()):
        return True
    stem_iso = stem_to_iso(stem).lower()
    if target == stem_iso:
        return True
    target_iso3 = to_iso3(target) if len(target) in (2, 3) else None
    return target_iso3 is not None and target_iso3 == to_iso3(stem_iso)


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
            src_match = stem_matches(row["src_project"], target)
            trg_match = stem_matches(row["trg_project"], target)
            if src_match and trg_match and row["src_project"] != row["trg_project"]:
                raise ValueError(
                    f"'{target}' matches both sides of a row in {stats_path}"
                    f" ({row['src_project']} and {row['trg_project']}); use the project name to disambiguate."
                )
            if src_match:
                oriented.append((row["src_project"], row["src_script"], row["trg_project"], row["trg_script"], row))
            elif trg_match:
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
    incomplete = 0
    for _, _, ref_stem, ref_script, row in oriented:
        if any(pd.isna(row[column]) for column in ("count", "parallel", "align_score")):
            incomplete += 1
            continue
        candidates[stem_to_project(ref_stem)] = Candidate(
            name=stem_to_project(ref_stem),
            stem=ref_stem,
            iso=stem_to_iso(ref_stem),
            count=int(row["count"]),
            parallel=int(row["parallel"]),
            alignment=float(row["align_score"]),
            script=str(ref_script).strip(),
        )
    if incomplete:
        LOGGER.warning(f"Skipping {incomplete} row(s) in {stats_path.name} with missing statistics.")

    # Take the main project's script from the first row that actually reports one: a pair with
    # no parallel data has an empty ('None'/'nan') script, and such a row must not decide the
    # main script just because it sorts first (this is what produced '<iso>_nan' configs).
    main_stem = oriented[0][0]
    main_script = next((script for entry in oriented if (script := clean_script(entry[1])) is not None), None)
    main = MainProject(
        name=stem_to_project(main_stem),
        stem=main_stem,
        iso=stem_to_iso(main_stem),
        verses=None,
        script=main_script,
    )
    return main, list(candidates.values())


def parse_log_main_name(log_path: Path) -> Optional[str]:
    """Return the main project name from onboarding.log, if the log names one."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            m = MAIN_PROJECT_RE.search(line)
            if m is not None:
                return m.group(1)
    return None


def clean_script(value: object) -> Optional[str]:
    """A usable script abbreviation, or None for a missing/placeholder value.

    analyze writes 'None' (predict_script_code's empty-text result) and 'nan' for pairs with
    no parallel data; pandas also reads both as NaN. Treat all of these as "no script" so a
    stray empty row never becomes a literal '<iso>_nan' language tag.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "nan", "none"):
        return None
    return text


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


def find_prior_copy(scripture_dir: Optional[Path], main: MainProject, real_isos: AbstractSet[str]) -> Optional[str]:
    """Return the synthetic iso of an extract copy made by a previous run, if one exists.

    The copied file on disk is the durable record of the code chosen earlier, so re-runs
    reuse it instead of deriving a possibly different code from the current iso tables.
    """
    if scripture_dir is None or not scripture_dir.is_dir():
        return None
    for path in sorted(scripture_dir.glob(f"*-{stem_to_project(main.stem)}.txt")):
        iso = stem_to_iso(path.stem)
        if iso != main.iso and iso not in real_isos and iso not in NLLB_TAG_FROM_ISO:
            return iso
    return None


def execute_copy(scripture_dir: Path, terms_dir: Optional[Path], old_stem: str, new_stem: str) -> None:
    """Copy the target extract file (and its terms renderings files) to the synthetic stem.

    The original files are kept: they may be referenced by other experiments and tools.
    """
    old_path = scripture_dir / f"{old_stem}.txt"
    shutil.copyfile(old_path, scripture_dir / f"{new_stem}.txt")
    print(f"Copied {old_path.name} to {new_stem}.txt in {scripture_dir}")
    if terms_dir is not None and terms_dir.is_dir():
        for path in sorted(terms_dir.glob(f"{old_stem}-*-renderings.txt")):
            target = f"{new_stem}{path.name[len(old_stem):]}"
            shutil.copyfile(path, terms_dir / target)
            print(f"Copied {path.name} to {target} in {terms_dir}")


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


def load_vref_books(assets_dir: Path) -> List[str]:
    """The book id of each line of vref.txt (the verse layout of every extract file)."""
    with open(assets_dir / "vref.txt", "r", encoding="utf-8") as file:
        return [line.split(" ", 1)[0] for line in file if line.strip()]


def extract_book_counts(extract_path: Path, vref_books: Sequence[str]) -> Dict[str, int]:
    """Per-book verse counts of a vref-aligned extract file, counting the non-blank lines.

    collect_verse_counts counts any line that is not exactly a newline; treating a
    whitespace-only line as missing only differs from that at the margin, conservatively.
    """
    counts: Dict[str, int] = Counter()
    with open(extract_path, "r", encoding="utf-8") as file:
        for book, line in zip(vref_books, file):
            if line.strip():
                counts[book] += 1
    return dict(counts)


class BookCoverage:
    """Per-book verse counts for extract stems.

    Counts come from verse_counts.csv rows, falling back to counting the vref-aligned
    extract file in MT/scripture, so a stem missing from the counts files never misreports
    a scripture's book coverage.
    """

    def __init__(self, verse_counts: Optional[pd.DataFrame], scripture_dir: Optional[Path], assets_dir: Path):
        self._verse_counts = verse_counts
        self._scripture_dir = scripture_dir
        self._assets_dir = assets_dir
        self._vref_books: Optional[List[str]] = None
        self._cache: Dict[str, Optional[Dict[str, int]]] = {}

    def counts(self, stem: str) -> Optional[Dict[str, int]]:
        """Verse counts per book id, or None when the stem has no counts row and no extract file."""
        if stem not in self._cache:
            self._cache[stem] = self._load(stem)
        return self._cache[stem]

    def complete(self) -> Dict[str, int]:
        """The full canon's verse count per book: the 'complete' row, or computed from vref.txt."""
        if self._verse_counts is not None and "complete" in self._verse_counts.index:
            return self._row_counts("complete")
        return dict(Counter(self._books()))

    def _load(self, stem: str) -> Optional[Dict[str, int]]:
        if self._verse_counts is not None and stem in self._verse_counts.index:
            return self._row_counts(stem)
        if self._scripture_dir is not None and (self._scripture_dir / f"{stem}.txt").is_file():
            return extract_book_counts(self._scripture_dir / f"{stem}.txt", self._books())
        return None

    def _row_counts(self, index: str) -> Dict[str, int]:
        assert self._verse_counts is not None
        row = self._verse_counts.loc[index]
        return {
            book: int(row[book])
            for book in self._verse_counts.columns
            if book in ALL_BOOK_IDS and not pd.isna(row[book])
        }

    def _books(self) -> List[str]:
        if self._vref_books is None:
            self._vref_books = load_vref_books(self._assets_dir)
        return self._vref_books


def missing_books(counts: Dict[str, int], required: Sequence[str], complete: Dict[str, int]) -> List[str]:
    """The required books that `counts` does not contain (>= 98% of the complete count)."""
    return [
        book
        for book in required
        if complete.get(book, 0) > 0 and counts.get(book, 0) < BOOK_COMPLETENESS_THRESHOLD * complete.get(book, 0)
    ]


def format_books(books: Sequence[str], limit: int = 10) -> str:
    if len(books) <= limit:
        return ";".join(books)
    return ";".join(books[:limit]) + f" (+{len(books) - limit} more)"


def drafting_qualified(counts: Dict[str, int], complete: Dict[str, int], book_set: AbstractSet[int]) -> bool:
    """Whether a training source looks like a published scripture that is safe to draft from.

    True when it holds at least 98% of a full Bible, or of the NT/OT when all the books
    to translate are in that testament (a published New/Old Testament).
    """

    def coverage(canon: Sequence[str]) -> float:
        complete_total = sum(complete.get(book, 0) for book in canon)
        if complete_total <= 0:
            return 0.0
        return sum(counts.get(book, 0) for book in canon) / complete_total

    if coverage(OT_CANON + NT_CANON) >= BOOK_COMPLETENESS_THRESHOLD:
        return True
    if all(is_nt(number) for number in book_set) and coverage(NT_CANON) >= BOOK_COMPLETENESS_THRESHOLD:
        return True
    if all(is_ot(number) for number in book_set) and coverage(OT_CANON) >= BOOK_COMPLETENESS_THRESHOLD:
        return True
    return False


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


def build_translate_config(projects: Sequence[str], translate_books: str) -> dict:
    return {
        "translate": [
            {
                "books": translate_books,
                "src_project": project,
                "checkpoint": CHECKPOINT,
            }
            for project in projects
        ],
        "postprocess": [{"paragraph_behavior": "place"}],
    }


@dataclass
class DraftingPlan:
    """The drafting sources and translate books of one experiment."""

    projects: List[str]  # drafting project names (before any Paratext replacement)
    books: str  # translate book selection for translate_config.yml
    book_set: AbstractSet[int]  # the same selection as canon book numbers
    auto: bool  # chosen automatically, so subject to the Paratext replacement prompt


def prompt_drafting_source(
    sources: Sequence[Candidate],
    books: str,
    book_set: AbstractSet[int],
    qualifies: Callable[[Candidate, AbstractSet[int]], bool],
    projects_dir: Optional[Path],
) -> Optional[Tuple[DraftingPlan, bool]]:
    """Ask for a drafting source when no training source qualifies as one.

    The user may name a project to draft from (checked, and accepted anyway when entered a
    second time), or pick a different selection of translate books, in which case the
    training sources are re-qualified and the user chooses whether the new selection applies
    to the whole run. Returns (plan, applies to all experiments) or None to skip the experiment.
    """
    last_project: Optional[str] = None
    while True:
        try:
            reply = input("Enter a project to draft from (or press Enter to choose different translate books): ")
        except EOFError:
            reply = ""
        reply = reply.strip()
        if reply:
            book_ids = [book_number_to_id(number) for number in sorted(book_set)]
            problem = check_translate_source(projects_dir, reply, book_ids) if projects_dir is not None else None
            if problem is not None and reply != last_project:
                print(
                    f"Warning: cannot translate {';'.join(book_ids)} from '{reply}': {problem}."
                    " Enter the same project again to use it anyway."
                )
                last_project = reply
                continue
            return DraftingPlan([reply], books, book_set, auto=False), False
        try:
            books_reply = input(
                "Enter a different selection of books to translate (or press Enter to skip this experiment): "
            )
        except EOFError:
            books_reply = ""
        books_reply = books_reply.strip()
        if not books_reply:
            return None
        try:
            new_books = parse_book_list(books_reply)
        except ValueError as e:
            print(str(e))
            continue
        new_set = frozenset(get_chapters(new_books))
        drafting = [source.name for source in sources if qualifies(source, new_set)]
        if not drafting:
            print("Still no training source qualifies as a drafting source with those books.")
            continue
        while True:
            try:
                scope = input("Apply the new translate books to all experiments or only this one? [all/one]: ")
            except EOFError:
                scope = "one"
            scope = scope.strip().lower()
            if scope in ("all", "a", "one", "o", ""):
                break
            print("Please answer 'all' or 'one'.")
        return DraftingPlan(drafting, new_books, new_set, auto=True), scope in ("all", "a")


def check_translate_source(projects_dir: Path, project: str, books: Sequence[str]) -> Optional[str]:
    """Return why `project` cannot supply `books` for translation, or None if it can.

    The books' file names come from the project's Settings.xml naming convention.
    """
    project_dir = projects_dir / project
    if not project_dir.is_dir():
        return f"there is no project folder '{project}' in {projects_dir}"
    try:
        settings = FileParatextProjectSettingsParser(project_dir).parse()
    except Exception as e:
        return f"the settings of project '{project}' could not be read ({e})"
    missing = [book for book in books if not (project_dir / settings.get_book_file_name(book)).is_file()]
    if missing:
        return f"project '{project}' does not contain {';'.join(missing)}"
    return None


def resolve_translate_sources(
    source_names: Sequence[str], books: Sequence[str], projects_dir: Optional[Path], dry_run: bool
) -> Dict[str, str]:
    """Check that each translation source project contains the books to be translated.

    Warns about a missing project or missing books and asks for a different project to
    translate from (checked too), so the user can decide how to proceed. Returns a mapping
    of source name -> project to use as src_project in translate_config.yml.
    """
    replacements: Dict[str, str] = {}
    if projects_dir is None:
        return replacements
    for name in source_names:
        current = name
        while True:
            problem = check_translate_source(projects_dir, current, books)
            if problem is None:
                break
            print(f"Warning: cannot translate {';'.join(books)} from '{current}': {problem}.")
            if dry_run:
                break
            try:
                reply = input(
                    f"Enter a different project to translate from (or press Enter to keep '{current}'): "
                ).strip()
            except EOFError:
                reply = ""
            if not reply or reply == current:
                break
            current = reply
        if current != name:
            print(f"Translating from '{current}' instead of '{name}'.")
            replacements[name] = current
    return replacements


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
    singles: List[List[Candidate]], mixed: List[List[Candidate]], dry_run: bool, top: int = TOP_EXPERIMENTS
) -> List[List[Candidate]]:
    """Show the top possible experiments (singles first) and ask which to create.

    Under dry_run the list is only displayed and every displayed experiment is returned.
    """
    total = len(singles) + len(mixed)
    displayed = (singles + mixed)[:top]
    shown = f"top {len(displayed)} of {total}" if total > len(displayed) else f"{total}"
    print(f"\nPossible experiments ({shown}, single sources first):")
    for i, sources in enumerate(displayed, start=1):
        names = " + ".join(f"{source.name} ({source.alignment:.4f})" for source in sources)
        print(f"  {i:>2}. {names}")
    if total > len(displayed):
        print(f"Use --top to list more than {top} experiments.")
    if dry_run:
        print("Dry run: all listed experiments are included in the report below.")
        return displayed
    try:
        reply = input("Enter the numbers to create (e.g. 1,3), 'all' or 'none': ").strip().lower()
    except EOFError:
        reply = ""
    if reply in ("", "none"):
        print("No experiments selected.")
        return []
    if reply == "all":
        return displayed
    chosen = []
    for token in re.split(r"[,\s]+", reply):
        if token.isdigit() and 1 <= int(token) <= len(displayed):
            selection = displayed[int(token) - 1]
            if selection not in chosen:
                chosen.append(selection)
        else:
            LOGGER.warning(f"Ignoring invalid selection '{token}'.")
    if not chosen:
        print("No experiments selected.")
    return chosen


def write_yaml(path: Path, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(content, file, sort_keys=False, default_flow_style=False, allow_unicode=True)


def update_translate_config(folder: Path, translate_config: dict, dry_run: bool) -> None:
    """Bring an existing experiment's translate_config.yml in line with the current drafting choice.

    An identical config.yml does not mean an identical translate_config.yml:
    --translate-scripture and the drafting prompts change only the latter.
    """
    path = folder / "translate_config.yml"
    on_disk: Optional[dict] = None
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as file:
                on_disk = yaml.safe_load(file)
        except yaml.YAMLError:
            LOGGER.warning(f"Could not parse {path}; it will be rewritten.")
    if on_disk == translate_config:
        return
    if dry_run:
        print(f"Would update {path} with the current drafting configuration.")
    else:
        write_yaml(path, translate_config)
        print(f"Updated {path} with the current drafting configuration.")


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
    projects_dir: Optional[Path] = None,
    test_variant: Optional[str] = None,
    target: Optional[str] = None,
    translate_scripture: Optional[Sequence[str]] = None,
    top: int = TOP_EXPERIMENTS,
    dry_run: bool = False,
    submit: Optional[bool] = False,
) -> List[Experiment]:
    if test_variant not in (None, "notest", "test100"):
        raise ValueError(f"Unknown test_variant '{test_variant}'; expected None, 'notest' or 'test100'.")
    log_path = request_dir / "onboarding.log"
    stats_paths = [request_dir / "corpus-stats.csv", request_dir / "alignments" / "corpus-stats.csv"]
    stats_path = next((path for path in stats_paths if path.is_file()), None)
    lang_dir_override: Optional[Path] = None
    flipped = False
    if stats_path is not None:
        # The CSV is the stable machine-readable artifact, so it is preferred over scraping
        # the log. An explicit --target is authoritative; the log's main-project line is only
        # a soft hint — if it does not fit the CSV, fall back to auto-detection and finally to
        # parsing the log itself, so a stale or partial CSV never breaks a working folder.
        log_hint = parse_log_main_name(log_path) if log_path.is_file() and target is None else None
        try:
            main, candidates = parse_corpus_stats(stats_path, target=target or log_hint)
        except ValueError as stats_error:
            if target is not None:
                raise
            try:
                if log_hint is None:
                    raise stats_error
                LOGGER.warning(f"{stats_error} Falling back to auto-detection.")
                main, candidates = parse_corpus_stats(stats_path)
            except ValueError as auto_error:
                if not log_path.is_file():
                    raise
                LOGGER.warning(f"{auto_error} Falling back to {log_path.name}.")
                main, candidates = parse_log(log_path)

        # Report when the chosen target differs from what the analyze run itself would give
        # (--target flipped the direction); the folder-derived location must not apply then.
        detected_stem: Optional[str] = main.stem
        if target is not None:
            natural_hint = parse_log_main_name(log_path) if log_path.is_file() else None
            try:
                detected_stem = parse_corpus_stats(stats_path, target=natural_hint)[0].stem
            except ValueError:
                detected_stem = None
        flipped = detected_stem is not None and detected_stem != main.stem
        if flipped:
            print(f"Note: --target overrides the analyze run's own target project ({detected_stem}).")

        # A stats folder inside the experiments tree (e.g. PNG/Taupota/Align) creates its
        # experiments next to itself, keeping whatever country/language naming already
        # exists — but not for request folders (which use the derived location), not directly
        # under MT/experiments itself, and not when --target flipped the target language.
        request_parents = request_dir.resolve().parents
        if (
            experiments_dir.resolve() in request_parents
            and request_dir.resolve().parent != experiments_dir.resolve()
            and (experiments_dir / "_OnboardingRequests").resolve() not in request_parents
            and not log_path.is_file()
            and not flipped
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

    # The primary (first) training source doubles as the drafting source, so it must not be a
    # back translation. Book coverage is the reliable signal: a primary source must contain
    # every book present in the target scripture plus every book to be translated. See
    # create_onboarding_experiments_brief.md for the rationale.
    verse_counts: Optional[pd.DataFrame] = None
    try:
        verse_counts = load_verse_counts(request_dir, experiments_dir)
    except FileNotFoundError:
        if training_books.lower() == "complete":
            raise
    except Exception as e:  # a malformed counts file must not break runs that never needed it
        if training_books.lower() == "complete":
            raise
        LOGGER.warning(f"Could not read verse counts ({e}); book coverage will use extract files only.")
    coverage = BookCoverage(verse_counts, scripture_dir, assets_dir)
    complete_counts = coverage.complete()
    translate_set = frozenset(get_chapters(translate_books))
    translate_book_ids = [book_number_to_id(number) for number in sorted(translate_set)]

    target_counts = coverage.counts(main.stem)
    if target_counts is None:
        LOGGER.warning(
            f"No verse counts or extract file found for the target '{main.stem}'; only the translate"
            " books can be required of a primary training source."
        )
    required_books = [
        book for book in OT_CANON + NT_CANON if (target_counts or {}).get(book, 0) > 0 or book in translate_book_ids
    ]
    required_books += [book for book in translate_book_ids if book not in required_books]

    target_total = sum(target_counts.values()) if target_counts is not None else main.verses
    secondary_min = max(1000.0, 0.25 * target_total) if target_total else 1000.0

    primary_pool: List[Candidate] = []
    for c in passing:
        counts = coverage.counts(c.stem)
        if counts is None:
            LOGGER.warning(
                f"No verse counts or extract file found for '{c.stem}'; cannot verify its book"
                " coverage — keeping it as a possible primary training source."
            )
            primary_pool.append(c)
            continue
        missing = missing_books(counts, required_books, complete_counts)
        if missing:
            print(f"Note: {c.name} cannot be the primary training source; it is missing {format_books(missing)}.")
        else:
            primary_pool.append(c)
    for c in passing:
        if c.parallel < secondary_min:
            print(
                f"Note: {c.name} cannot be the second source of a mixed pair; {c.parallel} parallel"
                f" verses is below {secondary_min:.0f} (max of 1000 and 25% of the target's verses)."
            )
    if not primary_pool:
        print(
            "\nNo passing reference contains all the books of the target scripture plus the"
            " translate books, so none can be the primary training source. Nothing to create."
        )
        return []

    # The src and trg isos of a corpus pair must differ. When a passing reference shares the
    # main project's iso, switch the main project to a synthetic code (not a real iso, not in
    # NLLB) and copy its extract file to the new stem, keeping the original. The copy is
    # deferred until an experiment is actually created; a run that creates nothing leaves
    # MT/scripture untouched. A copy made by a previous run (recorded by the file on disk) is
    # reused even when the current thresholds no longer surface the clash, so the configs
    # always match a file that exists.
    counts_stem = main.stem  # verse_counts.csv is keyed by the original stem
    real_isos = {entry["isoCode"] for entry in language_entries}
    prior_iso = find_prior_copy(scripture_dir, main, real_isos)
    # Only candidates that can still appear in an experiment can cause a clash: one that is
    # excluded from the primary pool and too small to be a second source is never written
    # to a config, so it must not trigger the synthetic code and the extract copy.
    usable = [
        c
        for c in passing
        if c in primary_pool or (c.parallel >= secondary_min and any(p is not c for p in primary_pool))
    ]
    clashing = [c for c in usable if to_iso3(c.iso) == to_iso3(main.iso)]
    pending_copy: Optional[Tuple[str, str]] = None  # (old stem, new stem), executed on first creation
    if clashing or prior_iso is not None:
        synthetic = prior_iso or synthesize_trg_iso(to_iso3(main.iso) or main.iso, real_isos)
        if clashing:
            names = ", ".join(c.name for c in clashing)
            print(
                f"\n{names} share{'s' if len(clashing) == 1 else ''} iso code '{main.iso}' with the target"
                f" project; using synthetic target code '{synthetic}' instead."
            )
        else:
            print(f"\nUsing synthetic target code '{synthetic}' from the previously copied extract file.")
        new_stem = f"{synthetic}-{stem_to_project(main.stem)}"
        if scripture_dir is None:
            if not dry_run:
                raise ValueError("No scripture directory available to copy the target extract file in.")
            print(f"Would copy {main.stem}.txt to {new_stem}.txt in the MT scripture folder.")
        else:
            old_path = scripture_dir / f"{main.stem}.txt"
            new_path = scripture_dir / f"{new_stem}.txt"
            if new_path.is_file():
                if old_path.is_file() and old_path.stat().st_mtime > new_path.stat().st_mtime:
                    LOGGER.warning(
                        f"{new_path.name} may be outdated: {old_path.name} is newer (probably re-extracted)."
                        f" Delete {new_path.name} and re-run to refresh the copy."
                    )
            elif old_path.is_file():
                pending_copy = (main.stem, new_stem)
                if dry_run:
                    print(
                        f"Would copy {main.stem}.txt to {new_stem}.txt in {scripture_dir}"
                        " (and matching terms renderings files)."
                    )
                else:
                    # The copy adds files to the shared MT/scripture store — always confirm first.
                    print(
                        f"{main.stem}.txt (and matching terms renderings files) will be copied to"
                        f" {new_stem}.txt in {scripture_dir}; the originals are kept."
                    )
                    if flipped:
                        print(
                            "Warning: the target was overridden with --target; make sure"
                            f" {main.stem} really is the intended target project."
                        )
                    elif to_iso3(main.iso) in NLLB_TAG_FROM_ISO:
                        print(
                            f"Caution: '{main.iso}' is an NLLB language code; make sure {main.stem} is the"
                            " minority-language project sharing that code, not a shared reference Bible."
                        )
                    try:
                        reply = input("Copy the file when the first experiment is created? [y/N]: ").strip().lower()
                    except EOFError:
                        reply = ""
                    if reply not in ("y", "yes"):
                        print("Aborted: the copy is required to create these experiments.")
                        return []
            elif not dry_run:
                raise FileNotFoundError(f"Neither {main.stem}.txt nor {new_stem}.txt found in {scripture_dir}.")
        main.iso, main.stem = synthetic, new_stem

    def order_pair(a: Candidate, b: Candidate) -> Optional[List[Candidate]]:
        # The higher-alignment member leads when it can; swap when only the other can be primary.
        for first, second in ((a, b), (b, a)):
            if first in primary_pool and second.parallel >= secondary_min:
                return [first, second]
        return None

    singles = [[c] for c in primary_pool]
    mixed = [pair for a, b in itertools.combinations(passing, 2) if (pair := order_pair(a, b)) is not None]
    chosen = select_experiments(singles, mixed, dry_run, top=top)

    # Decide the drafting source(s) of each experiment: --translate-scripture wins, otherwise
    # the training sources that look like published scriptures; when none qualifies the user
    # is asked for a project or for a different selection of translate books.
    if translate_scripture and chosen:
        for project in translate_scripture:
            problem = check_translate_source(projects_dir, project, translate_book_ids) if projects_dir else None
            if problem is not None:
                print(
                    f"Warning: cannot translate {';'.join(translate_book_ids)} from '{project}': {problem}."
                    " Including it anyway: it was explicitly requested with --translate-scripture."
                )
            stem = next((c.stem for c in candidates if c.name == project), None)
            counts = coverage.counts(stem) if stem is not None else None
            if counts is not None:
                incomplete_books = missing_books(counts, translate_book_ids, complete_counts)
                if incomplete_books:
                    print(
                        f"Warning: '{project}' is less than {BOOK_COMPLETENESS_THRESHOLD:.0%} complete in"
                        f" {format_books(incomplete_books)}. Including it anyway: it was explicitly"
                        " requested with --translate-scripture."
                    )

    def qualifies(source: Candidate, book_set: AbstractSet[int]) -> bool:
        counts = coverage.counts(source.stem)
        return counts is None or drafting_qualified(counts, complete_counts, book_set)

    plans: List[Optional[DraftingPlan]] = []
    current_books: str = translate_books
    current_set: AbstractSet[int] = translate_set
    replan = True
    while replan:
        # A run-wide book change applies to every experiment, including those already planned
        # with the old selection, so the whole run is replanned with the new books (sources
        # that no longer qualify under them are prompted about like any other).
        replan = False
        plans = []
        for sources in chosen:
            if translate_scripture:
                plans.append(DraftingPlan(list(translate_scripture), current_books, current_set, auto=False))
                continue
            drafting = [source.name for source in sources if qualifies(source, current_set)]
            if drafting:
                plans.append(DraftingPlan(drafting, current_books, current_set, auto=True))
                continue
            label = " + ".join(source.name for source in sources)
            print(
                f"Warning: no training source of {label} qualifies as a drafting source (a published"
                " Bible, or a complete testament containing all the books to translate)."
            )
            if dry_run:
                plans.append(DraftingPlan([], current_books, current_set, auto=False))
                continue
            resolution = prompt_drafting_source(sources, current_books, current_set, qualifies, projects_dir)
            if resolution is None:
                print(f"Skipped {label}: no drafting source was chosen.")
                plans.append(None)
                continue
            chosen_plan, run_wide = resolution
            if run_wide:
                current_books, current_set = chosen_plan.books, chosen_plan.book_set
                replan = True
                break
            plans.append(chosen_plan)

    # The automatically chosen drafting projects double as the projects to translate from;
    # make sure each project folder and every book to be translated actually exists before
    # writing the configs (once per distinct book selection).
    names_by_books: Dict[str, List[str]] = {}
    for plan in plans:
        if plan is not None and plan.auto:
            project_names = names_by_books.setdefault(plan.books, [])
            project_names.extend(name for name in plan.projects if name not in project_names)
    source_projects: Dict[Tuple[str, str], str] = {}
    for plan_books, project_names in names_by_books.items():
        book_ids = [book_number_to_id(number) for number in sorted(get_chapters(plan_books))]
        for name, project in resolve_translate_sources(project_names, book_ids, projects_dir, dry_run).items():
            source_projects[(plan_books, name)] = project

    experiments: List[Experiment] = []
    existing_experiments: List[Experiment] = []
    warned_removals: set = set()
    print()
    for sources, plan in zip(chosen, plans):
        if plan is None:
            continue  # skipped at the drafting prompt, already reported
        label = " + ".join(source.name for source in sources)
        try:
            corpus_books, removed = resolve_corpus_books(
                training_books, [s.stem for s in sources] + [counts_stem], verse_counts, exclude=plan.book_set
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
        translate_projects = [source_projects.get((plan.books, name), name) for name in plan.projects]
        translate_config = build_translate_config(translate_projects, plan.books)
        prefix = "_".join([source.name for source in sources] + [main.iso] + ([test_variant] if test_variant else []))
        existing, index = find_existing(lang_dir, prefix, config)
        if existing is not None:
            print(f"Skipped {label}: {existing} already contains an identical config.yml.")
            update_translate_config(existing, translate_config, dry_run)
            existing_experiments.append(
                Experiment(
                    sources=sources,
                    folder=existing,
                    config=config,
                    translate_config=translate_config,
                )
            )
            continue
        folder = lang_dir / f"{prefix}_{index}"
        experiment = Experiment(
            sources=sources,
            folder=folder,
            config=config,
            translate_config=translate_config,
        )
        experiments.append(experiment)
        if dry_run:
            print(f"Would create {folder} (corpus_books: {corpus_books})")
        else:
            if pending_copy is not None:
                assert scripture_dir is not None
                execute_copy(scripture_dir, terms_dir, *pending_copy)
                pending_copy = None
            folder.mkdir(parents=True, exist_ok=True)
            write_yaml(folder / "config.yml", experiment.config)
            write_yaml(folder / "translate_config.yml", experiment.translate_config)
            print(f"Created {folder} (corpus_books: {corpus_books})")
    if (experiments or existing_experiments) and not dry_run:
        # Existing folders with an identical config are offered too: their creation was
        # skipped, but the experiments themselves may not have been run yet. A dry run
        # only lists what would be created, without the run commands.
        submit_experiments(
            experiments + existing_experiments,
            experiments_dir,
            submit=submit,
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
        " corpus-stats.csv (it may appear in either column); with only an onboarding.log the"
        " value must match the log's main project",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_EXPERIMENTS,
        help=f"Maximum number of experiments listed for selection (default {TOP_EXPERIMENTS})",
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
    parser.add_argument(
        "--translate-scripture",
        nargs="+",
        metavar="PROJECT",
        help="Paratext project name(s) to draft from, overriding the automatic choice of drafting"
        " sources (the training sources that are published Bibles or complete testaments)",
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
        projects_dir=Path(environment.pt_projects_dir),
        test_variant="notest" if args.no_test else "test100" if args.test100 else None,
        target=args.target,
        translate_scripture=args.translate_scripture,
        top=args.top,
        dry_run=args.dry_run,
        submit=True if args.run else None,
    )


if __name__ == "__main__":
    main()
