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
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

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
# Optional in older logs; the main is the alignment source, so 'target only' is the reference's.
ONLY_COUNTS_RE = re.compile(r"source only count: (?P<src_only>\d+) target only count: (?P<trg_only>\d+)")

CHECKPOINT = 5000
SEED = 111
MODEL = "facebook/nllb-200-distilled-1.3B"
BOOK_COMPLETENESS_THRESHOLD = 0.98
MISSING_VERSE_WARN = 0.25  # warn when this fraction or more of the specified verses are absent
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
    src_only: int = 0  # verses only in this reference (drafting-source-only verses)
    trg_only: int = 0  # verses only in the target


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
            entry = m.groupdict()
            only = ONLY_COUNTS_RE.search(line)
            entry.update(only.groupdict() if only is not None else {"src_only": None, "trg_only": None})
            stats.append(entry)

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
            # The main is the alignment source, so the reference's own verses are 'target only'.
            src_only=int(entry["trg_only"]) if entry.get("trg_only") is not None else 0,
            trg_only=int(entry["src_only"]) if entry.get("src_only") is not None else 0,
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
        # 'source only' verses are the reference's own; which physical column that is depends
        # on the row's orientation (the reference may be the CSV's src or trg project).
        ref_is_src = ref_stem == row["src_project"]
        ref_col = "src_only" if ref_is_src else "trg_only"
        target_col = "trg_only" if ref_is_src else "src_only"
        candidates[stem_to_project(ref_stem)] = Candidate(
            name=stem_to_project(ref_stem),
            stem=ref_stem,
            iso=stem_to_iso(ref_stem),
            count=int(row["count"]),
            parallel=int(row["parallel"]),
            alignment=float(row["align_score"]),
            script=str(ref_script).strip(),
            src_only=int(row[ref_col]) if ref_col in row.index and not pd.isna(row[ref_col]) else 0,
            trg_only=int(row[target_col]) if target_col in row.index and not pd.isna(row[target_col]) else 0,
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


def load_name_overrides(assets_dir: Path) -> Dict[str, Dict[str, str]]:
    """Map official country/language names (as in languageFamilies.json) to common names.

    languageFamilies.json is kept as a verbatim Ethnologue snapshot, so friendlier folder
    names live in a separate nameOverrides.json. A missing file yields empty maps, so the
    tool still works without it; unmapped names pass through unchanged.
    """
    path = assets_dir / "nameOverrides.json"
    if not path.exists():
        return {"countries": {}, "languages": {}}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # `or {}` (not a get-default) so an explicit null in a hand-edited file is tolerated too.
    return {"countries": data.get("countries") or {}, "languages": data.get("languages") or {}}


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


def folder_name(name: str, keep_case: bool = False) -> str:
    parts = name.replace(",", "").replace("-", " ").split()
    return "_".join(parts) if keep_case else "_".join(word.capitalize() for word in parts)


def old_name_folder_warning(
    experiments_dir: Path, raw_country: str, raw_language: str, country_seg: str, language_seg: str
) -> Optional[str]:
    """Message to print when experiments already exist under the old official name(s).

    Covers both a country rename (which orphans every language under the old country folder)
    and a language-only rename (which orphans the old language folder). Non-destructive: it
    only advises a manual merge, so `find_existing` never re-uses the orphaned folder and
    silently creates duplicates. Returns None when nothing was remapped or no old folder exists.
    """
    old_country_dir = experiments_dir / folder_name(raw_country)
    new_country_dir = experiments_dir / country_seg
    if old_country_dir != new_country_dir and old_country_dir.exists():
        # The whole country folder moved, so warn at that level (all languages under it).
        old, new = old_country_dir, new_country_dir
    else:
        # Language-only rename: prior experiments live under the *current* country segment
        # (which may itself be an override), so look there — not under the official country.
        old_lang_dir = new_country_dir / folder_name(raw_language)
        new_lang_dir = new_country_dir / language_seg
        if old_lang_dir == new_lang_dir or not old_lang_dir.exists():
            return None
        old, new = old_lang_dir, new_lang_dir
    return (
        f"WARNING: a folder under the official name already exists: {old}\n"
        f"         new experiments will go under the common name: {new}\n"
        "         consider merging the old folder into the new location manually."
    )


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


def load_vref_chapters(assets_dir: Path) -> List[Tuple[str, int]]:
    """The (book id, chapter number) of each vref.txt line, for chapter-level presence checks."""
    result: List[Tuple[str, int]] = []
    with open(assets_dir / "vref.txt", "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            book, ref = line.split(" ", 1)
            result.append((book, int(ref.split(":", 1)[0])))
    return result


def selection_by_book(selection: Dict[int, List[int]]) -> Dict[str, Optional[AbstractSet[int]]]:
    """Turn a get_chapters selection ({book number: [chapters]}) into {book id: chapters or None}.

    An empty chapter list means the whole book, represented here as None (all chapters).
    """
    return {book_number_to_id(number): (set(chapters) if chapters else None) for number, chapters in selection.items()}


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
        self._vref_chapters: Optional[List[Tuple[str, int]]] = None
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

    def _chapters(self) -> List[Tuple[str, int]]:
        if self._vref_chapters is None:
            self._vref_chapters = load_vref_chapters(self._assets_dir)
        return self._vref_chapters

    def presence(self, stem: str, selection: Dict[int, List[int]]) -> Optional[Tuple[int, int]]:
        """(present, total) specified verses of a get_chapters selection in the stem's extract.

        `total` is how many verses the selection covers in the full vref layout; `present` is
        how many of those are non-blank in the extract. Chapter-level: honours selections like
        `GEN 1-10`. Returns None when the extract file is unavailable (so no check can be made).
        """
        if self._scripture_dir is None:
            return None
        extract_path = self._scripture_dir / f"{stem}.txt"
        if not extract_path.is_file():
            return None
        wanted = selection_by_book(selection)
        flags = [
            book in wanted and (wanted[book] is None or chapter in (wanted[book] or ()))
            for book, chapter in self._chapters()
        ]
        total = sum(flags)
        present = 0
        with open(extract_path, "r", encoding="utf-8") as file:
            for in_scope, line in zip(flags, file):
                if in_scope and line.strip():
                    present += 1
        return present, total


def book_mark(counts: Optional[Dict[str, int]], book: str, complete: Dict[str, int]) -> str:
    """Three-state coverage of a single book in a source: full '✓', partial '~', or none 'X'."""
    have = 0 if counts is None else counts.get(book, 0)
    if have <= 0:
        return "X"
    full = complete.get(book, 0)
    return "✓" if full > 0 and have >= BOOK_COMPLETENESS_THRESHOLD * full else "~"


def warn_missing_verses(source_name: str, kind: str, selection_str: str, presence: Optional[Tuple[int, int]]) -> None:
    """Warn when at least MISSING_VERSE_WARN of the verses a selection specifies are absent.

    `presence` is (present, total) from BookCoverage.presence, or None when it could not be
    measured (no extract file), in which case nothing is printed.
    """
    if presence is None:
        return
    present, total = presence
    if total == 0 or present / total > 1 - MISSING_VERSE_WARN:
        return
    print(
        f"Warning: '{source_name}' is missing {100 * (1 - present / total):.0f}% of the {kind} verses"
        f" specified by '{selection_str}' ({present} of {total} present); check before running."
    )


def overlapping_books(training: Dict[int, List[int]], translate: Dict[int, List[int]]) -> List[str]:
    """Book ids whose verses fall in both a training and a translate selection.

    Each argument is a get_chapters result ({book number: [chapters]}, [] = whole book). A book
    overlaps when either side takes the whole book, or their chapter lists intersect.
    """
    overlap = []
    for number in sorted(set(training) & set(translate)):
        train_chapters, translate_chapters = training[number], translate[number]
        if not train_chapters or not translate_chapters or set(train_chapters) & set(translate_chapters):
            overlap.append(book_number_to_id(number))
    return overlap


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


def select_candidates(
    candidates: List[Candidate],
    coverage: "BookCoverage",
    complete_counts: Dict[str, int],
    translate_book_ids: Sequence[str],
    dry_run: bool,
) -> List[Candidate]:
    """Show a table of candidates and ask which to use as training/drafting sources.

    Each candidate appears once with its corpus-stats data (alignment, total, parallel
    'train' verses, source-only 'draft' verses, target-only, script) and a per-translate-book
    coverage mark. Manual selection replaces the book-coverage filter: whatever is chosen may
    be a primary source regardless of its coverage. Under dry_run the table is displayed and
    every candidate is returned without prompting.
    """
    name_w = max([len("Candidate")] + [len(c.name) for c in candidates])
    book_w = {book: max(3, len(book)) for book in translate_book_ids}
    marks_header = "".join(f"  {book:>{book_w[book]}}" for book in translate_book_ids)
    print("\nCandidates (train = parallel/training verses, draft = source-only/drafting verses):")
    print(
        f"  # {'Candidate':<{name_w}}  {'align':>6}  {'total':>7}  {'train':>7}  {'draft':>7}"
        f"  {'trg-only':>8}  {'script':<6}{marks_header}"
    )
    for i, c in enumerate(candidates, start=1):
        counts = coverage.counts(c.stem)
        marks = "".join(f"  {book_mark(counts, book, complete_counts):>{book_w[book]}}" for book in translate_book_ids)
        print(
            f"  {i:>2} {c.name:<{name_w}}  {c.alignment:>6.3f}  {c.count:>7}  {c.parallel:>7}  {c.src_only:>7}"
            f"  {c.trg_only:>8}  {(c.script or ''):<6}{marks}"
        )
    print("Marks: ✓ = source has ≥98% of the book, ~ = partial, X = none.")
    if dry_run:
        print("Dry run: all candidates are included.")
        return candidates
    try:
        reply = input("Enter the candidates to use (e.g. 1,3), 'all' or 'none': ").strip().lower()
    except EOFError:
        reply = ""
    if reply in ("", "none"):
        print("No candidates selected.")
        return []
    if reply != "all":
        chosen: List[Candidate] = []
        for token in re.split(r"[,\s]+", reply):
            if token.isdigit() and 1 <= int(token) <= len(candidates):
                if candidates[int(token) - 1] not in chosen:
                    chosen.append(candidates[int(token) - 1])
            else:
                LOGGER.warning(f"Ignoring invalid selection '{token}'.")
        if not chosen:
            print("No candidates selected.")
        return chosen
    return candidates


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
    --translate-scripture and a replaced drafting project change only the latter.
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
    overrides = load_name_overrides(assets_dir)
    raw_language, raw_country = lookup_language(main.iso, language_entries)

    # Map official names to common ones for friendlier folders; keep the official
    # (raw) names to detect and warn about folders created under the old naming. A blank
    # (whitespace-only) override is ignored so it cannot collapse a path level.
    country_override = overrides["countries"].get(raw_country)
    language_override = overrides["languages"].get(raw_language)
    use_country = isinstance(country_override, str) and bool(country_override.strip())
    use_language = isinstance(language_override, str) and bool(language_override.strip())
    country = country_override if use_country else raw_country
    language = language_override if use_language else raw_language
    # keep_case only for override values (already human-chosen); official names keep the
    # default title-casing so a blank/absent override collapses to the original behaviour.
    country_seg = folder_name(country, keep_case=use_country)
    language_seg = folder_name(language, keep_case=use_language)

    lang_dir = lang_dir_override or experiments_dir / country_seg / language_seg
    print(f"Main project: {main.name} ({main.stem}), language: {language} " f"[{main.iso}], country: {country}")
    print(f"Experiment location: {lang_dir}")

    # Warn (non-destructive) when a folder under the old official name(s) already exists, so it
    # can be merged manually. Only in the derived-location case — the corpus-stats override path
    # deliberately reuses whatever naming already exists.
    if lang_dir_override is None:
        warning = old_name_folder_warning(experiments_dir, raw_country, raw_language, country_seg, language_seg)
        if warning is not None:
            print(warning)

    candidates.sort(key=lambda c: c.alignment, reverse=True)
    passing = [c for c in candidates if c.parallel >= min_parallel and c.alignment >= min_alignment]
    print(f"\n{'Reference':<24} {'iso':<5} {'count':>7} {'parallel':>9} {'alignment':>10}  result")
    for c in candidates:
        result = "pass" if c in passing else "fail"
        print(f"{c.name:<24} {c.iso:<5} {c.count:>7} {c.parallel:>9} {c.alignment:>10.4f}  {result}")
    if not passing:
        print(f"\nNo references passed the thresholds (parallel >= {min_parallel}, alignment >= {min_alignment}).")
        return []

    # Verse counts drive the candidate table below: per source it shows the parallel (training)
    # and source-only (drafting) verse counts and a coverage mark for each --translate-book, so
    # the user can judge which sources to use — including whether one looks like a back
    # translation (narrow book coverage). See create_onboarding_experiments_brief.md.
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
    target_total = sum(target_counts.values()) if target_counts is not None else main.verses
    secondary_min = max(1000.0, 0.25 * target_total) if target_total else 1000.0
    if not target_total:
        LOGGER.warning(
            f"No verse count found for the target '{main.stem}'; the second-source threshold"
            f" falls back to {secondary_min:.0f} parallel verses (25% of the target is unknown)."
        )

    # The user picks which candidates to use from the table; this manual choice replaces the
    # automatic book-coverage filter (which over-excluded sources narrower than a partial
    # target). A chosen candidate may be a primary/single source regardless of its coverage.
    selected = select_candidates(passing, coverage, complete_counts, translate_book_ids, dry_run)
    if not selected:
        return []
    for c in selected:
        if c.parallel < secondary_min:
            print(
                f"Note: {c.name} can be a single source or the primary of a pair, but not a pair's"
                f" second source: its {c.parallel} parallel verses are below {secondary_min:.0f}"
                " (max of 1000 and 25% of the target's verses)."
            )

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
    # Every selected candidate appears at least as a single-source experiment, so any of them
    # sharing the target's iso forces the synthetic code and the extract copy.
    clashing = [c for c in selected if to_iso3(c.iso) == to_iso3(main.iso)]
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
        # The higher-alignment source leads; the other must clear the second-source minimum.
        lead, other = sorted((a, b), key=lambda c: c.alignment, reverse=True)
        for first, second in ((lead, other), (other, lead)):
            if second.parallel >= secondary_min:
                return [first, second]
        return None

    ordered = sorted(selected, key=lambda c: c.alignment, reverse=True)
    singles = [[c] for c in ordered]
    mixed = [pair for a, b in itertools.combinations(ordered, 2) if (pair := order_pair(a, b)) is not None]
    chosen = select_experiments(singles, mixed, dry_run, top=top)

    # Every source of an experiment is also asked to draft; --translate-scripture overrides the
    # drafting projects for all experiments. There is no drafting-qualification gate — a source
    # sparse in the translate selection is warned about below, not excluded.
    translate_selection = get_chapters(translate_books)
    training_is_complete = training_books.lower() == "complete"

    # Verify the drafting projects as Paratext translate sources (their book files are present),
    # prompting for a different project when one is missing. --translate-scripture projects are
    # used exactly as given (warned about, never replaced).
    source_projects: Dict[str, str] = {}
    if translate_scripture:
        if chosen and projects_dir is not None:
            for project in translate_scripture:
                problem = check_translate_source(projects_dir, project, translate_book_ids)
                if problem is not None:
                    print(
                        f"Warning: cannot translate {';'.join(translate_book_ids)} from '{project}': {problem}."
                        " Including it anyway: it was explicitly requested with --translate-scripture."
                    )
    else:
        source_names = list(dict.fromkeys(source.name for exp in chosen for source in exp))
        source_projects = resolve_translate_sources(source_names, translate_book_ids, projects_dir, dry_run)

    def drafting_projects_for(sources: List[Candidate]) -> List[str]:
        if translate_scripture:
            return list(translate_scripture)
        return [source_projects.get(source.name, source.name) for source in sources]

    experiments: List[Experiment] = []
    existing_experiments: List[Experiment] = []
    warned_removals: set = set()
    warned_missing: set = set()  # (stem, kind) pairs already warned about
    warned_overlap: set = set()  # training selections already checked for translate-book overlap
    print()
    for sources in chosen:
        label = " + ".join(source.name for source in sources)
        # corpus_books is the user's --training-books spec verbatim (they subtract what they
        # want, e.g. NT;-MRK); only the auto-derived 'complete' list removes the translate books.
        try:
            corpus_books, removed = resolve_corpus_books(
                training_books,
                [s.stem for s in sources] + [counts_stem],
                verse_counts,
                exclude=translate_set if training_is_complete else frozenset(),
            )
        except ValueError as e:
            print(f"Skipped {label}: {e}")
            continue
        if removed and tuple(removed) not in warned_removals:
            warned_removals.add(tuple(removed))
            print(f"Warning: excluded the books being translated from corpus_books: {';'.join(removed)}")
        if not corpus_books:
            print(f"Skipped {label}: no training books remain in '{training_books}' after the exclusions.")
            continue
        # Warn (once per selection) when the training and translate books overlap: the model
        # would train on text it is also meant to draft/test. Not blocked — the user decides.
        training_selection = get_chapters(corpus_books)
        if corpus_books not in warned_overlap:
            warned_overlap.add(corpus_books)
            overlap = overlapping_books(training_selection, translate_selection)
            if overlap:
                shown = ";".join(overlap[:10]) + (f" (+{len(overlap) - 10} more)" if len(overlap) > 10 else "")
                print(
                    f"Warning: the training books '{corpus_books}' and translate books '{translate_books}'"
                    f" overlap in {shown}; the model would train on text it is meant to translate."
                )
        # Warn (once per source) when a source is missing a quarter or more of the verses the
        # translate or training selection specifies (chapter-level; only where an extract exists).
        for source in sources:
            if (source.stem, "translate") not in warned_missing:
                warned_missing.add((source.stem, "translate"))
                warn_missing_verses(
                    source.name, "translate", translate_books, coverage.presence(source.stem, translate_selection)
                )
            if (source.stem, corpus_books) not in warned_missing:
                warned_missing.add((source.stem, corpus_books))
                warn_missing_verses(
                    source.name, "training", corpus_books, coverage.presence(source.stem, training_selection)
                )
        config = build_config(sources, main, corpus_books, test_variant)
        translate_projects = drafting_projects_for(sources)
        translate_config = build_translate_config(translate_projects, translate_books)
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
        help="Paratext project name(s) to draft from for every experiment, overriding the default"
        " of drafting from each experiment's own training sources",
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
