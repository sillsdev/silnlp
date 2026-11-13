"""Create up to four experiments for an onboarding request without asking about sources or books.

The rules are in automated_onboarding_experiments_plan.md; create_onboarding_experiments.py
remains the interactive tool for overriding them by hand.
"""

import argparse
import dataclasses
import fnmatch
import json
import logging
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

import yaml
from machine.corpora import FileParatextProjectSettingsParser
from machine.scripture import ALL_BOOK_IDS, book_id_to_number, book_number_to_id, is_canonical, is_nt

from .create_onboarding_experiments import (
    EXPERIMENT_ARGS,
    Candidate,
    MainProject,
    build_config,
    build_translate_config,
    compact_canons,
    find_prior_copy,
    folder_name,
    load_language_entries,
    load_name_overrides,
    load_vref_books,
    lookup_language,
    old_name_folder_warning,
    parse_corpus_stats,
    synthesize_trg_iso,
    to_iso3,
    write_yaml,
)
from .environment import SilNlpEnv


class OnboardingError(Exception):
    pass


@dataclass
class SignUp:
    main_project: str
    planned_books: List[str]
    completed_books: List[str]
    draft_source: str
    bt_project: Optional[str]


@dataclass
class DataPaths:
    experiments_dir: Path
    assets_dir: Path
    scripture_dir: Path
    terms_dir: Path
    projects_dir: Path


class RequestFolder:
    def __init__(self, experiments_dir: Path):
        self._experiments_dir = experiments_dir

    def resolve(self, name: str) -> Path:
        for folder in (self._experiments_dir / name, self._experiments_dir / f"{name.rstrip('/')}_Request"):
            if folder.is_dir():
                return folder
        raise OnboardingError(f"Neither {name} nor {name.rstrip('/')}_Request is a folder in {self._experiments_dir}.")


class OnboardingRequest:
    def __init__(self, folder: Path, experiments_dir: Path):
        self._folder = folder
        self._experiments_dir = experiments_dir

    def sign_up(self) -> SignUp:
        path = self._folder / "onboarding_config.yml"
        if not path.is_file():
            raise OnboardingError(f"No onboarding_config.yml in {self._folder}.")
        with open(path, "r", encoding="utf-8") as file:
            onboarding = (yaml.safe_load(file) or {}).get("onboarding") or {}
        missing = [key for key in ("main_project", "planned_books", "draft_source") if not onboarding.get(key)]
        if missing:
            raise OnboardingError(f"{path} has no value for: {', '.join(missing)}.")
        bt_project = onboarding.get("bt_project")
        return SignUp(
            main_project=str(onboarding["main_project"]),
            planned_books=[self._book_id(book, "planned_books", path) for book in onboarding["planned_books"]],
            completed_books=[
                self._book_id(book, "completed_books", path) for book in onboarding.get("completed_books") or []
            ],
            draft_source=str(onboarding["draft_source"]),
            bt_project=str(bt_project) if bt_project else None,
        )

    def corpus_stats(self, main_project: str) -> Tuple[MainProject, List[Candidate]]:
        for path in (self._folder / "corpus-stats.csv", self._folder / "alignments" / "corpus-stats.csv"):
            if path.is_file():
                return parse_corpus_stats(path, target=main_project)
        raise OnboardingError(f"No corpus-stats.csv in {self._folder} or its alignments folder.")

    def _book_id(self, book: object, key: str, path: Path) -> str:
        text = str(book).strip().upper()
        book_id = book_number_to_id(int(text)) if text.isdigit() else text
        if book_id_to_number(book_id) == 0 or not is_canonical(book_id):
            raise OnboardingError(f"'{book}' in {key} of {path} is not a book.")
        return book_id


class Thresholds:
    def __init__(self, min_parallel: int, min_alignment: float):
        self._min_parallel = min_parallel
        self._min_alignment = min_alignment

    def passes(self, candidate: Candidate) -> bool:
        return candidate.parallel >= self._min_parallel and candidate.alignment >= self._min_alignment

    def describe(self) -> str:
        return f"at least {self._min_parallel} parallel verses and alignment >= {self._min_alignment}"


class SourceSummary:
    def __init__(self, candidate: Candidate):
        self._candidate = candidate

    def describe(self) -> str:
        c = self._candidate
        return (
            f"{c.name}: alignment {c.alignment:.4f}, {c.parallel + c.src_only} verses,"
            f" {c.parallel} in parallel with the target, {c.src_only} not in the target."
        )

    def has_no_more_data_than_target(self) -> bool:
        # Both totals include the parallel verses, so only the unshared verses differ.
        return self._candidate.src_only <= self._candidate.trg_only


class SameProjectRole:
    def __init__(self, confirmer: "Confirmer"):
        self._confirmer = confirmer

    def resolve(self, sign_up: SignUp, candidates: Sequence[Candidate]) -> SignUp:
        if sign_up.bt_project != sign_up.draft_source:
            return sign_up
        project = next((c for c in candidates if c.name == sign_up.bt_project), None)
        if project is None:
            return sign_up
        summary = SourceSummary(project)
        print(f"Warning: {project.name} is both the back translation and the draft source.")
        print(summary.describe())
        default = "back translation" if summary.has_no_more_data_than_target() else None
        role = self._confirmer.choose(f"Treat {project.name} as", ["back translation", "draft source"], default)
        if role is None:
            raise OnboardingError(f"No role chosen for {project.name}, which is both bt_project and draft_source.")
        if role == "draft source":
            return dataclasses.replace(sign_up, bt_project=None)
        return dataclasses.replace(sign_up, draft_source="")


class BackTranslationCheck:
    def __init__(self, thresholds: Thresholds):
        self._thresholds = thresholds

    def usable(self, sign_up: SignUp, candidates: Sequence[Candidate]) -> Optional[Candidate]:
        if sign_up.bt_project is None:
            print("Back translation: none named at sign-up.")
            return None
        if sign_up.bt_project == sign_up.draft_source:
            raise OnboardingError(
                f"bt_project and draft_source are both '{sign_up.draft_source}';"
                " a back translation is never a drafting source."
            )
        bt = next((c for c in candidates if c.name == sign_up.bt_project), None)
        if bt is None:
            print(
                f"Warning: back translation '{sign_up.bt_project}' is not in corpus-stats.csv; continuing without it."
            )
            return None
        print(f"Back translation {SourceSummary(bt).describe()}")
        if not self._thresholds.passes(bt):
            print(f"Warning: back translation {bt.name} needs {self._thresholds.describe()}; continuing without it.")
            return None
        return bt


class SourcePreferences:
    def __init__(self, preferred: Sequence[Sequence[str]], avoid: Sequence[str]):
        self._preferred = [list(names) for names in preferred]
        self._avoid = list(avoid)

    def is_avoided(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self._avoid)

    def later_editions(self, name: str) -> List[str]:
        """Projects preferred over `name`, most preferred first."""
        for names in self._preferred:
            if name in names:
                return names[: names.index(name)]
        return []


class SourcePreferencesFile:
    def __init__(self, assets_dir: Path):
        self._path = assets_dir / "preferredSources.json"

    def load(self) -> SourcePreferences:
        if not self._path.is_file():
            return SourcePreferences([], [])
        with open(self._path, "r", encoding="utf-8") as file:
            data = json.load(file)
        preferred = [names for lists in (data.get("preferred") or {}).values() for names in lists]
        return SourcePreferences(preferred, data.get("avoid") or [])


class ExperimentChooser:
    def __init__(self, thresholds: Thresholds, preferences: SourcePreferences, close: float = 0.025):
        self._thresholds = thresholds
        self._preferences = preferences
        self._close = close

    def choose(
        self, candidates: Sequence[Candidate], sign_up: SignUp, back_translation: Optional[Candidate]
    ) -> List[List[Candidate]]:
        passing = sorted(
            (c for c in candidates if c.name != sign_up.bt_project and self._thresholds.passes(c)),
            key=lambda c: c.alignment,
            reverse=True,
        )
        if not passing:
            print(f"No source has {self._thresholds.describe()}; no experiments.")
            return []
        pool = [c for c in passing if not self._preferences.is_avoided(c.name)] or passing
        first = next((c for c in passing if c.name == sign_up.draft_source), None)
        if first is not None:
            first = self._later_edition_of_draft_source(first, passing)
            best = self._best(pool, first)
        else:
            print(
                f"Warning: draft_source '{sign_up.draft_source}' is not a candidate (it needs"
                f" {self._thresholds.describe()}); using the two best-aligned sources instead."
            )
            best = self._best(pool, None)
            rest = [c for c in pool if c is not best]
            first = self._best(rest, None) if rest else None

        experiments: List[List[Candidate]] = []
        for sources in (
            [first] if first else None,
            [best],
            self._pair(first, back_translation),
            self._pair(best, back_translation),
        ):
            if sources is not None and sources not in experiments:
                experiments.append(sources)
        return experiments

    def _later_edition_of_draft_source(self, draft_source: Candidate, passing: Sequence[Candidate]) -> Candidate:
        if self._preferences.is_avoided(draft_source.name):
            print(f"Warning: draft_source {draft_source.name} is normally avoided; keeping it as the team's choice.")
        for name in self._preferences.later_editions(draft_source.name):
            later = next((c for c in passing if c.name == name and c.alignment > draft_source.alignment), None)
            if later is not None:
                print(
                    f"Drafting from {later.name} ({later.alignment:.4f}) instead of draft_source"
                    f" {draft_source.name} ({draft_source.alignment:.4f}): a later edition that aligns better."
                )
                return later
        return draft_source

    def _best(self, pool: Sequence[Candidate], draft_source: Optional[Candidate]) -> Candidate:
        chosen = pool[0]
        if chosen is draft_source:
            return chosen
        if draft_source is not None and not self._same_language(chosen, draft_source):
            same_language = [
                c
                for c in pool
                if self._same_language(c, draft_source)
                and c.alignment > draft_source.alignment
                and self._is_close(c, chosen)
            ]
            if same_language:
                chosen = same_language[0]
        for name in self._preferences.later_editions(chosen.name):
            later = next(
                (c for c in pool if c.name == name and c is not draft_source and self._is_close(c, chosen)), None
            )
            if later is not None:
                return later
        return chosen

    def _is_close(self, candidate: Candidate, other: Candidate) -> bool:
        return abs(candidate.alignment - other.alignment) <= self._close

    def _same_language(self, a: Candidate, b: Candidate) -> bool:
        return (to_iso3(a.iso) or a.iso) == (to_iso3(b.iso) or b.iso)

    def _pair(self, source: Optional[Candidate], bt: Optional[Candidate]) -> Optional[List[Candidate]]:
        if source is None or bt is None or not self._same_language(source, bt):
            return None
        return [source, bt]


class ExtractFiles:
    def __init__(self, scripture_dir: Path, vref_books: Sequence[str]):
        self._scripture_dir = scripture_dir
        self._vref_books = vref_books
        self._present: Dict[str, List[bool]] = {}

    def parallel_verses(self, target_stem: str, source_stem: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for book, in_target, in_source in zip(self._vref_books, self._lines(target_stem), self._lines(source_stem)):
            if in_target and in_source:
                counts[book] = counts.get(book, 0) + 1
        return counts

    def problem(self, stem: str) -> Optional[str]:
        try:
            self._lines(stem)
        except OnboardingError as e:
            return str(e)
        return None

    def books_present(self, stem: str) -> AbstractSet[str]:
        return {book for book, present in zip(self._vref_books, self._lines(stem)) if present}

    def _lines(self, stem: str) -> List[bool]:
        if stem not in self._present:
            path = self._scripture_dir / f"{stem}.txt"
            if not path.is_file():
                raise OnboardingError(f"No extract file {path}.")
            try:
                with open(path, "r", encoding="utf-8-sig") as file:
                    present = [bool(line.strip()) for line in file]
            except (OSError, UnicodeDecodeError) as e:
                raise OnboardingError(f"Could not read {path}: {e}") from e
            if len(present) != len(self._vref_books):
                raise OnboardingError(f"{path} has {len(present)} lines, but vref.txt has {len(self._vref_books)}.")
            self._present[stem] = present
        return self._present[stem]


@dataclass
class BookChoice:
    training: List[str]
    drafts: List[str]
    reason: str


class ParatextBooks:
    def __init__(self, projects_dir: Path):
        self._projects_dir = projects_dir
        self._cache: Dict[str, AbstractSet[str]] = {}

    def has(self, project: str, book: str) -> bool:
        if project not in self._cache:
            self._cache[project] = self._load(project)
        return book in self._cache[project]

    def _load(self, project: str) -> AbstractSet[str]:
        project_dir = self._projects_dir / project
        if not project_dir.is_dir():
            return frozenset()
        try:
            settings = FileParatextProjectSettingsParser(project_dir).parse()
        except Exception as e:
            print(f"Warning: the settings of project '{project}' could not be read ({e}).")
            return frozenset()
        return frozenset(book for book in ALL_BOOK_IDS if (project_dir / settings.get_book_file_name(book)).is_file())


@dataclass
class BookThresholds:
    declared_enough: int = 4000
    left_after_drafting: int = 3500
    minimum: int = 1500
    complete_gospel: float = 0.9


class ExperimentBooks:
    def __init__(
        self,
        sign_up: SignUp,
        target_books: AbstractSet[str],
        complete_counts: Dict[str, int],
        paratext: ParatextBooks,
        thresholds: BookThresholds,
    ):
        nt = (
            "MAT MRK LUK JHN ACT 1JN 2JN 3JN 1TH 2TH REV 1CO 2TI GAL 1PE PHP ROM TIT EPH PHM JAS 2PE JUD 1TI COL"
            " HEB 2CO"
        ).split()
        ot = (
            "RUT JOS EST JON GEN EXO LEV NUM DEU JDG 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH ISA JER LAM EZK DAN HOS JOL"
            " AMO OBA MIC NAM HAB ZEP HAG ZEC MAL JOB ECC SNG PRO PSA"
        ).split()
        deuterocanon = "TOB JDT ESG WIS SIR BAR LJE S3Y SUS BEL 1MA 2MA 3MA 4MA 1ES 2ES MAN PS2 ODA PSS EZA JUB ENO"
        self._ordered = nt + ot
        # Deuterocanonical books are drafted only for a team that has already translated some.
        if target_books & set(deuterocanon.split()):
            self._ordered += deuterocanon.split()
        self._gospels = ["MAT", "MRK", "LUK", "JHN"]
        planned = set(sign_up.planned_books)
        self._planned = [book for book in self._ordered if book in planned]
        ignored = [book for book in sign_up.planned_books if book not in self._ordered]
        if ignored:
            print(f"Warning: planned book(s) {';'.join(ignored)} can't be drafted for this target; ignoring them.")
        self._declared = set(sign_up.completed_books) - set(sign_up.planned_books)
        self._complete_counts = complete_counts
        self._paratext = paratext
        self._thresholds = thresholds

    def choose(self, drafting_project: str, parallel: Dict[str, int]) -> BookChoice:
        limits = self._thresholds
        declared = {book for book in self._declared if parallel.get(book, 0) > 0}
        empty = self._canonical(self._declared - declared)
        if empty:
            print(f"  Declared complete but with no parallel verses: {';'.join(empty)}")
        declared_total = self._total(declared, parallel)
        if declared_total >= limits.declared_enough:
            training = declared
            source = f"the declared books give {declared_total} parallel verses"
        else:
            training = {book for book, count in parallel.items() if count > 0}
            source = f"the declared books give only {declared_total} parallel verses, so all parallel data is used"
        training_total = self._total(training, parallel)

        if training_total < limits.minimum:
            complete_gospels = [g for g in self._gospels if g in training and self._is_complete(g, parallel)]
            short = f"{source}; {training_total} is fewer than {limits.minimum}"
            if not complete_gospels:
                return BookChoice([], [], f"{short} and there is no complete gospel")
            gospel = self._other_gospel(drafting_project, training, complete_gospels)
            if gospel is None:
                return BookChoice([], [], f"{short} and {drafting_project} has no other gospel to draft from")
            training.discard(gospel)
            reason = f"{short}, so one other gospel is drafted ({self._total(training, parallel)} training verses)"
            return BookChoice(self._canonical(training), [gospel], reason)

        drafts: List[str] = []
        for book in self._planned:
            if len(drafts) == 2:
                break
            if not self._paratext.has(drafting_project, book):
                continue
            if book in training:
                if training_total - parallel[book] < limits.left_after_drafting:
                    continue
                training.discard(book)
                training_total -= parallel[book]
            drafts.append(book)

        testaments = {self._testament(book) for book in training}
        while len(drafts) < 2:
            second = self._second_book(drafting_project, training, drafts, testaments)
            if second is None:
                break
            drafts.append(second)
        reason = f"{source}; training on {training_total}"
        return BookChoice(self._canonical(training), sorted(drafts, key=self._ordered.index), reason)

    def _other_gospel(self, project: str, training: AbstractSet[str], complete: Sequence[str]) -> Optional[str]:
        planned = [g for g in self._planned if g in self._gospels and g not in complete]
        others = [g for g in self._gospels if g not in training]
        return next((g for g in planned + others if self._paratext.has(project, g)), None)

    def _second_book(
        self, project: str, training: AbstractSet[str], drafts: Sequence[str], testaments: AbstractSet[str]
    ) -> Optional[str]:
        drafted = {self._testament(book) for book in drafts}
        if len(testaments) == 2:
            order = [t for t in ("NT", "OT") if t not in drafted] + [t for t in ("NT", "OT") if t in drafted]
        else:
            (only,) = testaments
            order = [only, "OT" if only == "NT" else "NT"]
        for testament in order:
            book = next(
                (
                    b
                    for b in self._ordered
                    if self._testament(b) == testament
                    and b not in training
                    and b not in drafts
                    and self._paratext.has(project, b)
                ),
                None,
            )
            if book is not None:
                return book
        return None

    def _is_complete(self, gospel: str, parallel: Dict[str, int]) -> bool:
        full = self._complete_counts.get(gospel, 0)
        return full > 0 and parallel.get(gospel, 0) >= self._thresholds.complete_gospel * full

    def _testament(self, book: str) -> str:
        return "NT" if is_nt(book_id_to_number(book)) else "OT"

    def _total(self, books: AbstractSet[str], parallel: Dict[str, int]) -> int:
        return sum(parallel.get(book, 0) for book in books)

    def _canonical(self, books: AbstractSet[str]) -> List[str]:
        return sorted(books, key=book_id_to_number)


class TargetExtractCopy:
    def __init__(self, scripture_dir: Path, terms_dir: Path, real_isos: AbstractSet[str], main: MainProject):
        self._scripture_dir = scripture_dir
        self._terms_dir = terms_dir
        self._real_isos = real_isos
        self._original = main
        self._prior_iso = find_prior_copy(scripture_dir, main, real_isos)
        self._copy_stem: Optional[str] = f"{self._prior_iso}-{main.name}" if self._prior_iso else None
        original = self._path(main.stem)
        prior = self._path(self._copy_stem) if self._copy_stem else None
        # Decided once: an earlier copy is kept unless the original has been re-extracted since.
        self._reuse_prior = (
            prior is not None
            and prior.is_file()
            and not (original.is_file() and original.stat().st_mtime > prior.stat().st_mtime)
        )

    def extract_to_read(self) -> str:
        """The extract whose content the experiments' target will have once execute() has run."""
        return self._copy_stem if self._reuse_prior and self._copy_stem else self._original.stem

    def apply(self, sources: Sequence[Candidate]) -> MainProject:
        main = self._original
        target_iso = to_iso3(main.iso) or main.iso
        clashing = sorted({c.name for c in sources if (to_iso3(c.iso) or c.iso) == target_iso})
        if not clashing and self._prior_iso is None:
            return main
        synthetic = self._prior_iso or synthesize_trg_iso(target_iso, self._real_isos)
        new_stem = f"{synthetic}-{main.name}"
        self._copy_stem = new_stem
        original, copy = self._path(main.stem), self._path(new_stem)
        if clashing:
            print(
                f"{', '.join(clashing)} share(s) iso code '{main.iso}' with the target; the target uses '{synthetic}'."
            )
        else:
            print(f"The target uses synthetic iso code '{synthetic}' from an earlier copy of its extract.")
        if self._reuse_prior:
            print(f"  Reusing {copy} (a copy of {original}).")
        elif copy.is_file():
            print(
                f"  {copy} is older than {original}: it will be renamed to {self._backup(new_stem)} and copied again."
            )
        else:
            print(f"  Will copy {original} to {copy}; the original is kept.")
        print("  Matching MT/terms renderings copies will be overwritten with the current ones.")
        return dataclasses.replace(main, iso=synthetic, stem=new_stem)

    def execute(self) -> None:
        if self._copy_stem is None:
            return
        old_stem, new_stem = self._original.stem, self._copy_stem
        original, copy = self._path(old_stem), self._path(new_stem)
        if not self._reuse_prior:
            if copy.is_file():
                renamed = self._backup(new_stem)
                copy.rename(renamed)
                print(f"Warning: renamed the outdated copy {copy} to {renamed}.")
            shutil.copyfile(original, copy)
            print(f"Copied {original} to {copy}.")
        if self._terms_dir.is_dir():
            for path in sorted(self._terms_dir.glob(f"{old_stem}-*-renderings.txt")):
                target = self._terms_dir / f"{new_stem}{path.name[len(old_stem):]}"
                shutil.copyfile(path, target)
                print(f"Copied {path} to {target}.")
        self._copy_stem = None

    def _path(self, stem: str) -> Path:
        return self._scripture_dir / f"{stem}.txt"

    def _backup(self, stem: str) -> Path:
        index = 1
        while self._path(f"{stem}_{index}").is_file():
            index += 1
        return self._path(f"{stem}_{index}")


class Confirmer:
    def choose(self, question: str, options: Sequence[str], default: Optional[str]) -> Optional[str]:
        suffix = f" (default: {default})" if default else ""
        while True:
            try:
                reply = input(f"{question} [{'/'.join(options)}]{suffix}: ").strip().lower()
            except EOFError:
                print("No answer; stopping.")
                return None
            if not reply and default:
                return default
            matches = [option for option in options if option.startswith(reply)] if reply else []
            if len(matches) == 1:
                return matches[0]

    def confirm(self, question: str) -> bool:
        try:
            reply = input(f"{question} [Y/n]: ").strip().lower()
        except EOFError:
            print("No answer; stopping.")
            return False
        return reply in ("", "y", "yes")


class ExperimentRunner:
    def run(self, name: str, translate: bool) -> int:
        return subprocess.run([sys.executable] + self.arguments(translate) + [name]).returncode

    def arguments(self, translate: bool) -> List[str]:
        return [arg for arg in EXPERIMENT_ARGS if translate or arg != "--translate"]


@dataclass
class PlannedExperiment:
    sources: List[Candidate]
    folder: Path
    config: dict
    translate_config: Optional[dict]


class AutoOnboarding:
    def __init__(self, paths: DataPaths, confirmer: Confirmer, runner: ExperimentRunner):
        self._paths = paths
        self._confirmer = confirmer
        self._runner = runner

    def run(self, folder: Path) -> List[PlannedExperiment]:
        request = OnboardingRequest(folder, self._paths.experiments_dir)
        sign_up = request.sign_up()
        main, candidates = request.corpus_stats(sign_up.main_project)
        sign_up = SameProjectRole(self._confirmer).resolve(sign_up, candidates)
        lang_dir, language_entries = self._language_dir(main)
        print(f"Target: {main.name} ({main.stem}); planned books: {';'.join(sign_up.planned_books)}")
        print(f"Experiment location: {lang_dir}")
        self._print_candidates(candidates, sign_up)

        back_translation = BackTranslationCheck(Thresholds(400, 0.25)).usable(sign_up, candidates)
        preferences = SourcePreferencesFile(self._paths.assets_dir).load()
        chooser = ExperimentChooser(Thresholds(1000, 0.2), preferences)
        choices = chooser.choose(candidates, sign_up, back_translation)
        if not choices:
            return []

        try:
            vref_books = load_vref_books(self._paths.assets_dir)
        except FileNotFoundError as e:
            raise OnboardingError(str(e)) from e
        extracts = ExtractFiles(self._paths.scripture_dir, vref_books)
        real_isos = {entry["isoCode"] for entry in language_entries}
        copier = TargetExtractCopy(self._paths.scripture_dir, self._paths.terms_dir, real_isos, main)
        target_stem = copier.extract_to_read()
        books = ExperimentBooks(
            sign_up,
            extracts.books_present(target_stem),
            dict(Counter(vref_books)),
            ParatextBooks(self._paths.projects_dir),
            BookThresholds(),
        )

        print()
        chosen: List[Tuple[str, List[Candidate], BookChoice]] = []
        for sources in choices:
            label = " + ".join(source.name for source in sources)
            problems = [problem for problem in (extracts.problem(source.stem) for source in sources) if problem]
            if problems:
                print(f"Skipped {label}: {' '.join(problems)}")
                continue
            choice = books.choose(sources[0].name, extracts.parallel_verses(target_stem, sources[0].stem))
            if not choice.training:
                print(f"Skipped {label}: {choice.reason}.")
                continue
            chosen.append((label, sources, choice))

        if not chosen:
            print("Nothing created.")
            return []
        main = copier.apply([source for _, sources, _ in chosen for source in sources])
        planned: List[PlannedExperiment] = []
        for label, sources, choice in chosen:
            training_books = compact_canons(choice.training)
            config = build_config(sources, main, training_books)
            translate_config = (
                build_translate_config([sources[0].name], ";".join(choice.drafts)) if choice.drafts else None
            )
            prefix = "_".join([source.name for source in sources] + [main.iso])
            folder = lang_dir / f"{prefix}_{self._next_index(lang_dir, prefix)}"
            planned.append(PlannedExperiment(sources, folder, config, translate_config))
            print(f"{label}: {choice.reason}")
            print(f"  train on {training_books}")
            if choice.drafts:
                print(f"  draft {';'.join(choice.drafts)} from {sources[0].name}")
            else:
                print(f"  Warning: {sources[0].name} has no book available to draft; training only.")
            print(f"  in {folder}")

        if not planned or not self._confirmer.confirm("\nCreate these experiments?"):
            print("Nothing created.")
            return planned
        copier.execute()
        for experiment in planned:
            self._write(experiment)
        self._submit(planned)
        return planned

    def _language_dir(self, main: MainProject) -> Tuple[Path, List[dict]]:
        entries = load_language_entries(self._paths.assets_dir)
        overrides = load_name_overrides(self._paths.assets_dir)
        raw_language, raw_country = lookup_language(main.iso, entries)
        country = str(overrides["countries"].get(raw_country) or "").strip()
        language = str(overrides["languages"].get(raw_language) or "").strip()
        country_seg = folder_name(country or raw_country, keep_case=bool(country))
        language_seg = folder_name(language or raw_language, keep_case=bool(language))
        warning = old_name_folder_warning(
            self._paths.experiments_dir, raw_country, raw_language, country_seg, language_seg
        )
        if warning is not None:
            print(warning)
        return self._paths.experiments_dir / country_seg / language_seg, entries

    def _print_candidates(self, candidates: Sequence[Candidate], sign_up: SignUp) -> None:
        print(f"\n{'Source':<24} {'iso':<5} {'script':<6} {'parallel':>9} {'alignment':>10}  role")
        for c in sorted(candidates, key=lambda c: c.alignment, reverse=True):
            role = {sign_up.draft_source: "draft_source", sign_up.bt_project: "bt_project"}.get(c.name, "")
            print(f"{c.name:<24} {c.iso:<5} {c.script:<6} {c.parallel:>9} {c.alignment:>10.4f}  {role}")
        print()

    def _next_index(self, lang_dir: Path, prefix: str) -> int:
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        matches = (pattern.match(folder.name) for folder in lang_dir.iterdir()) if lang_dir.is_dir() else ()
        return max((int(m.group(1)) for m in matches if m is not None), default=0) + 1

    def _write(self, experiment: PlannedExperiment) -> None:
        experiment.folder.mkdir(parents=True)
        write_yaml(experiment.folder / "config.yml", experiment.config)
        if experiment.translate_config is not None:
            write_yaml(experiment.folder / "translate_config.yml", experiment.translate_config)
        print(f"Created {experiment.folder}")

    def _submit(self, planned: Sequence[PlannedExperiment]) -> None:
        print("\nTo run the experiments:")
        for experiment in planned:
            translate = experiment.translate_config is not None
            name = experiment.folder.relative_to(self._paths.experiments_dir).as_posix()
            print(f"  uv run python {' '.join(self._runner.arguments(translate))} {name}")
        if not self._confirmer.confirm("\nRun them now?"):
            return
        for experiment in planned:
            name = experiment.folder.relative_to(self._paths.experiments_dir).as_posix()
            code = self._runner.run(name, experiment.translate_config is not None)
            if code != 0:
                print(f"Experiment {name} exited with code {code}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Create experiments for an onboarding request automatically.")
    parser.add_argument("folder", help="Request folder relative to MT/experiments; '_Request' is added if needed")
    args = parser.parse_args()

    environment = SilNlpEnv.create_standard_environment()
    paths = DataPaths(
        experiments_dir=Path(environment.mt_experiments_dir),
        assets_dir=Path(environment.assets_dir),
        scripture_dir=Path(environment.mt_scripture_dir),
        terms_dir=Path(environment.mt_terms_dir),
        projects_dir=Path(environment.pt_projects_dir),
    )
    try:
        folder = RequestFolder(paths.experiments_dir).resolve(args.folder)
        AutoOnboarding(paths, Confirmer(), ExperimentRunner()).run(folder)
    except OnboardingError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
