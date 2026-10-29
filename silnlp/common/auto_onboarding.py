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
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from machine.corpora import FileParatextProjectSettingsParser
from machine.scripture import ALL_BOOK_IDS, book_id_to_number, book_number_to_id, is_nt, is_ot

from .create_onboarding_experiments import (
    EXPERIMENT_ARGS,
    BookCoverage,
    Candidate,
    MainProject,
    build_config,
    build_translate_config,
    compact_canons,
    find_prior_copy,
    folder_name,
    load_language_entries,
    load_name_overrides,
    load_verse_counts,
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
            planned_books=[self._book_id(book, path) for book in onboarding["planned_books"]],
            draft_source=str(onboarding["draft_source"]),
            bt_project=str(bt_project) if bt_project else None,
        )

    def corpus_stats(self, main_project: str) -> Tuple[MainProject, List[Candidate]]:
        for path in (self._folder / "corpus-stats.csv", self._folder / "alignments" / "corpus-stats.csv"):
            if path.is_file():
                return parse_corpus_stats(path, target=main_project)
        raise OnboardingError(f"No corpus-stats.csv in {self._folder} or its alignments folder.")

    def verse_counts(self) -> pd.DataFrame:
        try:
            return load_verse_counts(self._folder, self._experiments_dir)
        except FileNotFoundError as e:
            raise OnboardingError(str(e)) from e

    def _book_id(self, book: object, path: Path) -> str:
        text = str(book).strip().upper()
        book_id = book_number_to_id(int(text)) if text.isdigit() else text
        if book_id_to_number(book_id) == 0:
            raise OnboardingError(f"'{book}' in planned_books of {path} is not a book.")
        return book_id


class Thresholds:
    def __init__(self, min_parallel: int, min_alignment: float):
        self._min_parallel = min_parallel
        self._min_alignment = min_alignment

    def passes(self, candidate: Candidate) -> bool:
        return candidate.parallel >= self._min_parallel and candidate.alignment >= self._min_alignment

    def describe(self) -> str:
        return f"at least {self._min_parallel} parallel verses and alignment >= {self._min_alignment}"


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
        print(
            f"Back translation {bt.name}: alignment {bt.alignment:.4f}, {bt.parallel + bt.src_only} verses,"
            f" {bt.parallel} in parallel with the target, {bt.src_only} not in the target."
        )
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
        return to_iso3(a.iso) == to_iso3(b.iso)

    def _pair(self, source: Optional[Candidate], bt: Optional[Candidate]) -> Optional[List[Candidate]]:
        if source is None or bt is None or not self._same_language(source, bt):
            return None
        return [source, bt]


class TrainingBooks:
    def __init__(self, coverage: BookCoverage, completeness: float = 0.98):
        self._coverage = coverage
        self._completeness = completeness

    def books(self, stems: Sequence[str]) -> List[str]:
        complete = self._coverage.complete()
        counts = []
        for stem in stems:
            stem_counts = self._coverage.counts(stem)
            if stem_counts is None:
                raise OnboardingError(f"No verse counts or extract file for '{stem}'.")
            counts.append(stem_counts)
        canon = [book for book in complete if is_ot(book_id_to_number(book)) or is_nt(book_id_to_number(book))]
        canon.sort(key=book_id_to_number)
        return [
            book
            for book in canon
            if complete[book] > 0 and all(c.get(book, 0) >= self._completeness * complete[book] for c in counts)
        ]


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


class DraftBookChooser:
    def __init__(self, planned_books: Sequence[str], target_books: AbstractSet[str], paratext: ParatextBooks):
        nt = "MAT MRK LUK JHN ACT 1JN 2JN 3JN 1TH 2TH REV 1CO 2TI GAL 1PE PHP ROM TIT EPH PHM JAS 2PE JUD 1TI COL"
        nt += " HEB 2CO"
        ot = "RUT JOS EST JON GEN EXO LEV NUM DEU JDG 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH ISA JER LAM EZK DAN HOS JOL"
        ot += " AMO OBA MIC NAM HAB ZEP HAG ZEC MAL JOB ECC SNG PRO PSA"
        deuterocanon = "TOB JDT ESG WIS SIR BAR LJE S3Y SUS BEL 1MA 2MA 3MA 4MA 1ES 2ES MAN PS2 ODA PSS EZA JUB ENO"
        self._ordered = ot.split()
        # A team planning only OT books is not offered NT drafts.
        if set(planned_books) & set(nt.split()):
            self._ordered = nt.split() + self._ordered
        # Deuterocanonical books are drafted only for a team that has already translated some.
        if target_books & set(deuterocanon.split()):
            self._ordered += deuterocanon.split()
        self._planned = set(planned_books)
        self._paratext = paratext

    def choose(self, drafting_project: str, training_books: AbstractSet[str]) -> List[str]:
        eligible = [
            book for book in self._ordered if book not in training_books and self._paratext.has(drafting_project, book)
        ]
        planned = [book for book in eligible if book in self._planned][:2]
        others = [book for book in eligible if book not in planned]
        return sorted(planned + others[: 2 - len(planned)], key=self._ordered.index)


class TargetExtractCopy:
    def __init__(self, scripture_dir: Path, terms_dir: Path, real_isos: AbstractSet[str]):
        self._scripture_dir = scripture_dir
        self._terms_dir = terms_dir
        self._real_isos = real_isos
        self._pending: Optional[Tuple[str, str]] = None

    def apply(self, main: MainProject, sources: Sequence[Candidate]) -> MainProject:
        clashing = sorted({c.name for c in sources if to_iso3(c.iso) == to_iso3(main.iso)})
        prior_iso = find_prior_copy(self._scripture_dir, main, self._real_isos)
        if not clashing and prior_iso is None:
            return main
        synthetic = prior_iso or synthesize_trg_iso(to_iso3(main.iso) or main.iso, self._real_isos)
        new_stem = f"{synthetic}-{main.name}"
        if not (self._scripture_dir / f"{main.stem}.txt").is_file() and not self._copy_path(new_stem).is_file():
            raise OnboardingError(f"Neither {main.stem}.txt nor {new_stem}.txt is in {self._scripture_dir}.")
        if clashing:
            print(
                f"{', '.join(clashing)} share(s) iso code '{main.iso}' with the target; the target uses '{synthetic}'."
            )
        else:
            print(f"The target uses synthetic iso code '{synthetic}' from an earlier copy of its extract.")
        self._pending = (main.stem, new_stem)
        self._describe(main.stem, new_stem)
        return dataclasses.replace(main, iso=synthetic, stem=new_stem)

    def execute(self) -> None:
        if self._pending is None:
            return
        old_stem, new_stem = self._pending
        old_path, new_path = self._scripture_dir / f"{old_stem}.txt", self._copy_path(new_stem)
        if new_path.is_file() and self._is_outdated(old_path, new_path):
            renamed = self._free_backup(new_stem)
            new_path.rename(renamed)
            print(f"Warning: renamed the outdated copy {new_path} to {renamed}.")
        if not new_path.is_file():
            shutil.copyfile(old_path, new_path)
            print(f"Copied {old_path} to {new_path}.")
        if self._terms_dir.is_dir():
            for path in sorted(self._terms_dir.glob(f"{old_stem}-*-renderings.txt")):
                target = self._terms_dir / f"{new_stem}{path.name[len(old_stem):]}"
                shutil.copyfile(path, target)
                print(f"Copied {path} to {target}.")
        self._pending = None

    def _describe(self, old_stem: str, new_stem: str) -> None:
        old_path, new_path = self._scripture_dir / f"{old_stem}.txt", self._copy_path(new_stem)
        if not new_path.is_file():
            print(f"  Will copy {old_path} to {new_path}; the original is kept.")
        elif self._is_outdated(old_path, new_path):
            print(
                f"  {new_path} is older than {old_path}: it will be renamed to {self._free_backup(new_stem)}"
                " and copied again."
            )
        else:
            print(f"  Reusing {new_path} (a copy of {old_path}).")
        print("  Matching MT/terms renderings copies will be overwritten with the current ones.")

    def _copy_path(self, stem: str) -> Path:
        return self._scripture_dir / f"{stem}.txt"

    def _is_outdated(self, original: Path, copy: Path) -> bool:
        return original.is_file() and original.stat().st_mtime > copy.stat().st_mtime

    def _free_backup(self, stem: str) -> Path:
        index = 1
        while (self._scripture_dir / f"{stem}_{index}.txt").is_file():
            index += 1
        return self._scripture_dir / f"{stem}_{index}.txt"


class Confirmer:
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
        verse_counts = request.verse_counts()
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

        coverage = BookCoverage(verse_counts, self._paths.scripture_dir, self._paths.assets_dir)
        target_books = {book for book, count in (coverage.counts(main.stem) or {}).items() if count > 0}
        training = TrainingBooks(coverage)
        drafts = DraftBookChooser(sign_up.planned_books, target_books, ParatextBooks(self._paths.projects_dir))
        copier = TargetExtractCopy(
            self._paths.scripture_dir, self._paths.terms_dir, {entry["isoCode"] for entry in language_entries}
        )
        counts_stem = main.stem
        main = copier.apply(main, [source for sources in choices for source in sources])

        planned: List[PlannedExperiment] = []
        print()
        for sources in choices:
            label = " + ".join(source.name for source in sources)
            training_books = training.books([s.stem for s in sources] + [counts_stem])
            if not training_books:
                print(f"Skipped {label}: no book is complete in the target and every source.")
                continue
            books = drafts.choose(sources[0].name, set(training_books))
            config = build_config(sources, main, compact_canons(training_books))
            translate_config = build_translate_config([sources[0].name], ";".join(books)) if books else None
            prefix = "_".join([source.name for source in sources] + [main.iso])
            folder = lang_dir / f"{prefix}_{self._next_index(lang_dir, prefix)}"
            planned.append(PlannedExperiment(sources, folder, config, translate_config))
            print(f"{label}: train on {compact_canons(training_books)}")
            if books:
                print(f"  draft {';'.join(books)} from {sources[0].name}")
            else:
                print(f"  Warning: {sources[0].name} has no eligible book to draft; training only.")
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
            print(f"  poetry run python {' '.join(self._runner.arguments(translate))} {name}")
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
