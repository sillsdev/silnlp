import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytest
import yaml
from machine.corpora import FileParatextProjectSettingsParser

from silnlp.common.auto_onboarding import (
    AutoOnboarding,
    BackTranslationCheck,
    Confirmer,
    DataPaths,
    DraftBookChooser,
    ExperimentChooser,
    ExperimentRunner,
    OnboardingError,
    OnboardingRequest,
    ParatextBooks,
    RequestFolder,
    SignUp,
    SourcePreferences,
    SourcePreferencesFile,
    Thresholds,
)
from silnlp.common.create_onboarding_experiments import Candidate

ASSETS_DIR = Path(__file__).parent.parent.parent / "silnlp" / "assets"

COMPLETE = {"GEN": 1533, "RUT": 85, "MAT": 1071, "MRK": 678, "LUK": 1151, "JHN": 879, "TOB": 244}
ALL_BOOKS = list(COMPLETE)


class ScriptedConfirmer(Confirmer):
    def __init__(self, *answers: bool):
        self._answers = list(answers)
        self.questions: List[str] = []

    def confirm(self, question: str) -> bool:
        self.questions.append(question)
        return self._answers.pop(0)


class RecordingRunner(ExperimentRunner):
    def __init__(self):
        self.runs: List[Tuple[str, bool]] = []

    def run(self, name: str, translate: bool) -> int:
        self.runs.append((name, translate))
        return 0


class KnownBooks(ParatextBooks):
    def __init__(self, books: Dict[str, Sequence[str]]):
        self._books = books

    def has(self, project: str, book: str) -> bool:
        return book in self._books.get(project, ())


def candidate(name: str, iso: str = "en", parallel: int = 5000, alignment: float = 0.3, script: str = "Latn"):
    return Candidate(
        name=name, stem=f"{iso}-{name}", iso=iso, count=parallel, parallel=parallel, alignment=alignment, script=script
    )


def chooser_with(preferred=(), avoid=()) -> ExperimentChooser:
    return ExperimentChooser(Thresholds(1000, 0.2), SourcePreferences(preferred, avoid))


def chosen_names(experiments) -> List[List[str]]:
    return [[source.name for source in experiment] for experiment in experiments]


def sign_up(draft_source: str = "SRCB", bt_project: Optional[str] = None, planned: Sequence[str] = ("MRK", "LUK")):
    return SignUp(main_project="TRG", planned_books=list(planned), draft_source=draft_source, bt_project=bt_project)


class RequestFixture:
    """A request folder plus the shared data folders it refers to, all under tmp_path."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.paths = DataPaths(
            experiments_dir=tmp_path / "MT" / "experiments",
            assets_dir=ASSETS_DIR,
            scripture_dir=tmp_path / "MT" / "scripture",
            terms_dir=tmp_path / "MT" / "terms",
            projects_dir=tmp_path / "Paratext" / "projects",
        )
        self.folder = self.paths.experiments_dir / "_OnboardingRequests" / "TRG_Request"
        self.folder.mkdir(parents=True)
        self.paths.scripture_dir.mkdir(parents=True)
        self.paths.terms_dir.mkdir(parents=True)
        self.config: dict = {
            "main_project": "TRG",
            "planned_books": ["41", "42"],
            "draft_source": "SRCB",
            "bt_project": "BTX",
        }
        self.stats: Dict[str, Tuple[int, float, str]] = {
            "en-SRCA": (5000, 0.40, "Latn"),
            "en-SRCB": (5000, 0.35, "Latn"),
            "fr-SRCC": (5000, 0.30, "Latn"),
            "en-BTX": (800, 0.60, "Latn"),
        }
        self.target_books = {"MAT": 1071, "JHN": 879}
        self.target_stem = "tpi-TRG"
        self.projects: Dict[str, List[str]] = {name: ALL_BOOKS for name in ("SRCA", "SRCB", "SRCC")}

    def write(self) -> "RequestFixture":
        (self.folder / "onboarding_config.yml").write_text(
            yaml.safe_dump({"onboarding": self.config}), encoding="utf-8"
        )
        rows = [
            {
                "src_project": self.target_stem,
                "trg_project": stem,
                "count": parallel + 100,
                "src_only": 0,
                "trg_only": 100,
                "parallel": parallel,
                "align_score": alignment,
                "src_script": "Latn",
                "trg_script": script,
            }
            for stem, (parallel, alignment, script) in self.stats.items()
        ]
        pd.DataFrame(rows).to_csv(self.folder / "corpus-stats.csv", index=False)
        counts = [{"file": "complete", **COMPLETE}, {"file": self.target_stem, **self.target_books}]
        counts += [{"file": stem, **COMPLETE} for stem in self.stats]
        pd.DataFrame(counts).fillna(0).to_csv(self.folder / "verse_counts.csv", index=False)
        (self.paths.scripture_dir / f"{self.target_stem}.txt").write_text("target\n", encoding="utf-8")
        for name, books in self.projects.items():
            make_paratext_project(self.paths.projects_dir, name, books)
        return self

    def run(self, *answers: bool) -> Tuple[list, RecordingRunner, ScriptedConfirmer]:
        runner = RecordingRunner()
        confirmer = ScriptedConfirmer(*answers)
        planned = AutoOnboarding(self.paths, confirmer, runner).run(self.folder)
        return planned, runner, confirmer


def make_paratext_project(projects_dir: Path, name: str, books: Sequence[str]) -> None:
    project_dir = projects_dir / name
    project_dir.mkdir(parents=True)
    (project_dir / "Settings.xml").write_text(
        "<ScriptureText>\n"
        "  <Versification>4</Versification>\n"
        "  <LanguageIsoCode>en:::</LanguageIsoCode>\n"
        "  <Encoding>65001</Encoding>\n"
        f"  <Name>{name}</Name>\n"
        f'  <Naming PrePart="" BookNameForm="41MAT" PostPart="{name}.SFM" />\n'
        "</ScriptureText>\n",
        encoding="utf-8",
    )
    settings = FileParatextProjectSettingsParser(project_dir).parse()
    for book in books:
        (project_dir / settings.get_book_file_name(book)).write_text(f"\\id {book}\n", encoding="utf-8")


def source_names(planned) -> List[List[str]]:
    return [[source.name for source in experiment.sources] for experiment in planned]


@pytest.fixture
def request_fixture(tmp_path: Path) -> RequestFixture:
    return RequestFixture(tmp_path)


# Inputs


def test_missing_onboarding_config_stops(request_fixture: RequestFixture):
    request_fixture.write()
    (request_fixture.folder / "onboarding_config.yml").unlink()
    with pytest.raises(OnboardingError, match="onboarding_config.yml"):
        request_fixture.run()


def test_missing_planned_books_stops(request_fixture: RequestFixture):
    request_fixture.config["planned_books"] = []
    request_fixture.write()
    with pytest.raises(OnboardingError, match="planned_books"):
        request_fixture.run()


def test_missing_draft_source_stops(request_fixture: RequestFixture):
    del request_fixture.config["draft_source"]
    request_fixture.write()
    with pytest.raises(OnboardingError, match="draft_source"):
        request_fixture.run()


def test_missing_corpus_stats_stops(request_fixture: RequestFixture):
    request_fixture.write()
    (request_fixture.folder / "corpus-stats.csv").unlink()
    with pytest.raises(OnboardingError, match="corpus-stats.csv"):
        request_fixture.run()


def test_request_folder_is_found_as_given(request_fixture: RequestFixture):
    resolver = RequestFolder(request_fixture.paths.experiments_dir)
    assert resolver.resolve("_OnboardingRequests/TRG_Request") == request_fixture.folder


def test_request_folder_gains_the_request_suffix(request_fixture: RequestFixture):
    resolver = RequestFolder(request_fixture.paths.experiments_dir)
    assert resolver.resolve("_OnboardingRequests/TRG") == request_fixture.folder


def test_missing_request_folder_names_both_tries(request_fixture: RequestFixture):
    resolver = RequestFolder(request_fixture.paths.experiments_dir)
    with pytest.raises(OnboardingError, match="OTHER_Request"):
        resolver.resolve("_OnboardingRequests/OTHER")


def test_planned_book_numbers_become_ids(request_fixture: RequestFixture):
    request_fixture.config["planned_books"] = ["40", 41, "luk"]
    request_fixture.write()
    request = OnboardingRequest(request_fixture.folder, request_fixture.paths.experiments_dir)
    assert request.sign_up().planned_books == ["MAT", "MRK", "LUK"]


def test_unknown_planned_book_stops(request_fixture: RequestFixture):
    request_fixture.config["planned_books"] = ["XYZ"]
    request_fixture.write()
    request = OnboardingRequest(request_fixture.folder, request_fixture.paths.experiments_dir)
    with pytest.raises(OnboardingError, match="XYZ"):
        request.sign_up()


def test_null_bt_project_means_no_back_translation(request_fixture: RequestFixture):
    request_fixture.config["bt_project"] = None
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    assert source_names(planned) == [["SRCB"], ["SRCA"]]


# Sources


def test_draft_source_failing_thresholds_uses_the_two_best_aligned(request_fixture: RequestFixture):
    request_fixture.stats["en-SRCB"] = (500, 0.35, "Latn")
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    assert source_names(planned) == [["SRCC"], ["SRCA"], ["SRCA", "BTX"]]


def test_candidate_in_another_script_is_kept():
    chooser = chooser_with()
    arabic = candidate("SRCA", iso="arb", alignment=0.5, script="Arab")
    experiments = chooser.choose([arabic, candidate("SRCB")], sign_up(), None)
    assert [[s.name for s in e] for e in experiments] == [["SRCB"], ["SRCA"]]


def test_no_passing_source_gives_no_experiments():
    chooser = chooser_with()
    assert chooser.choose([candidate("SRCB", parallel=10)], sign_up(), None) == []


# Back translation


def test_back_translation_missing_from_stats(capsys):
    check = BackTranslationCheck(Thresholds(400, 0.25))
    assert check.usable(sign_up(bt_project="BTX"), [candidate("SRCB")]) is None
    assert "not in corpus-stats.csv" in capsys.readouterr().out


def test_back_translation_below_400_verses(capsys):
    check = BackTranslationCheck(Thresholds(400, 0.25))
    bt = candidate("BTX", parallel=399, alignment=0.6)
    assert check.usable(sign_up(bt_project="BTX"), [bt]) is None
    assert "continuing without it" in capsys.readouterr().out


def test_back_translation_alignment_below_025():
    check = BackTranslationCheck(Thresholds(400, 0.25))
    bt = candidate("BTX", parallel=800, alignment=0.24)
    assert check.usable(sign_up(bt_project="BTX"), [bt]) is None


def test_back_translation_at_the_thresholds_is_used():
    check = BackTranslationCheck(Thresholds(400, 0.25))
    bt = candidate("BTX", parallel=400, alignment=0.25)
    assert check.usable(sign_up(bt_project="BTX"), [bt]) is bt


def test_back_translation_figures_are_printed(capsys):
    check = BackTranslationCheck(Thresholds(400, 0.25))
    bt = candidate("BTX", parallel=800, alignment=0.6)
    bt.src_only = 50
    check.usable(sign_up(bt_project="BTX"), [bt])
    assert "alignment 0.6000, 850 verses, 800 in parallel with the target, 50 not in the target" in (
        capsys.readouterr().out
    )


def test_back_translation_same_as_draft_source_stops():
    check = BackTranslationCheck(Thresholds(400, 0.25))
    with pytest.raises(OnboardingError, match="never a drafting source"):
        check.usable(sign_up(draft_source="SRCB", bt_project="SRCB"), [candidate("SRCB")])


def test_back_translation_in_another_language_gives_no_pairs():
    chooser = chooser_with()
    bt = candidate("BTX", iso="fr", parallel=800, alignment=0.6)
    experiments = chooser.choose([candidate("SRCA", alignment=0.4), candidate("SRCB"), bt], sign_up("SRCB", "BTX"), bt)
    assert [[s.name for s in e] for e in experiments] == [["SRCB"], ["SRCA"]]


def test_two_and_three_letter_isos_are_the_same_language():
    chooser = chooser_with()
    bt = candidate("BTX", iso="eng", parallel=800, alignment=0.6)
    experiments = chooser.choose([candidate("SRCB"), bt], sign_up("SRCB", "BTX"), bt)
    assert [[s.name for s in e] for e in experiments] == [["SRCB"], ["SRCB", "BTX"]]


def test_unusable_back_translation_is_never_a_source(request_fixture: RequestFixture):
    request_fixture.stats["en-BTX"] = (5000, 0.24, "Latn")
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    assert source_names(planned) == [["SRCB"], ["SRCA"]]


# Experiments


def test_four_experiments_when_everything_qualifies(request_fixture: RequestFixture):
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    assert source_names(planned) == [["SRCB"], ["SRCA"], ["SRCB", "BTX"], ["SRCA", "BTX"]]


def test_one_experiment_when_draft_source_is_best_aligned_without_back_translation(request_fixture: RequestFixture):
    request_fixture.config["draft_source"] = "SRCA"
    request_fixture.config["bt_project"] = None
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    assert source_names(planned) == [["SRCA"]]


def test_two_experiments_when_draft_source_is_best_aligned_with_back_translation(request_fixture: RequestFixture):
    request_fixture.config["draft_source"] = "SRCA"
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    assert source_names(planned) == [["SRCA"], ["SRCA", "BTX"]]


def test_configs_written_and_run(request_fixture: RequestFixture):
    request_fixture.write()
    planned, runner, confirmer = request_fixture.run(True, True)
    assert len(confirmer.questions) == 2
    pair = next(e for e in planned if source_names([e]) == [["SRCB", "BTX"]])
    config = yaml.safe_load((pair.folder / "config.yml").read_text(encoding="utf-8"))
    corpus_pair = config["data"]["corpus_pairs"][0]
    assert corpus_pair["src"] == ["en-SRCB", "en-BTX"]
    assert corpus_pair["trg"] == "tpi-TRG"
    assert corpus_pair["corpus_books"] == "MAT;JHN"
    translate = yaml.safe_load((pair.folder / "translate_config.yml").read_text(encoding="utf-8"))["translate"]
    assert translate == [{"books": "MRK;LUK", "src_project": "SRCB", "checkpoint": 5000}]
    assert len(runner.runs) == 4 and all(translate for _, translate in runner.runs)


def test_declining_creation_writes_nothing(request_fixture: RequestFixture):
    request_fixture.write()
    request_fixture.run(False)
    assert not any(p.name == "config.yml" for p in request_fixture.paths.experiments_dir.rglob("*"))


def test_declining_the_run_creates_but_does_not_run(request_fixture: RequestFixture):
    request_fixture.write()
    planned, runner, _ = request_fixture.run(True, False)
    assert all((e.folder / "config.yml").is_file() for e in planned)
    assert runner.runs == []


def test_rerun_creates_folders_with_the_next_index(request_fixture: RequestFixture):
    request_fixture.write()
    first, _, _ = request_fixture.run(True, False)
    second, _, _ = request_fixture.run(True, False)
    assert [e.folder.name for e in first] == ["SRCB_tpi_1", "SRCA_tpi_1", "SRCB_BTX_tpi_1", "SRCA_BTX_tpi_1"]
    assert [e.folder.name for e in second] == ["SRCB_tpi_2", "SRCA_tpi_2", "SRCB_BTX_tpi_2", "SRCA_BTX_tpi_2"]
    assert all((e.folder / "config.yml").is_file() for e in first + second)


def test_next_index_follows_the_highest_existing_folder(request_fixture: RequestFixture):
    request_fixture.config["bt_project"] = None
    request_fixture.write()
    lang_dir = request_fixture.paths.experiments_dir / "PNG" / "Tok_Pisin"
    (lang_dir / "SRCB_tpi_3").mkdir(parents=True)
    planned, _, _ = request_fixture.run(False)
    assert planned[0].folder == lang_dir / "SRCB_tpi_4"


# Books to draft


def test_two_planned_books_drafted_in_ordered_list_order():
    chooser = DraftBookChooser(["GEN", "MRK"], set(), KnownBooks({"SRC": ALL_BOOKS}))
    assert chooser.choose("SRC", {"MAT"}) == ["MRK", "GEN"]


def test_planned_training_book_is_skipped():
    chooser = DraftBookChooser(["MAT", "MRK", "LUK"], set(), KnownBooks({"SRC": ALL_BOOKS}))
    assert chooser.choose("SRC", {"MAT"}) == ["MRK", "LUK"]


def test_next_planned_book_replaces_one_the_source_lacks():
    chooser = DraftBookChooser(["MRK", "LUK", "JHN"], set(), KnownBooks({"SRC": ["LUK", "JHN"]}))
    assert chooser.choose("SRC", set()) == ["LUK", "JHN"]


def test_ordered_list_supplies_a_replacement_when_no_planned_book_is_left():
    chooser = DraftBookChooser(["MRK", "LUK"], set(), KnownBooks({"SRC": ["MAT", "LUK"]}))
    assert chooser.choose("SRC", set()) == ["MAT", "LUK"]


def test_every_planned_book_used_for_training_drafts_the_first_eligible_books():
    chooser = DraftBookChooser(["MAT"], set(), KnownBooks({"SRC": ALL_BOOKS}))
    assert chooser.choose("SRC", {"MAT"}) == ["MRK", "LUK"]


def test_single_planned_book_is_drafted_with_the_next_book_in_the_list():
    chooser = DraftBookChooser(["JHN"], set(), KnownBooks({"SRC": ALL_BOOKS}))
    assert chooser.choose("SRC", {"MAT"}) == ["MRK", "JHN"]


def test_ot_only_planned_books_skip_the_nt():
    chooser = DraftBookChooser(["PSA"], set(), KnownBooks({"SRC": ALL_BOOKS + ["PSA"]}))
    assert chooser.choose("SRC", set()) == ["RUT", "PSA"]


def test_ot_only_planned_books_never_fall_back_to_the_nt():
    chooser = DraftBookChooser(["RUT"], set(), KnownBooks({"SRC": ["RUT", "MAT"]}))
    assert chooser.choose("SRC", {"RUT"}) == []


def test_planned_books_in_both_testaments_draft_in_list_order():
    chooser = DraftBookChooser(["PSA", "GEN", "LUK"], set(), KnownBooks({"SRC": ALL_BOOKS + ["PSA"]}))
    assert chooser.choose("SRC", set()) == ["LUK", "GEN"]


def test_no_eligible_book_drafts_nothing():
    chooser = DraftBookChooser(["MRK"], set(), KnownBooks({"SRC": []}))
    assert chooser.choose("SRC", set()) == []


def test_deuterocanon_skipped_when_target_has_none():
    chooser = DraftBookChooser(["TOB"], set(), KnownBooks({"SRC": ["TOB"]}))
    assert chooser.choose("SRC", set()) == []


def test_deuterocanon_considered_when_target_has_one():
    chooser = DraftBookChooser(["TOB"], {"JDT"}, KnownBooks({"SRC": ["TOB"]}))
    assert chooser.choose("SRC", set()) == ["TOB"]


def test_no_eligible_book_runs_without_translate(request_fixture: RequestFixture):
    request_fixture.projects = {}
    request_fixture.config["bt_project"] = None
    request_fixture.write()
    planned, runner, _ = request_fixture.run(True, True)
    assert all(not (e.folder / "translate_config.yml").exists() for e in planned)
    assert runner.runs and all(not translate for _, translate in runner.runs)
    assert "--translate" not in runner.arguments(False)


def test_experiments_with_different_sources_draft_different_books(request_fixture: RequestFixture):
    request_fixture.projects["SRCA"] = ["MAT", "LUK", "JHN"]
    request_fixture.config["bt_project"] = None
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    books = {e.sources[0].name: e.translate_config["translate"][0]["books"] for e in planned}
    assert books == {"SRCB": "MRK;LUK", "SRCA": "LUK"}


def test_books_to_draft_never_include_a_training_book(request_fixture: RequestFixture):
    request_fixture.config["planned_books"] = ["40", "41"]
    request_fixture.write()
    planned, _, _ = request_fixture.run(False)
    for experiment in planned:
        drafted = experiment.translate_config["translate"][0]["books"].split(";")
        assert "MAT" not in drafted
        assert experiment.config["data"]["corpus_pairs"][0]["corpus_books"] == "MAT;JHN"


# Iso clash


def test_iso_clash_copies_the_target_extract(request_fixture: RequestFixture):
    request_fixture.stats["tpi-SRCD"] = request_fixture.stats.pop("fr-SRCC")
    request_fixture.config["draft_source"] = "SRCD"
    request_fixture.projects["SRCD"] = ALL_BOOKS
    request_fixture.write()
    terms = request_fixture.paths.terms_dir / "tpi-TRG-Major-renderings.txt"
    terms.write_text("terms\n", encoding="utf-8")
    planned, _, _ = request_fixture.run(True, False)
    copies = [p for p in request_fixture.paths.scripture_dir.glob("*-TRG.txt") if p.name != "tpi-TRG.txt"]
    assert len(copies) == 1
    synthetic = copies[0].name.split("-")[0]
    assert all(e.config["data"]["corpus_pairs"][0]["trg"] == f"{synthetic}-TRG" for e in planned)
    assert (request_fixture.paths.terms_dir / f"{synthetic}-TRG-Major-renderings.txt").is_file()


def test_iso_clash_reuses_an_existing_copy(request_fixture: RequestFixture):
    request_fixture.write()
    copy = request_fixture.paths.scripture_dir / "tai-TRG.txt"
    copy.write_text("copy\n", encoding="utf-8")
    old = time.time() - 100
    os.utime(request_fixture.paths.scripture_dir / "tpi-TRG.txt", (old, old))
    planned, _, _ = request_fixture.run(True, False)
    assert copy.read_text(encoding="utf-8") == "copy\n"
    assert planned[0].config["data"]["corpus_pairs"][0]["trg"] == "tai-TRG"


def test_outdated_copy_renamed_to_1_then_2(request_fixture: RequestFixture):
    request_fixture.write()
    scripture = request_fixture.paths.scripture_dir
    copy = scripture / "tai-TRG.txt"
    for expected in ("tai-TRG_1.txt", "tai-TRG_2.txt"):
        copy.write_text("old copy\n", encoding="utf-8")
        old = time.time() - 100
        os.utime(copy, (old, old))
        request_fixture.run(True, False)
        assert (scripture / expected).read_text(encoding="utf-8") == "old copy\n"
        assert copy.read_text(encoding="utf-8") == "target\n"


# Close sources


def test_same_language_source_replaces_a_close_best_aligned_source_in_another_language():
    sources = [
        candidate("BEST", iso="en", alignment=0.2561),
        candidate("FRNEW", iso="fra", alignment=0.2507),
        candidate("FRTEAM", iso="fr", alignment=0.2485),
    ]
    assert chosen_names(chooser_with().choose(sources, sign_up("FRTEAM"), None)) == [["FRTEAM"], ["FRNEW"]]


def test_same_language_source_that_is_not_close_does_not_replace_the_best():
    sources = [
        candidate("BEST", iso="en", alignment=0.30),
        candidate("FRNEW", iso="fr", alignment=0.26),
        candidate("FRTEAM", iso="fr", alignment=0.25),
    ]
    assert chosen_names(chooser_with().choose(sources, sign_up("FRTEAM"), None)) == [["FRTEAM"], ["BEST"]]


def test_same_language_source_below_draft_source_does_not_replace_the_best():
    sources = [
        candidate("BEST", iso="en", alignment=0.26),
        candidate("FRTEAM", iso="fr", alignment=0.25),
        candidate("FRLOW", iso="fr", alignment=0.24),
    ]
    assert chosen_names(chooser_with().choose(sources, sign_up("FRTEAM"), None)) == [["FRTEAM"], ["BEST"]]


def test_close_later_edition_replaces_the_best():
    sources = [candidate("OLD", alignment=0.26), candidate("NEW", alignment=0.25), candidate("TEAM", alignment=0.22)]
    chooser = chooser_with(preferred=[["NEW", "OLD"]])
    assert chosen_names(chooser.choose(sources, sign_up("TEAM"), None)) == [["TEAM"], ["NEW"]]


def test_later_edition_that_is_not_close_does_not_replace_the_best():
    sources = [candidate("OLD", alignment=0.30), candidate("NEW", alignment=0.25), candidate("TEAM", alignment=0.22)]
    chooser = chooser_with(preferred=[["NEW", "OLD"]])
    assert chosen_names(chooser.choose(sources, sign_up("TEAM"), None)) == [["TEAM"], ["OLD"]]


def test_the_latest_close_edition_is_chosen():
    sources = [
        candidate("OLDEST", alignment=0.26),
        candidate("MIDDLE", alignment=0.255),
        candidate("NEWEST", alignment=0.25),
        candidate("TEAM", alignment=0.21),
    ]
    chooser = chooser_with(preferred=[["NEWEST", "MIDDLE", "OLDEST"]])
    assert chosen_names(chooser.choose(sources, sign_up("TEAM"), None)) == [["TEAM"], ["NEWEST"]]


def test_later_edition_step_never_chooses_draft_source():
    sources = [candidate("OLD", alignment=0.26), candidate("NEW", alignment=0.25)]
    chooser = chooser_with(preferred=[["NEW", "OLD"]])
    assert chosen_names(chooser.choose(sources, sign_up("NEW"), None)) == [["NEW"], ["OLD"]]


def test_avoided_source_is_skipped():
    sources = [
        candidate("ESVUS16", alignment=0.30),
        candidate("OTHER", alignment=0.28),
        candidate("TEAM", alignment=0.22),
    ]
    chooser = chooser_with(avoid=["ESV*"])
    assert chosen_names(chooser.choose(sources, sign_up("TEAM"), None)) == [["TEAM"], ["OTHER"]]


def test_avoided_source_is_used_when_it_is_the_only_candidate():
    chooser = chooser_with(avoid=["ESV*"])
    experiments = chooser.choose([candidate("ESVUS16")], sign_up("MISSING"), None)
    assert chosen_names(experiments) == [["ESVUS16"]]


def test_avoided_draft_source_is_kept(capsys):
    sources = [candidate("BEST", alignment=0.4), candidate("ESVUS16", alignment=0.3)]
    chooser = chooser_with(avoid=["ESV*"])
    assert chosen_names(chooser.choose(sources, sign_up("ESVUS16"), None)) == [["ESVUS16"], ["BEST"]]
    assert "keeping it as the team's choice" in capsys.readouterr().out


def test_later_edition_that_aligns_better_replaces_draft_source(capsys):
    sources = [candidate("BEST", alignment=0.4), candidate("NEW", alignment=0.3), candidate("OLD", alignment=0.29)]
    chooser = chooser_with(preferred=[["NEW", "OLD"]])
    assert chosen_names(chooser.choose(sources, sign_up("OLD"), None)) == [["NEW"], ["BEST"]]
    assert "instead of draft_source OLD" in capsys.readouterr().out


def test_later_edition_that_aligns_worse_does_not_replace_draft_source():
    sources = [candidate("BEST", alignment=0.4), candidate("NEW", alignment=0.28), candidate("OLD", alignment=0.29)]
    chooser = chooser_with(preferred=[["NEW", "OLD"]])
    assert chosen_names(chooser.choose(sources, sign_up("OLD"), None))[0] == ["OLD"]


def test_with_draft_source_failing_both_sources_follow_the_preferences():
    sources = [
        candidate("OLD", alignment=0.30),
        candidate("NEW", alignment=0.29),
        candidate("THIRD", alignment=0.28),
        candidate("TEAM", parallel=10),
    ]
    chooser = chooser_with(preferred=[["NEW", "OLD"]])
    assert chosen_names(chooser.choose(sources, sign_up("TEAM"), None)) == [["OLD"], ["NEW"]]


def test_preferred_sources_file_is_read():
    preferences = SourcePreferencesFile(ASSETS_DIR).load()
    assert preferences.later_editions("PDV") == ["PDV2017"]
    assert preferences.later_editions("BSB") == ["NIV11R", "NIV84", "NLT07"]
    assert preferences.is_avoided("ESVUS16")
    assert not preferences.is_avoided("NIV11R")


def test_missing_preferred_sources_file_means_no_preferences(tmp_path: Path):
    preferences = SourcePreferencesFile(tmp_path).load()
    assert preferences.later_editions("PDV") == []
    assert not preferences.is_avoided("ESVUS16")
