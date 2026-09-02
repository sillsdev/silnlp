import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from machine.scripture import book_id_to_number

from silnlp.common.create_onboarding_experiments import (
    EXPERIMENT_ARGS,
    NT_CANON,
    TOP_EXPERIMENTS,
    BookCoverage,
    Candidate,
    Experiment,
    book_mark,
    extract_book_counts,
    find_existing,
    folder_name,
    load_language_entries,
    load_name_overrides,
    load_vref_books,
    load_vref_chapters,
    nllb_tag,
    old_name_folder_warning,
    overlapping_books,
    parse_book_list,
    parse_corpus_stats,
    parse_log,
    resolve_corpus_books,
    resolve_request_dir,
    run,
    select_experiments,
    submit_experiments,
    synthesize_trg_iso,
)
from silnlp.common.iso_info import NLLB_TAG_FROM_ISO

ASSETS_DIR = Path(__file__).parent.parent / "silnlp" / "assets"
SAMPLE_LOG_PATH = Path(__file__).parent / "data" / "create_onboarding_experiments" / "onboarding.log"

BOOKS = ["GEN", "EXO", "MAT", "MRK"]


def make_verse_counts(path: Path) -> None:
    df = pd.DataFrame(
        {
            "file": ["complete", "sdl-A33_2026_07_02", "en-NIV11R", "hi-HINCLBSI", "arb-a55_2026_07_02"],
            "GEN": [1533, 0, 1533, 1533, 0],
            "EXO": [1213, 0, 1213, 1213, 0],
            "MAT": [1071, 1071, 1071, 1071, 0],
            "MRK": [678, 678, 678, 678, 179],
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture
def request_dir(tmp_path: Path) -> Path:
    request = tmp_path / "_OnboardingRequests" / "A33_2026_07_02_Request"
    request.mkdir(parents=True)
    (request / "onboarding.log").write_text(SAMPLE_LOG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    make_verse_counts(request / "verse_counts.csv")
    return request


@pytest.fixture
def select_all(monkeypatch):
    """Answer the selection prompt with 'all', the copy confirmation with 'y', keep translate sources."""

    def answer(prompt: str) -> str:
        if "Copy" in prompt:
            return "y"
        if "different project" in prompt:
            return ""
        return "all"

    monkeypatch.setattr("builtins.input", answer)


def make_paratext_project(projects_dir: Path, name: str, books: list) -> None:
    """Create a minimal Paratext project folder with a Settings.xml and the given book files."""
    from machine.corpora import FileParatextProjectSettingsParser

    project_dir = projects_dir / name
    project_dir.mkdir(parents=True)
    (project_dir / "Settings.xml").write_text(
        "<ScriptureText>\n"
        "  <Versification>4</Versification>\n"
        "  <LanguageIsoCode>en:::</LanguageIsoCode>\n"
        "  <Language>English</Language>\n"
        "  <Encoding>65001</Encoding>\n"
        f"  <Name>{name}</Name>\n"
        f'  <Naming PrePart="" BookNameForm="41MAT" PostPart="{name}.SFM" />\n'
        "</ScriptureText>\n",
        encoding="utf-8",
    )
    settings = FileParatextProjectSettingsParser(project_dir).parse()
    for book in books:
        (project_dir / settings.get_book_file_name(book)).write_text(f"\\id {book}\n", encoding="utf-8")


def test_parse_log(request_dir: Path):
    main, candidates = parse_log(request_dir / "onboarding.log")
    assert main.name == "A33_2026_07_02"
    assert main.stem == "sdl-A33_2026_07_02"
    assert main.iso == "sdl"
    assert main.verses == 5084
    assert main.script == "Arab"

    by_name = {c.name: c for c in candidates}
    assert set(by_name) == {"NIV11R", "HINCLBSI", "a55_2026_07_02"}
    # NIV11R has no "Extracted corpus file" line; its stem comes from the alignment line.
    niv = by_name["NIV11R"]
    assert (niv.stem, niv.iso, niv.count, niv.parallel, niv.alignment, niv.script) == (
        "en-NIV11R",
        "en",
        31096,
        5070,
        0.3388,
        "Latn",
    )
    assert by_name["HINCLBSI"].script == "Deva"
    assert by_name["a55_2026_07_02"].parallel == 1


def test_folder_name():
    assert folder_name("Russian Federation") == "Russian_Federation"
    assert folder_name("Arabic, Standard") == "Arabic_Standard"
    assert folder_name("Mari-Hill") == "Mari_Hill"
    assert folder_name("Saudi Arabian Sign Language") == "Saudi_Arabian_Sign_Language"
    # keep_case preserves acronyms/interior caps that capitalize() would mangle
    assert folder_name("DR Congo") == "Dr_Congo"
    assert folder_name("DR Congo", keep_case=True) == "DR_Congo"


def test_name_overrides_keys_exist_in_language_families():
    # Guards against a refreshed languageFamilies.json orphaning an override key.
    entries = load_language_entries(ASSETS_DIR)
    overrides = load_name_overrides(ASSETS_DIR)
    countries = {entry["langCountry"] for entry in entries}
    languages = {entry["language"] for entry in entries}
    assert not [key for key in overrides["countries"] if key not in countries]
    assert not [key for key in overrides["languages"] if key not in languages]


def test_name_override_applied():
    overrides = load_name_overrides(ASSETS_DIR)
    assert overrides["countries"]["Tanzania, United Republic of"] == "Tanzania"
    assert folder_name(overrides["countries"]["Russian Federation"], keep_case=True) == "Russia"
    assert folder_name(overrides["countries"]["Virgin Islands, U.S."], keep_case=True) == "US_Virgin_Islands"
    assert folder_name(overrides["countries"]["Congo, The Democratic Republic of the"], keep_case=True) == "DR_Congo"
    assert folder_name(overrides["languages"]["German, Standard"], keep_case=True) == "German"
    assert folder_name(overrides["languages"]["German, Swiss"], keep_case=True) == "Swiss_German"
    # an unmapped country is not in the override map (passes through unchanged)
    assert "Peru" not in overrides["countries"]


def test_name_override_values_non_blank():
    # The shipped asset must not carry a blank value. At runtime a blank override is ignored
    # (falls back to the official name — see run()), but this keeps the seed data clean.
    overrides = load_name_overrides(ASSETS_DIR)
    for group in ("countries", "languages"):
        for key, value in overrides[group].items():
            assert value and value.strip(), f"blank override value for {group!r} key {key!r}"


def test_old_name_folder_warning_country_rename(tmp_path: Path):
    # A renamed country warns at the country level (covers every language under it).
    (tmp_path / "Russian_Federation").mkdir()
    warning = old_name_folder_warning(tmp_path, "Russian Federation", "Abaza", "Russia", "Abaza")
    assert warning is not None
    assert str(tmp_path / "Russian_Federation") in warning
    assert str(tmp_path / "Russia") in warning


def test_old_name_folder_warning_language_rename(tmp_path: Path):
    # A language-only rename (country unchanged) must still warn on the old language folder,
    # otherwise find_existing never sees the prior experiments and creates duplicates.
    (tmp_path / "Germany" / "German_Standard").mkdir(parents=True)
    warning = old_name_folder_warning(tmp_path, "Germany", "German, Standard", "Germany", "German")
    assert warning is not None
    assert str(tmp_path / "Germany" / "German_Standard") in warning
    assert str(tmp_path / "Germany" / "German") in warning


def test_old_name_folder_warning_language_rename_under_overridden_country(tmp_path: Path):
    # Country already overridden (Russia) so prior runs live under the override folder; a later
    # language rename must be detected under the *current* country segment, not the official one.
    (tmp_path / "Russia" / "German_Standard").mkdir(parents=True)
    warning = old_name_folder_warning(tmp_path, "Russian Federation", "German, Standard", "Russia", "German")
    assert warning is not None
    assert str(tmp_path / "Russia" / "German_Standard") in warning
    assert str(tmp_path / "Russia" / "German") in warning


def test_old_name_folder_warning_none(tmp_path: Path):
    # No old folder present -> no warning.
    assert old_name_folder_warning(tmp_path, "Georgia", "Abkhaz", "Georgia", "Abkhaz") is None
    # Nothing remapped (old == new) -> no warning even if the folder exists.
    (tmp_path / "Peru" / "Quechua").mkdir(parents=True)
    assert old_name_folder_warning(tmp_path, "Peru", "Quechua", "Peru", "Quechua") is None


def test_nllb_tag():
    assert nllb_tag("en", "Latn") == "eng_Latn"  # 2-letter code resolved to the NLLB tag
    assert nllb_tag("hi", "Deva") == "hin_Deva"
    assert nllb_tag("sdl", "Arab") == "sdl_Arab"  # not in NLLB: iso3 + script from the log


def test_parse_book_list():
    assert parse_book_list("GEN;RUT") == "GEN;RUT"
    assert parse_book_list("GEN,RUT") == "GEN,RUT"  # already valid downstream syntax, kept verbatim
    assert parse_book_list("GEN RUT") == "GEN;RUT"
    assert parse_book_list("gen rut") == "GEN;RUT"
    assert parse_book_list(" GEN, RUT ;JON ") == "GEN;RUT;JON"  # internal spaces force normalisation
    assert parse_book_list("GEN") == "GEN"
    # Full get_chapters syntax is preserved verbatim (ranges, chapters, subtraction) on both
    # --translate-books and --training-books, so complex selections round-trip into the config.
    assert parse_book_list("MAT 1-4") == "MAT 1-4"
    assert parse_book_list("mat 1-4") == "MAT 1-4"
    assert parse_book_list("NT;-REV") == "NT;-REV"
    assert parse_book_list("GEN-DEU") == "GEN-DEU"
    assert parse_book_list("GEN 1-10,12,15;NT") == "GEN 1-10,12,15;NT"  # chapters + comma list + testament
    assert parse_book_list("NT;-HEB;-REV") == "NT;-HEB;-REV"  # multiple subtractions
    assert parse_book_list("GEN-PSA;NT") == "GEN-PSA;NT"  # book range + testament
    # Invalid book lists are rejected at the CLI instead of failing at preprocess time.
    with pytest.raises(ValueError, match="not a valid book list"):
        parse_book_list("GEN;MTT")
    with pytest.raises(ValueError, match="not a valid book list"):
        parse_book_list("")


def test_resolve_corpus_books_verbatim():
    assert resolve_corpus_books("GEN;EXO;NT", [], None) == ("GEN;EXO;NT", [])


def test_resolve_corpus_books_complete(request_dir: Path):
    df = pd.read_csv(request_dir / "verse_counts.csv", index_col="file")
    books, removed = resolve_corpus_books("complete", ["en-NIV11R", "sdl-A33_2026_07_02"], df)
    assert (books, removed) == ("MAT;MRK", [])
    # arb-a55 has partial MRK (179 < 98% of 678), so nothing qualifies with it as a source.
    assert resolve_corpus_books("complete", ["arb-a55_2026_07_02", "sdl-A33_2026_07_02"], df) == ("", [])


def test_resolve_corpus_books_nt_compaction():
    stems = ["en-NIV11R", "sdl-TRG"]
    df = pd.DataFrame(100, index=["complete"] + stems, columns=NT_CANON)
    df.index.name = "file"
    assert resolve_corpus_books("complete", stems, df) == ("NT", [])


def test_resolve_corpus_books_excludes_translate_books():
    mat = {book_id_to_number("MAT")}
    # Explicit lists are kept verbatim, with subtraction selections appended for excluded books.
    assert resolve_corpus_books("GEN;EXO;MAT", [], None, exclude=mat) == ("GEN;EXO;MAT;-MAT", ["MAT"])
    assert resolve_corpus_books("NT", [], None, exclude=mat) == ("NT;-MAT", ["MAT"])
    assert resolve_corpus_books("OT;NT", [], None, exclude=mat) == ("OT;NT;-MAT", ["MAT"])
    # Books hidden inside ranges are excluded too.
    exo = {book_id_to_number("EXO")}
    assert resolve_corpus_books("GEN-DEU", [], None, exclude=exo) == ("GEN-DEU;-EXO", ["EXO"])
    # Chapter-level selections subtract exactly the selected chapters.
    assert resolve_corpus_books("GEN;MAT 1-4", [], None, exclude=mat) == ("GEN;MAT 1-4;-MAT1,2,3,4", ["MAT"])
    # Excluding everything empties the list.
    assert resolve_corpus_books("MAT", [], None, exclude=mat) == ("", ["MAT"])
    # Books not in the list are not reported as removed.
    assert resolve_corpus_books("GEN;EXO", [], None, exclude=mat) == ("GEN;EXO", [])
    # Case-insensitive: exclusion works regardless of how the CLI value was cased (numbers, not strings).
    books, removed = resolve_corpus_books("complete", ["en-NIV11R", "sdl-TRG"], _nt_counts(), exclude=mat)
    assert removed == ["MAT"]


def _nt_counts() -> pd.DataFrame:
    df = pd.DataFrame(100, index=["complete", "en-NIV11R", "sdl-TRG"], columns=NT_CANON)
    df.index.name = "file"
    return df


def test_resolve_corpus_books_complete_excludes_translate_books(request_dir: Path):
    df = pd.read_csv(request_dir / "verse_counts.csv", index_col="file")
    books, removed = resolve_corpus_books(
        "complete", ["en-NIV11R", "sdl-A33_2026_07_02"], df, exclude={book_id_to_number("MAT")}
    )
    assert (books, removed) == ("MRK", ["MAT"])


def test_synthesize_trg_iso():
    real_isos = {entry["isoCode"] for entry in load_language_entries(ASSETS_DIR)}
    code = synthesize_trg_iso("sdl", real_isos)
    assert len(code) == 3 and code[0] == "s" and code != "sdl"
    assert code not in real_isos
    assert code not in NLLB_TAG_FROM_ISO
    # Deterministic: same input gives the same code on re-runs.
    assert synthesize_trg_iso("sdl", real_isos) == code


def test_find_existing(tmp_path: Path):
    config = {"model": "m", "data": {"seed": 111}}
    lang_dir = tmp_path / "Lang"
    existing, index = find_existing(lang_dir, "NIV11R_sdl", config)
    assert existing is None and index == 1

    folder = lang_dir / "NIV11R_sdl_2"
    folder.mkdir(parents=True)
    (folder / "config.yml").write_text(yaml.dump({"model": "other"}), encoding="utf-8")
    existing, index = find_existing(lang_dir, "NIV11R_sdl", config)
    assert existing is None and index == 3

    (folder / "config.yml").write_text(yaml.dump(config), encoding="utf-8")
    existing, _ = find_existing(lang_dir, "NIV11R_sdl", config)
    assert existing == folder


def make_corpus_stats(path: Path, main_is_trg: bool = True) -> None:
    """Write a corpus-stats.csv in the analyze format, matching the fixture log's stats."""
    main_stem, main_script = "sdl-A33_2026_07_02", "Arab"
    refs = [
        ("en-NIV11R", 31096, 5070, 0.3388, "Latn"),
        ("hi-HINCLBSI", 30998, 5068, 0.2605, "Deva"),
        ("arb-a55_2026_07_02", 5258, 1, 0.4000, "Arab"),
    ]
    records = []
    for ref_stem, count, parallel, score, script in refs:
        main_entry = {"project": main_stem, "script": main_script}
        ref_entry = {"project": ref_stem, "script": script}
        src, trg = (ref_entry, main_entry) if main_is_trg else (main_entry, ref_entry)
        records.append(
            {
                "src_project": src["project"],
                "trg_project": trg["project"],
                "count": count,
                "src_only": 0,
                "trg_only": 0,
                "parallel": parallel,
                "align_score": score,
                "filtered_count": 0,
                "filtered_align_score": score,
                "src_script": src["script"],
                "src_script_in_model": True,
                "trg_script": trg["script"],
                "trg_script_in_model": True,
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def test_parse_corpus_stats(tmp_path: Path):
    # Both directions parse to the same result: main as trg (alignment run) or src (analyze run).
    for main_is_trg in (True, False):
        stats_path = tmp_path / f"corpus-stats-{main_is_trg}.csv"
        make_corpus_stats(stats_path, main_is_trg=main_is_trg)
        main, candidates = parse_corpus_stats(stats_path)
        assert (main.name, main.stem, main.iso, main.script) == (
            "A33_2026_07_02",
            "sdl-A33_2026_07_02",
            "sdl",
            "Arab",
        )
        by_name = {c.name: c for c in candidates}
        assert set(by_name) == {"NIV11R", "HINCLBSI", "a55_2026_07_02"}
        niv = by_name["NIV11R"]
        assert (niv.stem, niv.iso, niv.count, niv.parallel, niv.alignment, niv.script) == (
            "en-NIV11R",
            "en",
            31096,
            5070,
            0.3388,
            "Latn",
        )
        assert by_name["HINCLBSI"].script == "Deva"

    # Neither column constant -> the main project cannot be determined.
    ambiguous = pd.DataFrame(
        {
            "src_project": ["en-NIV11R", "hi-HINCLBSI"],
            "trg_project": ["sdl-A", "sdl-B"],
            "count": [1, 1],
            "parallel": [1, 1],
            "align_score": [0.5, 0.5],
            "src_script": ["Latn", "Deva"],
            "trg_script": ["Arab", "Arab"],
        }
    )
    ambiguous.to_csv(tmp_path / "ambiguous.csv", index=False)
    with pytest.raises(ValueError, match="Cannot determine the main project"):
        parse_corpus_stats(tmp_path / "ambiguous.csv")


def test_parse_corpus_stats_target(tmp_path: Path):
    stats_path = tmp_path / "corpus-stats.csv"
    make_corpus_stats(stats_path)

    # The target may be given as an iso code, a project name, or a full stem.
    for target in ("sdl", "A33_2026_07_02", "sdl-A33_2026_07_02", "SDL"):
        main, candidates = parse_corpus_stats(stats_path, target=target)
        assert main.stem == "sdl-A33_2026_07_02"
        assert {c.name for c in candidates} == {"NIV11R", "HINCLBSI", "a55_2026_07_02"}

    # The target may sit on either side per row, and rows not involving it are ignored.
    mixed_direction = pd.DataFrame(
        {
            "src_project": ["en-NIV11R", "sdl-A33_2026_07_02", "en-NIV11R"],
            "trg_project": ["sdl-A33_2026_07_02", "hi-HINCLBSI", "fr-BDS"],
            "count": [10, 20, 30],
            "parallel": [5, 6, 7],
            "align_score": [0.5, 0.4, 0.3],
            "src_script": ["Latn", "Arab", "Latn"],
            "trg_script": ["Arab", "Deva", "Latn"],
        }
    )
    mixed_direction.to_csv(tmp_path / "mixed.csv", index=False)
    main, candidates = parse_corpus_stats(tmp_path / "mixed.csv", target="sdl")
    assert main.stem == "sdl-A33_2026_07_02"
    by_name = {c.name: c for c in candidates}
    assert set(by_name) == {"NIV11R", "HINCLBSI"}  # the NIV11R/BDS row is ignored
    assert by_name["HINCLBSI"].alignment == 0.4

    # A target matching two different projects (shared iso) is an error.
    two_projects = mixed_direction.copy()
    two_projects.loc[2, "trg_project"] = "sdl-OtherProject"
    two_projects.to_csv(tmp_path / "two.csv", index=False)
    with pytest.raises(ValueError, match="matches more than one project"):
        parse_corpus_stats(tmp_path / "two.csv", target="sdl")

    # A target matching nothing is an error.
    with pytest.raises(ValueError, match="does not match any project"):
        parse_corpus_stats(stats_path, target="xyz")

    # A single row is ambiguous without a target...
    single = mixed_direction.iloc[[0]]
    single.to_csv(tmp_path / "single.csv", index=False)
    with pytest.raises(ValueError, match="specify it with --target"):
        parse_corpus_stats(tmp_path / "single.csv")
    # ...and fine with one.
    main, candidates = parse_corpus_stats(tmp_path / "single.csv", target="sdl")
    assert main.stem == "sdl-A33_2026_07_02" and [c.name for c in candidates] == ["NIV11R"]


def test_parse_corpus_stats_skips_incomplete_rows(tmp_path: Path):
    # analyze.py writes empty count cells when original extracts are missing; such rows
    # must not crash the parse, only drop the affected candidate.
    stats_path = tmp_path / "corpus-stats.csv"
    stats_path.write_text(
        "src_project,trg_project,count,src_only,trg_only,parallel,align_score,filtered_count,"
        "filtered_align_score,src_script,src_script_in_model,trg_script,trg_script_in_model\n"
        "sdl-MAIN,en-REF1,10,0,0,5,0.5,0,0.5,Arab,True,Latn,True\n"
        "sdl-MAIN,en-REF2,,,,,,,,Arab,True,Latn,True\n",
        encoding="utf-8",
    )
    main, candidates = parse_corpus_stats(stats_path)
    assert main.stem == "sdl-MAIN"
    assert [c.name for c in candidates] == ["REF1"]


def test_clean_script():
    from silnlp.common.create_onboarding_experiments import clean_script

    assert clean_script("Arab") == "Arab"
    assert clean_script("  Latn ") == "Latn"
    assert clean_script(None) is None
    assert clean_script("") is None
    assert clean_script("nan") is None  # str(pandas NaN)
    assert clean_script("None") is None  # predict_script_code's empty-text result
    assert clean_script(float("nan")) is None


def test_book_mark():
    complete = {"MAT": 100, "MRK": 100}
    assert book_mark({"MAT": 100}, "MAT", complete) == "✓"
    assert book_mark({"MAT": 98}, "MAT", complete) == "✓"  # 98% boundary is inclusive
    assert book_mark({"MAT": 97}, "MAT", complete) == "~"  # partial
    assert book_mark({"MAT": 0}, "MAT", complete) == "X"  # present column, no verses
    assert book_mark({}, "MAT", complete) == "X"
    assert book_mark(None, "MAT", complete) == "X"  # no coverage data at all
    assert book_mark({"XYZ": 5}, "XYZ", complete) == "~"  # no complete count -> at best partial


BOOK_STATS_HEADER = (
    "src_project,trg_project,count,src_only,trg_only,parallel,align_score,filtered_count,"
    "filtered_align_score,src_script,src_script_in_model,trg_script,trg_script_in_model"
)


@pytest.mark.parametrize("main_is_trg", [True, False])
def test_parse_corpus_stats_src_trg_only(tmp_path: Path, main_is_trg: bool):
    # 'source only' verses are the reference's own (drafting potential); the parser must read
    # the right physical column for the row's orientation. The scenario is fixed: the reference
    # has 30 verses of its own, the target 50, and 20 are parallel.
    if main_is_trg:  # reference is the src column, so src_only is the reference's own
        row = "en-REF,arz-NGT,100,30,50,20,0.5,0,0.5,Latn,True,Arab,True"
    else:  # reference is the trg column, so trg_only is the reference's own
        row = "arz-NGT,en-REF,100,50,30,20,0.5,0,0.5,Arab,True,Latn,True"
    stats_path = tmp_path / "corpus-stats.csv"
    stats_path.write_text(f"{BOOK_STATS_HEADER}\n{row}\n", encoding="utf-8")
    _, candidates = parse_corpus_stats(stats_path, target="arz-NGT")
    c = candidates[0]
    assert (c.name, c.parallel, c.src_only, c.trg_only) == ("REF", 20, 30, 50)


def test_parse_log_src_trg_only(tmp_path: Path):
    # In the log the main is the alignment source, so the reference's own verses are logged as
    # 'target only count'. Older logs without these counts fall back to 0.
    log = (
        "Processing onboarding request for main project 'NGT'\n"
        "Extracted corpus file: /x/arz-NGT.txt\n"
        "# of Verses: 100\n"
        "Computing alignment beteween arz-NGT and en-REF using Eflomal\n"
        "Computing alignment beteween arz-NGT and en-OLD using Eflomal\n"
        "NGT -> REF stats - count: 100, source only count: 50 target only count: 30"
        " parallel count: 20 alignment: 0.5, source script: Arab, source script in model: True,"
        " target script: Latn, x\n"
        "NGT -> OLD stats - count: 100, parallel count: 20 alignment: 0.4,"
        " source script: Arab, source script in model: True, target script: Latn, x\n"
    )
    log_path = tmp_path / "onboarding.log"
    log_path.write_text(log, encoding="utf-8")
    _, candidates = parse_log(log_path)
    by_name = {c.name: c for c in candidates}
    assert (by_name["REF"].parallel, by_name["REF"].src_only, by_name["REF"].trg_only) == (20, 30, 50)
    assert (by_name["OLD"].src_only, by_name["OLD"].trg_only) == (0, 0)  # counts absent -> 0


@pytest.mark.parametrize("main_is_trg", [True, False])
@pytest.mark.parametrize("use_target", [True, False])
def test_parse_corpus_stats_main_script_skips_empty_rows(tmp_path: Path, main_is_trg: bool, use_target: bool):
    # A pair with no parallel data has an empty ('None'/'nan') script; when such a row sorts
    # first it must not become the main project's script (that produced '<iso>_nan' configs).
    # pandas reads both 'None' and 'nan' as NaN. The main may sit in either physical column
    # (the headers are always src_project/trg_project), so both orientations are checked, with
    # and without an explicit --target.
    header = (
        "src_project,trg_project,count,src_only,trg_only,parallel,align_score,filtered_count,"
        "filtered_align_score,src_script,src_script_in_model,trg_script,trg_script_in_model"
    )

    def row(other: str, count: str, stats: str, other_script: str, other_in_model: str, main_script: str) -> str:
        main_cell = (f"{count},{stats}", main_script, "True" if main_script == "Arab" else "False")
        # The main project sits in whichever column the caller chose; the other project fills
        # the remaining side. Scripts travel with their own side.
        if main_is_trg:
            return f"{other},arz-NGT,{main_cell[0]},{other_script},{other_in_model},{main_cell[1]},{main_cell[2]}"
        return f"arz-NGT,{other},{main_cell[0]},{main_cell[1]},{main_cell[2]},{other_script},{other_in_model}"

    # Leading no-data row (empty scripts on both sides) then a real aligned row.
    empty_row = row("en-EMPTY", "10337", "0,10337,0,nan,0,nan", "None", "False", "None")
    real_row = row("arb-REF", "10327", "0,5982,4345,0.52,0,0.52", "Arab", "True", "Arab")
    stats_path = tmp_path / "corpus-stats.csv"
    stats_path.write_text("\n".join([header, empty_row, real_row]) + "\n", encoding="utf-8")

    main, candidates = parse_corpus_stats(stats_path, target="arz-NGT" if use_target else None)
    assert main.stem == "arz-NGT"
    assert main.script == "Arab"  # from the row with data, not the leading empty row
    # The empty-data row is dropped as a candidate, the real one is kept.
    assert [c.name for c in candidates] == ["REF"]

    # The built config's lang_codes then use the real script even after a synthetic rename.
    assert nllb_tag("aaj", main.script or "") == "aaj_Arab"


def test_parse_log_main_script_skips_empty_rows(tmp_path: Path):
    # The log path is vulnerable too: analyze logs 'source script: nan' for a no-data pair.
    log = (
        "Processing onboarding request for main project 'NGT'\n"
        "Extracted corpus file: /x/arz-NGT.txt\n"
        "# of Verses: 5000\n"
        "Computing alignment beteween arz-NGT and en-EMPTY using Eflomal\n"
        "Computing alignment beteween arz-NGT and arb-REF using Eflomal\n"
        "NGT -> en-EMPTY stats - count: 100, parallel count: 0 alignment: 0.0,"
        " source script: nan, source script in model: False, target script: nan, x\n"
        "NGT -> arb-REF stats - count: 100, parallel count: 5000 alignment: 0.5,"
        " source script: Arab, source script in model: True, target script: Arab, x\n"
    )
    log_path = tmp_path / "onboarding.log"
    log_path.write_text(log, encoding="utf-8")
    main, candidates = parse_log(log_path)
    assert main.stem == "arz-NGT"
    assert main.script == "Arab"  # not 'nan' from the first stats row


def test_parse_corpus_stats_target_matching_both_sides_of_a_row(tmp_path: Path):
    stats_path = tmp_path / "corpus-stats.csv"
    stats_path.write_text(
        "src_project,trg_project,count,src_only,trg_only,parallel,align_score,filtered_count,"
        "filtered_align_score,src_script,src_script_in_model,trg_script,trg_script_in_model\n"
        "sdl-A33,sdl-BackTrans,10,0,0,5,0.5,0,0.5,Arab,True,Arab,True\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matches both sides"):
        parse_corpus_stats(stats_path, target="sdl")
    # The project name still disambiguates.
    main, _ = parse_corpus_stats(stats_path, target="BackTrans")
    assert main.stem == "sdl-BackTrans"


def test_stem_matches_iso_equivalence():
    from silnlp.common.create_onboarding_experiments import stem_matches

    assert stem_matches("fr-BDS", "fra")  # 3-letter target matches the 2-letter stem prefix
    assert stem_matches("fra-BDS", "fr")
    assert stem_matches("en-NIV11R", "ENG")
    assert not stem_matches("fr-BDS", "deu")
    assert stem_matches("fr-BDS", "bds")  # project-name leg is unaffected


def test_run_falls_back_to_log_when_csv_is_unusable(request_dir: Path, tmp_path: Path, capsys, select_all):
    # A stale/partial CSV that neither fits the log's main project nor auto-detects must
    # not brick a folder that previously worked from onboarding.log.
    alignments = request_dir / "alignments"
    alignments.mkdir()
    pd.DataFrame(
        {
            "src_project": ["en-X", "fr-Y"],
            "trg_project": ["de-P", "es-Q"],  # neither column constant, no A33 stem
            "count": [1, 1],
            "parallel": [1, 1],
            "align_score": [0.5, 0.5],
            "src_script": ["Latn", "Latn"],
            "trg_script": ["Latn", "Latn"],
        }
    ).to_csv(alignments / "corpus-stats.csv", index=False)

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    # Falls back to the log and behaves exactly like a log-only folder.
    assert sorted(e.folder.name for e in experiments) == ["HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1", "NIV11R_sdl_1"]


def test_run_target_flip_uses_derived_location(tmp_path: Path, capsys):
    # When --target flips the direction, the create-next-to-stats rule must not file the
    # experiments under the stats folder's (different) language.
    align_dir = tmp_path / "KSA" / "SignLang" / "Align"
    align_dir.mkdir(parents=True)
    make_corpus_stats(align_dir / "corpus-stats.csv")
    make_verse_counts(align_dir / "verse_counts.csv")
    run(
        request_dir=align_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        target="NIV11R",
        dry_run=True,
    )
    output = capsys.readouterr().out
    assert "Note: --target overrides the analyze run's own target project (sdl-A33_2026_07_02)." in output
    assert "Experiment location: " + str(tmp_path / "KSA" / "SignLang") not in output


def test_run_stats_folder_directly_under_experiments_uses_derived_location(tmp_path: Path, capsys):
    stats_dir = tmp_path / "MyAlignRun"
    stats_dir.mkdir()
    make_corpus_stats(stats_dir / "corpus-stats.csv")
    make_verse_counts(stats_dir / "verse_counts.csv")
    run(
        request_dir=stats_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        dry_run=True,
    )
    output = capsys.readouterr().out
    # Not dumped at the experiments root: the derived Country/Language location is used.
    assert f"Experiment location: {tmp_path / 'Saudi_Arabia' / 'Saudi_Arabian_Sign_Language'}" in output


def test_run_applies_country_override_to_location(tmp_path: Path, capsys, monkeypatch):
    # End-to-end wiring: an overridden country must reach both the folder path and the report.
    import silnlp.common.create_onboarding_experiments as coe

    monkeypatch.setattr(
        coe,
        "load_name_overrides",
        lambda assets_dir: {"countries": {"Saudi Arabia": "KSA Common"}, "languages": {}},
    )
    stats_dir = tmp_path / "MyAlignRun"
    stats_dir.mkdir()
    make_corpus_stats(stats_dir / "corpus-stats.csv")
    make_verse_counts(stats_dir / "verse_counts.csv")
    run(
        request_dir=stats_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        dry_run=True,
    )
    output = capsys.readouterr().out
    assert f"Experiment location: {tmp_path / 'KSA_Common' / 'Saudi_Arabian_Sign_Language'}" in output
    assert "country: KSA Common" in output


def test_run_ignores_blank_country_override(tmp_path: Path, capsys, monkeypatch):
    # A whitespace-only override must be ignored, not collapse the country path level.
    import silnlp.common.create_onboarding_experiments as coe

    monkeypatch.setattr(
        coe,
        "load_name_overrides",
        lambda assets_dir: {"countries": {"Saudi Arabia": "  "}, "languages": {}},
    )
    stats_dir = tmp_path / "MyAlignRun"
    stats_dir.mkdir()
    make_corpus_stats(stats_dir / "corpus-stats.csv")
    make_verse_counts(stats_dir / "verse_counts.csv")
    run(
        request_dir=stats_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        dry_run=True,
    )
    output = capsys.readouterr().out
    assert f"Experiment location: {tmp_path / 'Saudi_Arabia' / 'Saudi_Arabian_Sign_Language'}" in output


def test_run_ignores_non_string_country_override(tmp_path: Path, capsys, monkeypatch):
    # A malformed (non-string) override value must degrade gracefully, not crash the run.
    import silnlp.common.create_onboarding_experiments as coe

    monkeypatch.setattr(
        coe,
        "load_name_overrides",
        lambda assets_dir: {"countries": {"Saudi Arabia": 5}, "languages": {}},
    )
    stats_dir = tmp_path / "MyAlignRun"
    stats_dir.mkdir()
    make_corpus_stats(stats_dir / "corpus-stats.csv")
    make_verse_counts(stats_dir / "verse_counts.csv")
    run(
        request_dir=stats_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        dry_run=True,
    )
    output = capsys.readouterr().out
    assert f"Experiment location: {tmp_path / 'Saudi_Arabia' / 'Saudi_Arabian_Sign_Language'}" in output


def test_run_copy_declined_aborts(request_dir: Path, tmp_path: Path, capsys, monkeypatch):
    log_path = request_dir / "onboarding.log"
    log_path.write_text(log_path.read_text(encoding="utf-8").replace("en-NIV11R", "sdl-NIV11R"), encoding="utf-8")
    counts_path = request_dir / "verse_counts.csv"
    counts_path.write_text(counts_path.read_text(encoding="utf-8").replace("en-NIV11R", "sdl-NIV11R"), encoding="utf-8")
    scripture_dir = tmp_path / "scripture"
    scripture_dir.mkdir()
    (scripture_dir / "sdl-A33_2026_07_02.txt").write_text("verses\n", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda prompt: "n" if "Copy" in prompt else "all")
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=scripture_dir,
    )
    assert experiments == []
    assert "Aborted: the copy is required" in capsys.readouterr().out
    # Nothing was copied and nothing was created.
    assert (scripture_dir / "sdl-A33_2026_07_02.txt").is_file()
    assert not (tmp_path / "Saudi_Arabia").exists()


def make_candidate(name: str, alignment: float) -> Candidate:
    return Candidate(name=name, stem=f"en-{name}", iso="en", count=1, parallel=1, alignment=alignment, script="Latn")


def test_select_experiments(monkeypatch, capsys):
    singles = [[make_candidate(f"S{i:02}", 0.9 - i / 100)] for i in range(15)]
    mixed = [[s[0], t[0]] for s, t in zip(singles, singles[1:])]  # 14 pairs

    # Capped at TOP_EXPERIMENTS with singles listed first.
    monkeypatch.setattr("builtins.input", lambda prompt: "all")
    chosen = select_experiments(singles, mixed, dry_run=False)
    assert len(chosen) == TOP_EXPERIMENTS
    assert chosen[:15] == singles and chosen[15:] == mixed[:5]
    output = capsys.readouterr().out
    assert f"top {TOP_EXPERIMENTS} of 29" in output
    assert "S00 (0.9000)" in output

    # Number selection picks from the displayed list; 'none' selects nothing.
    monkeypatch.setattr("builtins.input", lambda prompt: "1, 16")
    assert select_experiments(singles, mixed, dry_run=False) == [singles[0], mixed[0]]
    monkeypatch.setattr("builtins.input", lambda prompt: "none")
    assert select_experiments(singles, mixed, dry_run=False) == []

    # Duplicate tokens are deduplicated (a repeat would otherwise be submitted twice).
    monkeypatch.setattr("builtins.input", lambda prompt: "1,1 2")
    assert select_experiments(singles, mixed, dry_run=False) == [singles[0], singles[1]]

    # --top widens (or narrows) the display cap.
    monkeypatch.setattr("builtins.input", lambda prompt: "all")
    assert len(select_experiments(singles, mixed, dry_run=False, top=25)) == 25
    assert select_experiments(singles, mixed, dry_run=False, top=2) == singles[:2]

    # Dry run returns everything displayed without prompting.
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("dry run must not prompt"))
    assert len(select_experiments(singles, mixed, dry_run=True)) == TOP_EXPERIMENTS


def test_run_prefers_corpus_stats_over_log(request_dir: Path, tmp_path: Path, capsys, select_all):
    # A single-row CSV in alignments/ is preferred over the log; the log's main-project
    # line orients it (a lone row is otherwise ambiguous).
    alignments = request_dir / "alignments"
    alignments.mkdir()
    pd.DataFrame(
        {
            "src_project": ["sdl-A33_2026_07_02"],
            "trg_project": ["en-NIV11R"],
            "count": [31096],
            "parallel": [5070],
            "align_score": [0.5555],  # differs from the log's 0.3388 to prove the CSV is used
            "src_script": ["Arab"],
            "trg_script": ["Latn"],
        }
    ).to_csv(alignments / "corpus-stats.csv", index=False)

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    output = capsys.readouterr().out
    assert "0.5555" in output  # CSV numbers, not the log's
    # Experiments still go to the derived location, never into _OnboardingRequests.
    assert [e.folder for e in experiments] == [
        tmp_path / "Saudi_Arabia" / "Saudi_Arabian_Sign_Language" / "NIV11R_sdl_1"
    ]
    assert not (request_dir / "NIV11R_sdl_1").exists()


def test_run_target_mismatch_on_log_only(request_dir: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="does not match the main project"):
        run(
            request_dir=request_dir,
            experiments_dir=tmp_path,
            assets_dir=ASSETS_DIR,
            training_books="complete",
            translate_books="MAT",
            min_parallel=2000,
            min_alignment=0.2,
            target="nonexistent",
        )


def test_run_from_corpus_stats_folder(tmp_path: Path, capsys, select_all):
    # A stats folder inside the experiments tree creates experiments next to itself,
    # keeping the existing country/language folder naming (e.g. PNG/Taupota).
    experiments_dir = tmp_path
    align_dir = experiments_dir / "KSA" / "SignLang" / "Align"
    align_dir.mkdir(parents=True)
    make_corpus_stats(align_dir / "corpus-stats.csv")
    make_verse_counts(align_dir / "verse_counts.csv")

    assert resolve_request_dir("KSA/SignLang/Align", experiments_dir) == align_dir

    experiments = run(
        request_dir=align_dir,
        experiments_dir=experiments_dir,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    folders = sorted(e.folder.name for e in experiments)
    assert folders == ["HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1", "NIV11R_sdl_1"]
    # Created next to the Align folder, not under the derived Saudi_Arabia/... location.
    assert (experiments_dir / "KSA" / "SignLang" / "NIV11R_sdl_1" / "config.yml").is_file()
    assert not (experiments_dir / "Saudi_Arabia").exists()
    with open(experiments_dir / "KSA" / "SignLang" / "NIV11R_sdl_1" / "config.yml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    pair = config["data"]["corpus_pairs"][0]
    assert (pair["src"], pair["trg"], pair["corpus_books"]) == ("en-NIV11R", "sdl-A33_2026_07_02", "MRK")

    # A stats folder outside the experiments tree falls back to the derived location.
    outside = tmp_path / "outside_tree"
    outside.mkdir()
    make_corpus_stats(outside / "corpus-stats.csv")
    make_verse_counts(outside / "verse_counts.csv")
    experiments_dir2 = tmp_path / "experiments2"
    experiments_dir2.mkdir()
    capsys.readouterr()
    run(
        request_dir=outside,
        experiments_dir=experiments_dir2,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    assert (experiments_dir2 / "Saudi_Arabia" / "Saudi_Arabian_Sign_Language" / "NIV11R_sdl_1").is_dir()


def test_run_creates_experiments(request_dir: Path, tmp_path: Path, capsys, select_all):
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    lang_dir = tmp_path / "Saudi_Arabia" / "Saudi_Arabian_Sign_Language"
    folders = sorted(e.folder.name for e in experiments)
    assert folders == ["HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1", "NIV11R_sdl_1"]

    # The translated book MAT is excluded from corpus_books (complete would give MAT;MRK).
    output = capsys.readouterr().out
    assert "excluded the books being translated from corpus_books: MAT" in output

    with open(lang_dir / "NIV11R_sdl_1" / "config.yml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config == {
        "data": {
            "corpus_pairs": [
                {
                    "corpus_books": "MRK",
                    "mapping": "mixed_src",
                    "src": "en-NIV11R",
                    "trg": "sdl-A33_2026_07_02",
                    "type": "train,test",
                }
            ],
            "lang_codes": {"en": "eng_Latn", "sdl": "sdl_Arab"},
            "seed": 111,
        },
        "model": "facebook/nllb-200-distilled-1.3B",
    }

    # Mixed experiment: sources ordered by alignment (NIV11R 0.3388 first), one translate entry per source.
    with open(lang_dir / "NIV11R_HINCLBSI_sdl_1" / "config.yml", "r", encoding="utf-8") as file:
        mixed = yaml.safe_load(file)
    assert mixed["data"]["corpus_pairs"][0]["src"] == ["en-NIV11R", "hi-HINCLBSI"]
    assert mixed["data"]["lang_codes"] == {"en": "eng_Latn", "hi": "hin_Deva", "sdl": "sdl_Arab"}
    with open(lang_dir / "NIV11R_HINCLBSI_sdl_1" / "translate_config.yml", "r", encoding="utf-8") as file:
        translate = yaml.safe_load(file)
    assert translate == {
        "translate": [
            {"books": "MAT", "src_project": "NIV11R", "checkpoint": 5000},
            {"books": "MAT", "src_project": "HINCLBSI", "checkpoint": 5000},
        ],
        "postprocess": [{"paragraph_behavior": "place"}],
    }

    # Running again creates nothing (identical configs already exist), but the
    # existing experiments are still offered for running.
    again = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    assert again == []
    output = capsys.readouterr().out
    assert "To run the experiments:" in output
    for name in ["NIV11R_sdl_1", "HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1"]:
        assert f"Saudi_Arabia/Saudi_Arabian_Sign_Language/{name}" in output


def test_run_translating_all_training_books_skips(request_dir: Path, tmp_path: Path, capsys, select_all):
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT;MRK",
        min_parallel=2000,
        min_alignment=0.2,
    )
    assert experiments == []
    output = capsys.readouterr().out
    assert "no training books remain" in output


def test_run_test_variants(request_dir: Path, tmp_path: Path, select_all):
    for variant, expected_type in [("notest", "train"), ("test100", "train,test")]:
        experiments = run(
            request_dir=request_dir,
            experiments_dir=tmp_path,
            assets_dir=ASSETS_DIR,
            training_books="complete",
            translate_books="MAT",
            min_parallel=2000,
            min_alignment=0.2,
            test_variant=variant,
        )
        folders = sorted(e.folder.name for e in experiments)
        assert folders == [f"HINCLBSI_sdl_{variant}_1", f"NIV11R_HINCLBSI_sdl_{variant}_1", f"NIV11R_sdl_{variant}_1"]
        pair = experiments[0].config["data"]["corpus_pairs"][0]
        assert pair["type"] == expected_type
        if variant == "test100":
            assert pair["test_size"] == 100
        else:
            assert "test_size" not in pair


def test_run_iso_clash_copies_and_uses_synthetic_code(request_dir: Path, tmp_path: Path, capsys, caplog, select_all):
    # Make NIV11R clash with the main project by giving it the same iso prefix.
    log_path = request_dir / "onboarding.log"
    log_path.write_text(log_path.read_text(encoding="utf-8").replace("en-NIV11R", "sdl-NIV11R"), encoding="utf-8")
    counts_path = request_dir / "verse_counts.csv"
    counts_path.write_text(counts_path.read_text(encoding="utf-8").replace("en-NIV11R", "sdl-NIV11R"), encoding="utf-8")

    scripture_dir = tmp_path / "scripture"
    scripture_dir.mkdir()
    (scripture_dir / "sdl-A33_2026_07_02.txt").write_text("verses\n", encoding="utf-8")
    terms_dir = tmp_path / "terms"
    terms_dir.mkdir()
    (terms_dir / "sdl-A33_2026_07_02-Major-renderings.txt").write_text("terms\n", encoding="utf-8")

    # A dry run reports the clash and the would-be copy without touching anything,
    # even when no scripture directory is available.
    dry = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=scripture_dir,
        terms_dir=terms_dir,
        dry_run=True,
    )
    assert len(dry) == 3
    assert "Would copy sdl-A33_2026_07_02.txt" in capsys.readouterr().out
    assert (scripture_dir / "sdl-A33_2026_07_02.txt").is_file()
    run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=None,
        dry_run=True,
    )
    assert (scripture_dir / "sdl-A33_2026_07_02.txt").is_file()
    capsys.readouterr()

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=scripture_dir,
        terms_dir=terms_dir,
    )
    output = capsys.readouterr().out
    assert "shares iso code 'sdl' with the target project" in output

    real_isos = {entry["isoCode"] for entry in load_language_entries(ASSETS_DIR)}
    synthetic = synthesize_trg_iso("sdl", real_isos)
    # The extract and terms files were copied (originals kept) and the configs use the
    # synthetic stem and lang code.
    assert (scripture_dir / "sdl-A33_2026_07_02.txt").is_file()
    assert (scripture_dir / f"{synthetic}-A33_2026_07_02.txt").is_file()
    assert (terms_dir / "sdl-A33_2026_07_02-Major-renderings.txt").is_file()
    assert (terms_dir / f"{synthetic}-A33_2026_07_02-Major-renderings.txt").is_file()
    by_folder = {e.folder.name: e for e in experiments}
    assert f"NIV11R_{synthetic}_1" in by_folder
    pair = by_folder[f"NIV11R_{synthetic}_1"].config["data"]["corpus_pairs"][0]
    assert pair["src"] == "sdl-NIV11R"
    assert pair["trg"] == f"{synthetic}-A33_2026_07_02"
    lang_codes = by_folder[f"NIV11R_{synthetic}_1"].config["data"]["lang_codes"]
    assert lang_codes["sdl"] == "sdl_Latn"  # the source keeps its own (Latn) script tag
    assert lang_codes[synthetic] == f"{synthetic}_Arab"

    # Re-running with the copy already made reuses the code recorded by the file on disk.
    again = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=scripture_dir,
    )
    assert again == []

    # Even when no clash is detectable any more (here: the log no longer contains the
    # clashing stem), the prior copy recorded by the file on disk is adopted so the
    # configs keep matching the file that actually exists.
    log_path.write_text(log_path.read_text(encoding="utf-8").replace("sdl-NIV11R", "en-NIV11R"), encoding="utf-8")
    counts_path.write_text(counts_path.read_text(encoding="utf-8").replace("sdl-NIV11R", "en-NIV11R"), encoding="utf-8")
    capsys.readouterr()
    adopted = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=scripture_dir,
        dry_run=True,
    )
    assert "previously copied extract file" in capsys.readouterr().out
    assert adopted  # dry run still proposes experiments
    for experiment in adopted:
        assert experiment.config["data"]["corpus_pairs"][0]["trg"] == f"{synthetic}-A33_2026_07_02"

    # If the original is re-extracted after the copy was made, warn that the copy may be stale.
    now = time.time()
    os.utime(scripture_dir / "sdl-A33_2026_07_02.txt", (now + 60, now + 60))
    with caplog.at_level("WARNING"):
        run(
            request_dir=request_dir,
            experiments_dir=tmp_path,
            assets_dir=ASSETS_DIR,
            training_books="complete",
            translate_books="MAT",
            min_parallel=2000,
            min_alignment=0.2,
            scripture_dir=scripture_dir,
            dry_run=True,
        )
    assert any("may be outdated" in record.message for record in caplog.records)


def test_check_translate_source(tmp_path: Path):
    from silnlp.common.create_onboarding_experiments import check_translate_source

    projects_dir = tmp_path / "projects"
    make_paratext_project(projects_dir, "NIV11R", ["MAT"])
    assert check_translate_source(projects_dir, "NIV11R", ["MAT"]) is None
    assert "does not contain RUT" in check_translate_source(projects_dir, "NIV11R", ["MAT", "RUT"])
    assert "no project folder 'MISSING'" in check_translate_source(projects_dir, "MISSING", ["MAT"])
    # A project folder without a readable Settings.xml is reported, not crashed on.
    (projects_dir / "BROKEN").mkdir()
    assert "could not be read" in check_translate_source(projects_dir, "BROKEN", ["MAT"])


def test_run_checks_translate_sources(request_dir: Path, tmp_path: Path, capsys, monkeypatch):
    # NIV11R has MAT; HINCLBSI's project folder is missing. The user first tries a project
    # that lacks MAT (warned again), then one that has it; translate_config uses the choice.
    projects_dir = tmp_path / "projects"
    make_paratext_project(projects_dir, "NIV11R", ["MAT"])
    make_paratext_project(projects_dir, "NOMAT", ["RUT"])
    make_paratext_project(projects_dir, "HINDI2", ["MAT"])

    answers = iter(["NOMAT", "HINDI2"])

    def answer(prompt: str) -> str:
        if "different project" in prompt:
            return next(answers)
        return "all"

    monkeypatch.setattr("builtins.input", answer)
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        projects_dir=projects_dir,
    )
    output = capsys.readouterr().out
    assert "cannot translate MAT from 'HINCLBSI': there is no project folder 'HINCLBSI'" in output
    assert "cannot translate MAT from 'NOMAT': project 'NOMAT' does not contain MAT" in output
    assert "Translating from 'HINDI2' instead of 'HINCLBSI'." in output

    by_folder = {e.folder.name: e for e in experiments}
    translate = by_folder["NIV11R_HINCLBSI_sdl_1"].translate_config["translate"]
    assert [entry["src_project"] for entry in translate] == ["NIV11R", "HINDI2"]
    # The training config still uses the original extract stems.
    assert by_folder["NIV11R_HINCLBSI_sdl_1"].config["data"]["corpus_pairs"][0]["src"] == ["en-NIV11R", "hi-HINCLBSI"]


def test_run_translate_source_warnings_only_in_dry_run(request_dir: Path, tmp_path: Path, capsys, monkeypatch):
    projects_dir = tmp_path / "projects"
    make_paratext_project(projects_dir, "NIV11R", ["MAT"])
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("dry run must not prompt for translate sources"))
    run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        projects_dir=projects_dir,
        dry_run=True,
    )
    output = capsys.readouterr().out
    assert "cannot translate MAT from 'HINCLBSI'" in output


def test_run_rejects_unknown_test_variant(request_dir: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown test_variant"):
        run(
            request_dir=request_dir,
            experiments_dir=tmp_path,
            assets_dir=ASSETS_DIR,
            training_books="complete",
            translate_books="MAT",
            min_parallel=2000,
            min_alignment=0.2,
            test_variant="no-test",
        )


def test_submit_experiments(monkeypatch, capsys, tmp_path: Path):
    experiment = Experiment(
        sources=[], folder=tmp_path / "Country" / "Lang" / "NIV11R_sdl_1", config={}, translate_config={}
    )
    calls = []
    monkeypatch.setattr(
        "silnlp.common.create_onboarding_experiments.subprocess.run",
        lambda cmd: calls.append(cmd) or SimpleNamespace(returncode=0),
    )

    submit_experiments([experiment], tmp_path, submit=False)
    output = capsys.readouterr().out
    assert f"poetry run python {' '.join(EXPERIMENT_ARGS)} Country/Lang/NIV11R_sdl_1" in output
    assert calls == []

    submit_experiments([experiment], tmp_path, submit=True)
    assert calls == [[sys.executable] + EXPERIMENT_ARGS + ["Country/Lang/NIV11R_sdl_1"]]

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    submit_experiments([experiment], tmp_path, submit=None)
    assert len(calls) == 1  # declined at the prompt, nothing new ran

    # no_test drops the --test stage from both the printed and executed command.
    capsys.readouterr()
    submit_experiments([experiment], tmp_path, submit=True, no_test=True)
    assert calls[-1] == [sys.executable] + [a for a in EXPERIMENT_ARGS if a != "--test"] + ["Country/Lang/NIV11R_sdl_1"]
    assert "--test" not in capsys.readouterr().out


def test_run_dry_run(request_dir: Path, tmp_path: Path, capsys):
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MAT;MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        dry_run=True,
    )
    assert len(experiments) == 3
    # Explicit --training-books is used verbatim (no automatic subtraction of the translate books).
    assert all(e.config["data"]["corpus_pairs"][0]["corpus_books"] == "MAT;MRK" for e in experiments)
    assert not (tmp_path / "Saudi_Arabia").exists()
    # A dry run lists what would be created but does not print run commands.
    output = capsys.readouterr().out
    assert "Would create" in output
    assert "To run the experiments:" not in output


def test_book_coverage_extract_fallback(tmp_path: Path):
    # A stem with no verse_counts row falls back to counting its vref-aligned extract file.
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "vref.txt").write_text("GEN 1:1\nGEN 1:2\nMAT 1:1\n", encoding="utf-8")
    scripture_dir = tmp_path / "scripture"
    scripture_dir.mkdir()
    (scripture_dir / "en-EXTRACT.txt").write_text("In the beginning\n\n<range>\n", encoding="utf-8")

    assert extract_book_counts(scripture_dir / "en-EXTRACT.txt", load_vref_books(assets_dir)) == {"GEN": 1, "MAT": 1}

    counts_df = pd.DataFrame({"file": ["complete", "en-ROW"], "GEN": [2, 1], "MAT": [1, 0]}).set_index("file")
    coverage = BookCoverage(counts_df, scripture_dir, assets_dir)
    assert coverage.counts("en-ROW") == {"GEN": 1, "MAT": 0}  # csv row wins
    assert coverage.counts("en-EXTRACT") == {"GEN": 1, "MAT": 1}  # extract fallback
    assert coverage.counts("en-NOWHERE") is None
    assert coverage.complete() == {"GEN": 2, "MAT": 1}  # the csv's complete row

    # Without verse counts the complete counts come from vref.txt itself.
    coverage = BookCoverage(None, scripture_dir, assets_dir)
    assert coverage.complete() == {"GEN": 2, "MAT": 1}


def test_load_vref_chapters(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "vref.txt").write_text("GEN 1:1\nGEN 1:2\nGEN 2:1\nMAT 1:1\n", encoding="utf-8")
    assert load_vref_chapters(assets_dir) == [("GEN", 1), ("GEN", 1), ("GEN", 2), ("MAT", 1)]


def test_book_coverage_presence_chapter_level(tmp_path: Path):
    # Chapter-level presence honours a selection like GEN 1: only its verses are counted.
    from machine.scripture import get_chapters

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "vref.txt").write_text("GEN 1:1\nGEN 1:2\nGEN 2:1\nMAT 1:1\n", encoding="utf-8")
    scripture_dir = tmp_path / "scripture"
    scripture_dir.mkdir()
    # GEN 1:1 present, GEN 1:2 blank, GEN 2:1 present, MAT 1:1 present.
    (scripture_dir / "en-SRC.txt").write_text("verse\n\nverse\nverse\n", encoding="utf-8")
    coverage = BookCoverage(None, scripture_dir, assets_dir)

    assert coverage.presence("en-SRC", get_chapters("GEN 1")) == (1, 2)  # 1 of GEN 1's 2 verses present
    assert coverage.presence("en-SRC", get_chapters("GEN")) == (2, 3)  # whole book: 2 of 3 present
    assert coverage.presence("en-SRC", get_chapters("MAT")) == (1, 1)
    assert coverage.presence("en-NOWHERE", get_chapters("GEN")) is None  # no extract -> unmeasurable
    # No scripture dir at all -> unmeasurable.
    assert BookCoverage(None, None, assets_dir).presence("en-SRC", get_chapters("GEN")) is None


def test_overlapping_books():
    from machine.scripture import get_chapters

    assert overlapping_books(get_chapters("NT"), get_chapters("MAT")) == ["MAT"]
    assert overlapping_books(get_chapters("NT;-MAT"), get_chapters("MAT")) == []  # MAT subtracted from training
    assert overlapping_books(get_chapters("GEN"), get_chapters("MAT")) == []  # disjoint books
    # Chapter-level: same book, disjoint chapters -> no overlap; intersecting -> overlap.
    assert overlapping_books(get_chapters("MAT 1-5"), get_chapters("MAT 6-10")) == []
    assert overlapping_books(get_chapters("MAT 1-5"), get_chapters("MAT 4-8")) == ["MAT"]
    # A whole book on either side overlaps a chapter selection of it.
    assert overlapping_books(get_chapters("MAT"), get_chapters("MAT 6-10")) == ["MAT"]


def test_candidate_table_render(request_dir: Path, tmp_path: Path, capsys):
    # Pin the table the user specified: each candidate once, a heading row with 'total' (not
    # 'count'), the train/draft/trg-only columns in order, then one three-state column per
    # --translate-book. NIV11R has full MAT (✓) and partial MRK (~); HINCLBSI has no MRK (X).
    counts_path = request_dir / "verse_counts.csv"
    df = pd.read_csv(counts_path)
    df.loc[df["file"] == "en-NIV11R", "MRK"] = 300  # partial (< 98% of 678)
    df.loc[df["file"] == "hi-HINCLBSI", "MRK"] = 0  # none
    df.to_csv(counts_path, index=False)

    run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="GEN;EXO",
        translate_books="MAT;MRK",
        min_parallel=2000,
        min_alignment=0.2,
        dry_run=True,
    )
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if "Candidate" in line and "align" in line)
    assert header.split() == ["#", "Candidate", "align", "total", "train", "draft", "trg-only", "script", "MAT", "MRK"]

    def row_for(name: str) -> list:
        return next(line.split() for line in lines if line.split()[1:2] == [name])

    assert row_for("NIV11R")[-2:] == ["✓", "~"]  # MAT full, MRK partial
    assert row_for("HINCLBSI")[-2:] == ["✓", "X"]  # MAT full, MRK none


def test_run_selection_overrides_book_filter(request_dir: Path, tmp_path: Path, select_all):
    # NIV11R misses MRK, which the target contains — the old book-coverage filter would have
    # excluded it as a primary source. Now the user can pick it from the table, so it is used
    # as a single/primary source, leading the pair by its higher alignment.
    counts_path = request_dir / "verse_counts.csv"
    df = pd.read_csv(counts_path)
    df.loc[df["file"] == "en-NIV11R", "MRK"] = 0
    df.to_csv(counts_path, index=False)

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        translate_scripture=["NGT"],  # fix the drafting source so no drafter prompt is needed
    )
    folders = sorted(e.folder.name for e in experiments)
    assert folders == ["HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1", "NIV11R_sdl_1"]
    by_folder = {e.folder.name: e for e in experiments}
    # NIV11R (higher alignment) now leads the pair rather than being demoted for missing MRK.
    assert by_folder["NIV11R_HINCLBSI_sdl_1"].config["data"]["corpus_pairs"][0]["src"] == [
        "en-NIV11R",
        "hi-HINCLBSI",
    ]


def test_run_select_none_creates_nothing(request_dir: Path, tmp_path: Path, capsys, monkeypatch):
    # Selecting no candidates from the table creates nothing and asks nothing further.
    def answer(prompt: str) -> str:
        if "candidates to use" in prompt:
            return "none"
        pytest.fail(f"nothing else should be asked: {prompt}")

    monkeypatch.setattr("builtins.input", answer)
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    assert experiments == []
    assert "No candidates selected" in capsys.readouterr().out


def test_run_secondary_threshold_blocks_pairs(request_dir: Path, tmp_path: Path, capsys, select_all):
    # The second source of a pair still needs parallel verses >= max(1000, 25% of the target's
    # verses): with a 40000-verse target the references' ~5000 parallel verses are not worth a
    # pair, so only single-source experiments are offered.
    df = pd.DataFrame(
        {
            "file": ["complete", "sdl-A33_2026_07_02", "en-NIV11R", "hi-HINCLBSI", "arb-a55_2026_07_02"],
            "MAT": [20000, 20000, 20000, 20000, 0],
            "MRK": [20000, 20000, 20000, 20000, 0],
        }
    )
    df.to_csv(request_dir / "verse_counts.csv", index=False)

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    output = capsys.readouterr().out
    assert "Note: NIV11R can be a single source or the primary of a pair, but not a pair's second source" in output
    assert "5070 parallel verses are below 10000" in output
    assert sorted(e.folder.name for e in experiments) == ["HINCLBSI_sdl_1", "NIV11R_sdl_1"]


def test_run_extract_fallback_for_missing_counts_row(request_dir: Path, tmp_path: Path, select_all):
    # HINCLBSI has no verse_counts row; its book coverage is computed from the extract file.
    counts_path = request_dir / "verse_counts.csv"
    df = pd.read_csv(counts_path)
    df[df["file"] != "hi-HINCLBSI"].to_csv(counts_path, index=False)

    scripture_dir = tmp_path / "scripture"
    scripture_dir.mkdir()
    vref_books = load_vref_books(ASSETS_DIR)
    lines = ["text" if book in ("MAT", "MRK") else "" for book in vref_books]
    (scripture_dir / "hi-HINCLBSI.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=scripture_dir,
    )
    folders = sorted(e.folder.name for e in experiments)
    assert folders == ["HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1", "NIV11R_sdl_1"]
    # The extract holds a complete NT (MAT;MRK of the fixture canon), so HINCLBSI also
    # qualifies as a drafting source for an NT book.
    by_folder = {e.folder.name: e for e in experiments}
    translate = by_folder["NIV11R_HINCLBSI_sdl_1"].translate_config["translate"]
    assert [entry["src_project"] for entry in translate] == ["NIV11R", "HINCLBSI"]


def test_run_unknown_coverage_still_offered(request_dir: Path, tmp_path: Path, capsys, select_all):
    # A candidate with no verse_counts row and no extract file (unknown coverage) is still
    # listed in the table with 'X' marks and can be selected and used.
    counts_path = request_dir / "verse_counts.csv"
    df = pd.read_csv(counts_path)
    df[df["file"] != "hi-HINCLBSI"].to_csv(counts_path, index=False)

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    lines = capsys.readouterr().out.splitlines()
    hinclbsi_row = next(line.split() for line in lines if line.split()[1:2] == ["HINCLBSI"])
    assert hinclbsi_row[-1] == "X"  # unknown coverage (no counts, no extract) -> no-data mark
    folders = sorted(e.folder.name for e in experiments)
    assert folders == ["HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1", "NIV11R_sdl_1"]


def test_run_translate_scripture_overrides(request_dir: Path, tmp_path: Path, capsys, select_all):
    # --translate-scripture replaces the automatic drafting choice for every experiment;
    # a missing project or missing books is warned about but never dropped.
    projects_dir = tmp_path / "projects"
    make_paratext_project(projects_dir, "NIV11R", ["MAT"])

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        projects_dir=projects_dir,
        translate_scripture=["MYDRAFT"],
    )
    output = capsys.readouterr().out
    assert "cannot translate MAT from 'MYDRAFT': there is no project folder 'MYDRAFT'" in output
    assert "Including it anyway" in output
    assert len(experiments) == 3
    for experiment in experiments:
        assert [entry["src_project"] for entry in experiment.translate_config["translate"]] == ["MYDRAFT"]


def test_run_warns_missing_specified_verses(request_dir: Path, tmp_path: Path, capsys, select_all):
    # A source missing a quarter or more of the specified translate/training verses is warned
    # about (chapter-level, from its extract), but the experiment is still created.
    scripture_dir = tmp_path / "scripture"
    scripture_dir.mkdir()
    vref_books = load_vref_books(ASSETS_DIR)
    # NIV11R's extract has MRK but no MAT: it is missing 100% of the MAT translate verses.
    niv_lines = ["text" if book == "MRK" else "" for book in vref_books]
    (scripture_dir / "en-NIV11R.txt").write_text("\n".join(niv_lines) + "\n", encoding="utf-8")
    # HINCLBSI has both, so no warning for it.
    hin_lines = ["text" if book in ("MAT", "MRK") else "" for book in vref_books]
    (scripture_dir / "hi-HINCLBSI.txt").write_text("\n".join(hin_lines) + "\n", encoding="utf-8")

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MAT;MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        scripture_dir=scripture_dir,
    )
    output = capsys.readouterr().out
    assert "'NIV11R' is missing 100% of the translate verses specified by 'MAT'" in output
    # HINCLBSI has the translate books, so it is not warned about for translation.
    assert "'HINCLBSI' is missing" not in output.split("translate verses")[0]
    assert experiments  # the warning does not block creation


def test_run_warns_training_translate_overlap(request_dir: Path, tmp_path: Path, capsys, select_all):
    # Training on and translating the same books is warned about (once), but not blocked.
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MAT;MRK",  # overlaps the translate book MAT
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    output = capsys.readouterr().out
    assert "the training books 'MAT;MRK' and translate books 'MAT' overlap in MAT" in output
    assert output.count("overlap in MAT") == 1  # warned once, not per experiment
    assert experiments  # not blocked


def test_run_no_overlap_warning_when_disjoint(request_dir: Path, tmp_path: Path, capsys, select_all):
    # Subtracting the translate book from training removes the overlap and the warning.
    run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MAT;MRK;-MAT",  # MAT excluded from training
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    assert "and translate books" not in capsys.readouterr().out  # the overlap warning did not fire


def test_run_rerun_updates_translate_config(request_dir: Path, tmp_path: Path, capsys, select_all):
    # config.yml is unchanged by --translate-scripture, so a re-run skips the identical
    # folders — but the on-disk translate_config.yml must still be brought in line, or the
    # override would be a silent no-op.
    run_args = dict(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="complete",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    run(**run_args)
    config_path = tmp_path / "Saudi_Arabia" / "Saudi_Arabian_Sign_Language" / "NIV11R_sdl_1" / "translate_config.yml"
    with open(config_path, "r", encoding="utf-8") as file:
        assert yaml.safe_load(file)["translate"][0]["src_project"] == "NIV11R"

    capsys.readouterr()
    again = run(**run_args, translate_scripture=["MYDRAFT"])
    assert again == []  # nothing new created
    output = capsys.readouterr().out
    assert "already contains an identical config.yml" in output
    assert "Updated" in output and "translate_config.yml" in output
    with open(config_path, "r", encoding="utf-8") as file:
        assert yaml.safe_load(file)["translate"][0]["src_project"] == "MYDRAFT"


def test_run_clash_only_from_selected_candidates(tmp_path: Path, capsys, monkeypatch):
    # A back translation sharing the target's iso only forces a synthetic target code (and the
    # extract copy) if the user actually selects it. Choosing REF1 alone leaves the target
    # code untouched, even though BT shares the target's iso.
    stats_dir = tmp_path / "Scenario"
    stats_dir.mkdir()
    pd.DataFrame(
        {
            "src_project": ["en-REF1", "sdl-BT"],
            "trg_project": ["sdl-MAIN", "sdl-MAIN"],
            "count": [40000, 5500],
            "parallel": [20000, 5000],
            "align_score": [0.5, 0.9],
            "src_script": ["Latn", "Arab"],
            "trg_script": ["Arab", "Arab"],
        }
    ).to_csv(stats_dir / "corpus-stats.csv", index=False)
    pd.DataFrame(
        {
            "file": ["complete", "sdl-MAIN", "en-REF1", "sdl-BT"],
            "MAT": [20000, 20000, 20000, 500],
            "MRK": [20000, 20000, 20000, 0],
        }
    ).to_csv(stats_dir / "verse_counts.csv", index=False)

    # The table lists BT first (higher alignment) then REF1; pick only REF1.
    def answer(prompt: str) -> str:
        if "candidates to use" in prompt:
            return "2"
        if "numbers to create" in prompt:
            return "all"
        pytest.fail(f"Unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", answer)
    experiments = run(
        request_dir=stats_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    output = capsys.readouterr().out
    assert "synthetic" not in output
    assert [e.folder.name for e in experiments] == ["REF1_sdl_1"]
    assert experiments[0].config["data"]["corpus_pairs"][0]["trg"] == "sdl-MAIN"


def test_run_every_source_drafts(request_dir: Path, tmp_path: Path, select_all):
    # No drafting-qualification gate: every training source drafts, so a single-source
    # experiment has one translate entry and a mixed experiment has one per source, whatever
    # their book coverage.
    counts_path = request_dir / "verse_counts.csv"
    df = pd.read_csv(counts_path)
    df.loc[df["file"] == "en-NIV11R", ["GEN", "EXO", "MRK"]] = 0  # NIV11R now only partial NT
    df.to_csv(counts_path, index=False)

    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        training_books="MRK",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
    )
    by_folder = {e.folder.name: e for e in experiments}
    assert [e["src_project"] for e in by_folder["NIV11R_sdl_1"].translate_config["translate"]] == ["NIV11R"]
    assert [e["src_project"] for e in by_folder["NIV11R_HINCLBSI_sdl_1"].translate_config["translate"]] == [
        "NIV11R",
        "HINCLBSI",
    ]
