import sys
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
    Candidate,
    Experiment,
    find_existing,
    folder_name,
    load_language_entries,
    nllb_tag,
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
    """Answer the experiment-selection prompt with 'all'."""
    monkeypatch.setattr("builtins.input", lambda prompt: "all")


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
    # Full get_chapters syntax is preserved verbatim (ranges, chapters, subtraction).
    assert parse_book_list("MAT 1-4") == "MAT 1-4"
    assert parse_book_list("mat 1-4") == "MAT 1-4"
    assert parse_book_list("NT;-REV") == "NT;-REV"
    assert parse_book_list("GEN-DEU") == "GEN-DEU"
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


def test_run_iso_clash_renames_and_uses_synthetic_code(request_dir: Path, tmp_path: Path, capsys, select_all):
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

    # A dry run reports the clash and the would-be rename without touching anything,
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
    assert "Would rename sdl-A33_2026_07_02.txt" in capsys.readouterr().out
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
    # The extract and terms files were renamed and the configs use the synthetic stem and lang code.
    assert not (scripture_dir / "sdl-A33_2026_07_02.txt").exists()
    assert (scripture_dir / f"{synthetic}-A33_2026_07_02.txt").is_file()
    assert not (terms_dir / "sdl-A33_2026_07_02-Major-renderings.txt").exists()
    assert (terms_dir / f"{synthetic}-A33_2026_07_02-Major-renderings.txt").is_file()
    by_folder = {e.folder.name: e for e in experiments}
    assert f"NIV11R_{synthetic}_1" in by_folder
    pair = by_folder[f"NIV11R_{synthetic}_1"].config["data"]["corpus_pairs"][0]
    assert pair["src"] == "sdl-NIV11R"
    assert pair["trg"] == f"{synthetic}-A33_2026_07_02"
    lang_codes = by_folder[f"NIV11R_{synthetic}_1"].config["data"]["lang_codes"]
    assert lang_codes["sdl"] == "sdl_Latn"  # the source keeps its own (Latn) script tag
    assert lang_codes[synthetic] == f"{synthetic}_Arab"

    # Re-running with the file already renamed reuses the code recorded by the file on disk.
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
    # clashing stem), the prior rename recorded by the file on disk is adopted so the
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
    assert "previously renamed extract file" in capsys.readouterr().out
    assert adopted  # dry run still proposes experiments
    for experiment in adopted:
        assert experiment.config["data"]["corpus_pairs"][0]["trg"] == f"{synthetic}-A33_2026_07_02"

    # If the original extract reappears next to the renamed one, the ambiguity is an error.
    (scripture_dir / "sdl-A33_2026_07_02.txt").write_text("verses\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remove the stale one"):
        run(
            request_dir=request_dir,
            experiments_dir=tmp_path,
            assets_dir=ASSETS_DIR,
            training_books="complete",
            translate_books="MAT",
            min_parallel=2000,
            min_alignment=0.2,
            scripture_dir=scripture_dir,
        )


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


def test_run_dry_run(request_dir: Path, tmp_path: Path):
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
    assert all(e.config["data"]["corpus_pairs"][0]["corpus_books"] == "MAT;MRK;-MAT" for e in experiments)
    assert not (tmp_path / "Saudi_Arabia").exists()
