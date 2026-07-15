import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from silnlp.common.create_onboarding_experiments import (
    EXPERIMENT_ARGS,
    NT_CANON,
    Experiment,
    find_existing,
    folder_name,
    nllb_tag,
    parse_log,
    resolve_corpus_books,
    run,
    submit_experiments,
)

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


def test_resolve_corpus_books_verbatim():
    assert resolve_corpus_books("GEN;EXO;NT", [], None) == "GEN;EXO;NT"


def test_resolve_corpus_books_complete(request_dir: Path):
    df = pd.read_csv(request_dir / "verse_counts.csv", index_col="file")
    books = resolve_corpus_books("complete", ["en-NIV11R", "sdl-A33_2026_07_02"], df)
    assert books == "MAT;MRK"
    # arb-a55 has partial MRK (179 < 98% of 678), so nothing qualifies with it as a source.
    assert resolve_corpus_books("complete", ["arb-a55_2026_07_02", "sdl-A33_2026_07_02"], df) == ""


def test_resolve_corpus_books_nt_compaction():
    stems = ["en-NIV11R", "sdl-TRG"]
    df = pd.DataFrame(100, index=["complete"] + stems, columns=NT_CANON)
    df.index.name = "file"
    assert resolve_corpus_books("complete", stems, df) == "NT"


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


def test_run_creates_experiments(request_dir: Path, tmp_path: Path, capsys):
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        books="complete",
        translate_books="MAT;MRK",
        min_parallel=2000,
        min_alignment=0.2,
    )
    lang_dir = tmp_path / "Saudi_Arabia" / "Saudi_Arabian_Sign_Language"
    folders = sorted(e.folder.name for e in experiments)
    assert folders == ["HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1", "NIV11R_sdl_1"]

    with open(lang_dir / "NIV11R_sdl_1" / "config.yml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config == {
        "data": {
            "corpus_pairs": [
                {
                    "corpus_books": "MAT;MRK",
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
            {"books": "MAT;MRK", "src_project": "NIV11R", "checkpoint": 5000},
            {"books": "MAT;MRK", "src_project": "HINCLBSI", "checkpoint": 5000},
        ],
        "postprocess": [{"paragraph_behavior": "place"}],
    }

    # Running again creates nothing (identical configs already exist), but the
    # existing experiments are still offered for running.
    capsys.readouterr()
    again = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        books="complete",
        translate_books="MAT;MRK",
        min_parallel=2000,
        min_alignment=0.2,
    )
    assert again == []
    output = capsys.readouterr().out
    assert "To run the experiments:" in output
    for name in ["NIV11R_sdl_1", "HINCLBSI_sdl_1", "NIV11R_HINCLBSI_sdl_1"]:
        assert f"Saudi_Arabia/Saudi_Arabian_Sign_Language/{name}" in output


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


def test_run_dry_run(request_dir: Path, tmp_path: Path):
    experiments = run(
        request_dir=request_dir,
        experiments_dir=tmp_path,
        assets_dir=ASSETS_DIR,
        books="MAT",
        translate_books="MAT",
        min_parallel=2000,
        min_alignment=0.2,
        dry_run=True,
    )
    assert len(experiments) == 3
    assert not (tmp_path / "Saudi_Arabia").exists()
