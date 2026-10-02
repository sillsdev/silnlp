from pathlib import Path

from silnlp.common.check_config_files import get_corpus_paths
from silnlp.common.environment import SilNlpEnv


def test_corpus_paths_are_found_in_corpora_and_scripture(tmp_path: Path):
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path / "MT")
    for path in (environment.mt_corpora_dir / "en-AAA.txt", environment.mt_scripture_dir / "es-BBB.txt"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("text\n", encoding="utf-8")

    paths = get_corpus_paths([{"src": "en-AAA", "trg": "es-BBB"}], environment)

    assert paths == [environment.mt_corpora_dir / "en-AAA.txt", environment.mt_scripture_dir / "es-BBB.txt"]
