from pathlib import Path

import pytest

from silnlp.common.environment import SilNlpEnv
from silnlp.common.postprocessor import PostprocessConfig
from silnlp.nmt.corpora import parse_corpus_pairs


def test_quote_convention_detection_looks_for_no_project_behind_basic_corpora(tmp_path: Path):
    corpora_dir = tmp_path / "MT" / "corpora"
    corpora_dir.mkdir(parents=True)
    (corpora_dir / "en-AAA.txt").write_text("source\n", encoding="utf-8")
    (corpora_dir / "es-BBB.txt").write_text("objetivo\n", encoding="utf-8")
    environment = SilNlpEnv.create_environment_with_mt_dir(tmp_path / "MT")
    corpus_pairs = parse_corpus_pairs([{"src": "en-AAA", "trg": "es-BBB"}], environment)
    config = PostprocessConfig({"denormalize_quotation_marks": True}, environment)

    with pytest.raises(ValueError, match="specify a target project name"):
        config.create_denormalize_quotation_marks_postprocessor(corpus_pairs)
