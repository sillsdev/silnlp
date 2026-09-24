import pytest

from silnlp.nmt.draft_files import DraftFiles


@pytest.fixture
def files(tmp_path) -> DraftFiles:
    return DraftFiles(tmp_path / "exp")


def test_the_translate_config_sits_in_the_experiment_directory(files, tmp_path):
    assert files.translate_config() == tmp_path / "exp" / "translate_config.yml"


def test_drafts_are_filed_under_the_checkpoint_they_came_from(files, tmp_path):
    assert files.inference_directory("5000") == tmp_path / "exp" / "infer" / "5000"
    assert files.inference_directory("5000", "BSB") == tmp_path / "exp" / "infer" / "5000" / "BSB"
    assert files.inference_directory("5000", "BSB", "LBLA") == tmp_path / "exp" / "infer" / "5000" / "BSB" / "LBLA"


def test_a_draft_is_named_by_the_book_it_translates(files, tmp_path):
    assert files.draft("5000", "BSB", 40) == tmp_path / "exp" / "infer" / "5000" / "BSB" / "41MAT.SFM"


def test_book_numbers_are_padded_so_drafts_sort_in_canonical_order(files):
    names = [files.draft("5000", "BSB", book_num).name for book_num in (1, 9, 39, 40, 66)]

    # Paratext leaves 40 free between the testaments, so Matthew is filed as 41
    assert names == ["01GEN.SFM", "091SA.SFM", "39MAL.SFM", "41MAT.SFM", "67REV.SFM"]
    assert names == sorted(names)


def test_an_untrained_model_files_its_drafts_under_base(files, tmp_path):
    assert files.inference_directory("base", "BSB") == tmp_path / "exp" / "infer" / "base" / "BSB"
