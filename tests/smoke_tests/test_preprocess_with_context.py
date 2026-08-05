from pathlib import Path
from typing import List, cast

from silnlp.common.environment import SilNlpEnv
from silnlp.common.sentence_context import CONTEXT_END_TOKEN, CONTEXT_START_TOKEN, find_central_segment
from silnlp.nmt.config_utils import load_config_from_exp_dir
from silnlp.nmt.seq2seq_config import Seq2SeqConfig

TEST_MT_DIR = Path(__file__).parent
EXPERIMENT_NAME = "test_experiment_context"


def test_preprocess_with_sentence_context():
    environment = set_up_environment()
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    clean_experiment_directory(exp_dir)

    config = cast(Seq2SeqConfig, load_config_from_exp_dir(exp_dir, environment))
    assert config.context_size == 2
    assert config.context_window_size == 5

    config.set_seed()
    config.preprocess(stats=False)

    check_sequence_lengths_were_scaled(config)
    # A fresh config, as the separately invoked train, test and translate steps build.
    check_context_markers_are_single_tokens(cast(Seq2SeqConfig, load_config_from_exp_dir(exp_dir, environment)))
    check_windows_were_written(exp_dir)
    check_window_contents(exp_dir)

    clean_experiment_directory(exp_dir)


def set_up_environment() -> SilNlpEnv:
    # As with the other smoke tests, the source corpora are read from the MinIO bucket.
    return SilNlpEnv.create_environment_with_mt_experiments_dir(TEST_MT_DIR / "experiments")


def clean_experiment_directory(experiment_directory: Path):
    for pattern in ("train*", "test*", "val*", "tokenizer*", "special_tokens*", "sentencepiece*", "added_tokens*"):
        for file in experiment_directory.glob(pattern):
            file.unlink()


def check_sequence_lengths_were_scaled(config: Seq2SeqConfig):
    # The default limits are 200 tokens per sentence; a 5 sentence window needs room for all of them.
    assert config.train["max_source_length"] == 1000
    assert config.train["max_target_length"] == 1000


def check_context_markers_are_single_tokens(config: Seq2SeqConfig):
    tokenizer = config.get_tokenizer()
    for marker in (CONTEXT_START_TOKEN, CONTEXT_END_TOKEN):
        assert tokenizer.tokenize(marker) == [marker], f"{marker} was split into multiple tokens"
        assert tokenizer.convert_tokens_to_ids(marker) != tokenizer.unk_token_id
    # The markers have to survive decoding, otherwise a draft has no way to indicate which part of
    # the output is the translated sentence.
    ids = tokenizer(f"uno {CONTEXT_START_TOKEN} dos {CONTEXT_END_TOKEN} tres").input_ids
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert find_central_segment(decoded) == "dos"


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() != ""]


def check_windows_were_written(exp_dir: Path):
    for file_name in ("train.src.txt", "train.trg.txt", "val.src.txt", "val.trg.txt", "test.src.txt"):
        lines = read_lines(exp_dir / file_name)
        assert len(lines) > 0, f"{file_name} is empty"
        marked = [line for line in lines if CONTEXT_START_TOKEN in line and CONTEXT_END_TOKEN in line]
        assert len(marked) == len(lines), f"{len(lines) - len(marked)} lines in {file_name} have no context markers"

    # References are stored as whole windows too; only the marked sentence is scored.
    test_refs = read_lines(exp_dir / "test.trg.detok.txt")
    assert len(test_refs) > 0
    assert all(CONTEXT_START_TOKEN in line and CONTEXT_END_TOKEN in line for line in test_refs)


def check_window_contents(exp_dir: Path):
    # The detokenized files hold plain text, so the window structure can be checked directly against
    # the single sentences each window is centered on.
    windows = read_lines(exp_dir / "train.src.detok.txt")
    centrals = [find_central_segment(window) for window in windows]
    assert all(central is not None and central != "" for central in centrals)

    # A window in the middle of a run repeats its neighbors' central sentences as context.
    index = next(
        i
        for i in range(2, len(windows) - 2)
        if all(central is not None for central in centrals[i - 2 : i + 3])
        and windows[i].startswith(f"{centrals[i - 2]} {centrals[i - 1]} {CONTEXT_START_TOKEN}")
    )
    assert windows[index].endswith(f"{CONTEXT_END_TOKEN} {centrals[index + 1]} {centrals[index + 2]}")
