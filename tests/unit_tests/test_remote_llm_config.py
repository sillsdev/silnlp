import math
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock

import pytest
import yaml

from silnlp.nmt.config import Language
from silnlp.nmt.config_utils import is_llm_config, is_remote_llm_config
from silnlp.nmt.example_retrieval import (
    BM25ExampleRetriever,
    ExamplePair,
    ExampleRetriever,
    TfidfExampleRetriever,
    create_retriever,
    tokenize_for_retrieval,
)
from silnlp.nmt.remote_llm_config import (
    CONTEXT_MODE_FULL_CORPUS,
    CONTEXT_MODE_RAG,
    Completion,
    CompletionClient,
    CompletionClientFactory,
    RemoteLLMConfig,
    RemoteLLMModel,
    TokenLogprob,
    UsageTotals,
    count_tokens,
    extract_token_logprobs,
    group_indices_by_size,
    parse_numbered_response,
    strip_code_fence,
)

EN = Language("en", "English")
ES = Language("es", "Spanish")


# --- dispatch ---------------------------------------------------------------------------


def test_is_remote_llm_config_requires_an_explicit_model_type():
    assert is_remote_llm_config({"model_type": "remote_llm", "model": "anthropic/claude-sonnet-4-5"})
    assert is_remote_llm_config({"model_type": "REMOTE_LLM", "model": "gpt-4o"})
    # A LiteLLM model string is arbitrary, so it is never recognized on its own.
    assert not is_remote_llm_config({"model": "anthropic/claude-sonnet-4-5"})
    assert not is_remote_llm_config({})


def test_an_remote_llm_config_is_not_claimed_by_the_llm_dispatch():
    # google/gemma matches LLM_MODEL_PREFIXES, but the explicit model_type has to win.
    config = {"model_type": "remote_llm", "model": "google/gemma-2-2b-it"}
    assert is_remote_llm_config(config)
    assert not is_llm_config(config)


# --- response parsing -------------------------------------------------------------------


def test_parse_numbered_response_reads_one_translation_per_line():
    assert parse_numbered_response("1. uno\n2. dos\n3. tres", 3) == ["uno", "dos", "tres"]


@pytest.mark.parametrize("delimiter", [".", ")", ":", "]"])
def test_parse_numbered_response_accepts_common_delimiters(delimiter: str):
    assert parse_numbered_response(f"1{delimiter} uno\n2{delimiter} dos", 2) == ["uno", "dos"]


def test_parse_numbered_response_ignores_preamble_and_reorders():
    assert parse_numbered_response("Certainly! Here you go:\n\n2. dos\n1. uno", 2) == ["uno", "dos"]


def test_parse_numbered_response_strips_code_fences():
    assert parse_numbered_response("```text\n1. uno\n2. dos\n```", 2) == ["uno", "dos"]


def test_parse_numbered_response_treats_unnumbered_lines_as_continuations():
    assert parse_numbered_response("1. uno\nand more\n2. dos", 2) == ["uno and more", "dos"]


def test_parse_numbered_response_rejects_a_miscount():
    assert parse_numbered_response("1. uno\n2. dos", 3) is None
    assert parse_numbered_response("1. uno\n2. dos\n3. tres", 2) is None


def test_parse_numbered_response_rejects_gaps_and_duplicates():
    assert parse_numbered_response("1. uno\n3. tres", 2) is None
    assert parse_numbered_response("1. uno\n1. otro", 2) is None


def test_parse_numbered_response_rejects_unnumbered_prose():
    assert parse_numbered_response("uno dos tres", 3) is None


def test_strip_code_fence_leaves_unfenced_text_alone():
    assert strip_code_fence("plain text") == "plain text"
    assert strip_code_fence("```\nfenced\n```") == "fenced"


# --- batching ---------------------------------------------------------------------------


def test_group_indices_by_size():
    assert group_indices_by_size(5, 2) == [[0, 1], [2, 3], [4]]
    assert group_indices_by_size(3, 1) == [[0], [1], [2]]
    assert group_indices_by_size(0, 4) == []
    # A nonsensical size still yields usable batches rather than an empty or infinite grouping.
    assert group_indices_by_size(3, 0) == [[0], [1], [2]]


# --- retrieval --------------------------------------------------------------------------


PAIRS = [
    ExamplePair("In the beginning God created the heavens and the earth.", "target one"),
    ExamplePair("And God said, Let there be light, and there was light.", "target two"),
    ExamplePair("Jesus wept.", "target three"),
]


def test_tokenize_for_retrieval_lowercases_and_splits_on_word_characters():
    assert tokenize_for_retrieval("Let there be LIGHT!") == ["let", "there", "be", "light"]


def make_retriever(method: str) -> ExampleRetriever:
    if method == "bm25":
        pytest.importorskip("rank_bm25")
    return create_retriever(method)


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_retriever_ranks_the_most_similar_example_first(method: str):
    retriever = make_retriever(method)
    retriever.fit(PAIRS)
    assert retriever.retrieve("let there be light", 1) == [PAIRS[1]]


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_retriever_returns_at_most_the_whole_corpus(method: str):
    retriever = make_retriever(method)
    retriever.fit(PAIRS)
    assert len(retriever.retrieve("light", 99)) == len(PAIRS)
    assert retriever.retrieve("light", 0) == []


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_retriever_handles_an_empty_corpus(method: str):
    retriever = make_retriever(method)
    retriever.fit([])
    assert retriever.retrieve("anything", 5) == []


def test_retriever_classes_report_their_method():
    assert TfidfExampleRetriever.method == "tfidf"
    assert BM25ExampleRetriever.method == "bm25"


def test_retriever_save_and_load_roundtrip(tmp_path: Path):
    retriever = create_retriever("tfidf")
    retriever.fit(PAIRS)
    retriever.save(tmp_path)

    loaded = ExampleRetriever.load(tmp_path)
    assert loaded is not None
    assert loaded.method == "tfidf"
    assert loaded.pairs == PAIRS
    assert loaded.retrieve("let there be light", 1) == [PAIRS[1]]


def test_retriever_load_returns_none_when_missing(tmp_path: Path):
    assert ExampleRetriever.load(tmp_path) is None


def test_retriever_load_returns_none_for_a_corrupt_index(tmp_path: Path):
    # An unreadable index is a signal to rebuild, not an error: it can be left behind by a
    # scikit-learn upgrade.
    (tmp_path / "retrieval.pkl").write_bytes(b"not a pickle")
    assert ExampleRetriever.load(tmp_path) is None


def test_retriever_load_returns_none_for_an_unrelated_pickle(tmp_path: Path):
    with (tmp_path / "retrieval.pkl").open("wb") as file:
        pickle.dump({"not": "a retriever"}, file)
    assert ExampleRetriever.load(tmp_path) is None


def test_create_retriever_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown retrieval method"):
        create_retriever("embeddings")


# --- config -----------------------------------------------------------------------------


def make_config(exp_dir: Path, **overrides) -> RemoteLLMConfig:
    config: dict = {
        "model_type": "remote_llm",
        "model": "gpt-4o",
        "data": {"corpus_pairs": [], "lang_codes": {"en": "English", "es": "Spanish"}},
    }
    for key, value in overrides.items():
        section = config.setdefault(key, {})
        if isinstance(section, dict) and isinstance(value, dict):
            section.update(value)
        else:
            config[key] = value
    return RemoteLLMConfig(exp_dir, config, Mock())


def test_config_defaults(tmp_path: Path):
    config = make_config(tmp_path)
    assert config.context_mode == CONTEXT_MODE_RAG
    assert config.retrieval_method == "tfidf"
    assert config.num_examples == 10
    assert config.infer_batch_size == 1
    # The hosted model tokenizes for itself, so preprocessing writes raw text.
    assert config.data["tokenize"] is False
    assert config.model_dir == tmp_path / "run"


def test_config_requires_a_model(tmp_path: Path):
    with pytest.raises(ValueError, match="LiteLLM format"):
        RemoteLLMConfig(tmp_path, {"model_type": "remote_llm", "model": "", "data": {"corpus_pairs": []}}, Mock())


@pytest.mark.parametrize(
    "infer, message",
    [
        ({"context_mode": "magic"}, "Unknown infer.context_mode"),
        ({"retrieval": {"method": "embeddings"}}, "Unknown infer.retrieval.method"),
        ({"infer_batch_size": 0}, "infer.infer_batch_size"),
        ({"num_drafts": 0}, "infer.num_drafts"),
        ({"concurrency": 0}, "infer.concurrency"),
        ({"retrieval": {"num_examples": -1}}, "num_examples"),
    ],
)
def test_config_validation(tmp_path: Path, infer: dict, message: str):
    with pytest.raises(ValueError, match=message):
        make_config(tmp_path, infer=infer)


def test_language_falls_back_to_the_iso_code(tmp_path: Path):
    config = make_config(tmp_path)
    assert config.language("en") == EN
    assert config.language("xyz") == Language("xyz", "xyz")


def test_create_tokenizer_is_a_no_op(tmp_path: Path):
    from silnlp.common.utils import Side

    tokenizer = make_config(tmp_path).create_tokenizer()
    assert tokenizer.tokenize(Side.SOURCE, "unchanged text") == "unchanged text"
    assert tokenizer.detokenize("unchanged text") == "unchanged text"


# --- prompts ----------------------------------------------------------------------------


def test_single_segment_prompt_has_no_numbering(tmp_path: Path):
    config = make_config(tmp_path)
    messages = config.build_messages(["hello"], [], EN, ES)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "English" in messages[0]["content"] and "Spanish" in messages[0]["content"]
    assert messages[1]["content"].endswith("hello")
    assert "1. hello" not in messages[1]["content"]


def test_batch_prompt_numbers_the_segments(tmp_path: Path):
    config = make_config(tmp_path)
    user = config.build_messages(["one", "two", "three"], [], EN, ES)[1]["content"]

    assert "1. one\n2. two\n3. three" in user
    assert "exactly 3 lines" in user


def test_system_message_casts_the_model_as_a_team_member(tmp_path: Path):
    # Consistency with one project's decisions is the point, so the examples have to be
    # authoritative rather than the prompt asking for a generically good translation.
    system = make_config(tmp_path).build_messages(["hello"], [], EN, ES)[0]["content"]

    assert "Bible translation team" in system
    assert "not a translation of your own" in system
    assert "your authority" in system


@pytest.mark.parametrize("dimension", ["Style", "Key terms", "Exegesis", "Orthography"])
def test_system_message_names_what_to_infer_from_the_examples(tmp_path: Path, dimension: str):
    system = make_config(tmp_path).build_messages(["hello"], [], EN, ES)[0]["content"]
    assert dimension in system


def test_system_message_prefers_the_examples_over_a_remembered_translation(tmp_path: Path):
    # A model asked for a well-known verse will otherwise reproduce a published version it has
    # memorized, which is exactly the wrong output for a team with its own conventions.
    system = make_config(tmp_path).build_messages(["hello"], [], EN, ES)[0]["content"]
    assert "in preference to any published Spanish translation you may recall" in system


def test_examples_are_presented_as_the_team_own_work(tmp_path: Path):
    config = make_config(tmp_path)
    user = config.build_messages(["hello"], [ExamplePair("greeting", "saludo")], EN, ES)[1]["content"]
    assert "The team has already translated these passages" in user


def test_full_corpus_block_is_presented_as_the_team_own_work(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"context_mode": CONTEXT_MODE_FULL_CORPUS})
    write_training_corpus(tmp_path)
    list(model.translate(["anything"], "en", "es"))

    assert "everything the team has translated so far" in client.calls[0][0]["content"]


def test_batch_prompt_tells_the_model_to_read_the_passages_together(tmp_path: Path):
    # Chapter batching exists so that the model can see a passage as a unit; the prompt has to
    # actually ask it to use that context for participants and pronouns.
    user = make_config(tmp_path).build_messages(["one", "two"], [], EN, ES)[1]["content"]
    assert "consecutive" in user
    assert "translate each one on its own" in user


def test_prompt_includes_retrieved_examples(tmp_path: Path):
    config = make_config(tmp_path)
    user = config.build_messages(["hello"], [ExamplePair("greeting", "saludo")], EN, ES)[1]["content"]

    assert "English: greeting" in user
    assert "Spanish: saludo" in user


def test_example_template_is_configurable(tmp_path: Path):
    config = make_config(tmp_path, params={"prompt": {"example_template": "{source} => {target}"}})
    user = config.build_messages(["hello"], [ExamplePair("greeting", "saludo")], EN, ES)[1]["content"]
    assert "greeting => saludo" in user


def test_system_message_is_configurable(tmp_path: Path):
    config = make_config(tmp_path, params={"prompt": {"system_message": "Be terse."}})
    assert config.build_messages(["hello"], [], EN, ES)[0]["content"] == "Be terse."


def test_full_corpus_block_goes_in_the_system_message(tmp_path: Path):
    # The corpus has to sit ahead of everything that varies per request, so that the prompt
    # prefix is byte-identical across requests and provider-side caching can apply.
    config = make_config(tmp_path)
    corpus_block = "CORPUS"
    first = config.build_messages(["one"], [], EN, ES, corpus_block)
    second = config.build_messages(["two"], [], EN, ES, corpus_block)

    assert first[0]["content"] == second[0]["content"]
    assert corpus_block in first[0]["content"]
    assert corpus_block not in first[1]["content"]


# --- model ------------------------------------------------------------------------------

_NUMBERED_SOURCE = re.compile(r"^\d+\. ")


class ScriptedClient(CompletionClient):
    """A completion client that answers with a scripted function instead of a network call."""

    def __init__(
        self,
        responder: Callable[[List[Dict[str, str]]], str],
        logprobs_supported: bool = False,
        token_logprobs: Optional[List[TokenLogprob]] = None,
    ) -> None:
        self._responder = responder
        self._logprobs_supported = logprobs_supported
        self._token_logprobs = token_logprobs
        self.calls: List[List[Dict[str, str]]] = []
        self.logprobs_requested: List[bool] = []

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        self.calls.append(messages)
        self.logprobs_requested.append(logprobs)
        text = self._responder(messages)
        if logprobs and self._token_logprobs is not None:
            return Completion(text, list(self._token_logprobs))
        return Completion(text)

    def supports_logprobs(self) -> bool:
        return self._logprobs_supported


class ScriptedClientFactory(CompletionClientFactory):
    def __init__(self, client: CompletionClient) -> None:
        self._client = client

    def create(self, config: RemoteLLMConfig) -> CompletionClient:
        return self._client


def echo_translations(messages: List[Dict[str, str]]) -> str:
    """Reply in the requested format, for however many segments the prompt asked about."""
    user = messages[-1]["content"]
    num_segments = sum(1 for line in user.splitlines() if _NUMBERED_SOURCE.match(line))
    if num_segments == 0:
        return "translated"
    return "\n".join(f"{i}. translated {i}" for i in range(1, num_segments + 1))


def make_model(
    tmp_path: Path,
    responder,
    logprobs_supported: bool = False,
    token_logprobs: Optional[List[TokenLogprob]] = None,
    **overrides,
) -> Tuple[RemoteLLMModel, ScriptedClient]:
    client = ScriptedClient(responder, logprobs_supported, token_logprobs)
    config = make_config(tmp_path, **overrides)
    return RemoteLLMModel(config, ScriptedClientFactory(client)), client


def translations_of(groups, draft: int = 0) -> List[str]:
    return [list(group)[draft].get_translation() for group in groups]


def test_translate_yields_one_group_per_sentence_in_order(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola")
    groups = list(model.translate(["one", "two", "three"], "en", "es"))

    assert len(groups) == 3
    assert translations_of(groups) == ["hola", "hola", "hola"]


def test_translate_batches_several_segments_into_one_request(tmp_path: Path):
    model, client = make_model(tmp_path, echo_translations, infer={"infer_batch_size": 3, "concurrency": 1})
    groups = list(model.translate(["one", "two", "three"], "en", "es"))

    assert len(client.calls) == 1
    assert translations_of(groups) == ["translated 1", "translated 2", "translated 3"]


def test_translate_leaves_blank_segments_blank_without_a_request(tmp_path: Path):
    model, client = make_model(tmp_path, echo_translations, infer={"infer_batch_size": 4, "concurrency": 1})
    groups = list(model.translate(["", "  ", ""], "en", "es"))

    assert translations_of(groups) == ["", "", ""]
    assert client.calls == []


def test_translate_keeps_blank_segments_aligned_within_a_batch(tmp_path: Path):
    model, _ = make_model(tmp_path, echo_translations, infer={"infer_batch_size": 3, "concurrency": 1})
    groups = list(model.translate(["one", "", "three"], "en", "es"))

    # The blank segment is not sent, so the two real segments are numbered 1 and 2.
    assert translations_of(groups) == ["translated 1", "", "translated 2"]


def test_translate_falls_back_to_single_requests_when_the_reply_is_malformed(tmp_path: Path):
    # The recovery ladder is: correct, split the batch, then one request per segment, which
    # cannot be miscounted.
    model, client = make_model(
        tmp_path, lambda messages: "I translated everything for you!", infer={"infer_batch_size": 2, "concurrency": 1}
    )
    groups = list(model.translate(["one", "two"], "en", "es"))

    assert translations_of(groups) == ["I translated everything for you!"] * 2
    # batch attempt, corrective retry, then one request per segment
    assert len(client.calls) == 4
    assert client.calls[1][-1]["content"].startswith("That reply did not have the required format")


def test_translate_recovers_after_a_corrective_retry(tmp_path: Path):
    replies = iter(["nonsense", "1. uno\n2. dos"])
    model, client = make_model(
        tmp_path, lambda messages: next(replies), infer={"infer_batch_size": 2, "concurrency": 1}
    )
    groups = list(model.translate(["one", "two"], "en", "es"))

    assert translations_of(groups) == ["uno", "dos"]
    assert len(client.calls) == 2


def test_multiple_translations_make_one_request_per_draft(tmp_path: Path):
    model, client = make_model(
        tmp_path, lambda messages: "hola", infer={"num_drafts": 3, "temperature": 0.8, "concurrency": 1}
    )
    groups = list(model.translate(["one"], "en", "es", produce_multiple_translations=True))

    assert len(groups) == 1
    assert groups[0].num_drafts == 3
    assert len(client.calls) == 3


def test_a_single_draft_is_produced_when_multiple_translations_are_not_requested(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"num_drafts": 3, "concurrency": 1})
    groups = list(model.translate(["one"], "en", "es"))

    assert groups[0].num_drafts == 1
    assert len(client.calls) == 1


def test_concurrent_requests_keep_the_translations_in_order(tmp_path: Path):
    model, _ = make_model(tmp_path, echo_translations, infer={"infer_batch_size": 1, "concurrency": 4})
    groups = list(model.translate([f"segment {i}" for i in range(20)], "en", "es"))

    assert translations_of(groups) == ["translated"] * 20


def test_translations_carry_no_confidence_scores_when_none_were_requested(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola")
    translation = list(list(model.translate(["one"], "en", "es"))[0])[0]

    assert not translation.has_sequence_confidence_score()
    assert client.logprobs_requested == [False]
    # The text has to survive the test-file path, which drops a leading special token for
    # seq2seq models but must not for this one.
    assert translation.join_tokens_for_test_file() == "hola"


def test_translate_test_files_writes_one_line_per_source(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\ntwo\n", encoding="utf-8")
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"concurrency": 1})

    model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"])

    assert (tmp_path / "out.txt").read_text(encoding="utf-8").splitlines() == ["hola", "hola"]


def test_translate_test_files_writes_one_file_per_draft(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\n", encoding="utf-8")
    model, _ = make_model(
        tmp_path, lambda messages: "hola", infer={"num_drafts": 2, "temperature": 0.8, "concurrency": 1}
    )

    model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"], produce_multiple_translations=True)

    assert (tmp_path / "out.1.txt").is_file()
    assert (tmp_path / "out.2.txt").is_file()


# --- training ---------------------------------------------------------------------------


def write_training_corpus(exp_dir: Path) -> None:
    (exp_dir / "train.src.detok.txt").write_text("in the beginning\nlet there be light\n", encoding="utf-8")
    (exp_dir / "train.trg.detok.txt").write_text("en el principio\nsea la luz\n", encoding="utf-8")


def test_train_builds_and_saves_the_retrieval_index(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, _ = make_model(tmp_path, lambda messages: "hola")

    model.train()

    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    assert (checkpoint_dir / "retrieval.pkl").is_file()
    loaded = ExampleRetriever.load(checkpoint_dir)
    assert loaded is not None and len(loaded.pairs) == 2


def test_train_writes_a_checkpoint_so_the_last_checkpoint_resolves(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, _ = make_model(tmp_path, lambda messages: "hola")

    model.train()

    path, step = model.get_checkpoint_path("last")
    assert step == 1
    assert path == tmp_path / "run" / "checkpoint-1"


def test_train_in_full_corpus_mode_writes_a_checkpoint_but_no_index(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"context_mode": CONTEXT_MODE_FULL_CORPUS})

    model.train()

    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    assert checkpoint_dir.is_dir()
    assert not (checkpoint_dir / "retrieval.pkl").exists()


def test_train_tolerates_a_missing_training_corpus(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola")
    model.train()
    assert (tmp_path / "run" / "checkpoint-1").is_dir()


def test_retrieved_examples_reach_the_prompt(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"retrieval": {"num_examples": 1}})
    model.train()

    list(model.translate(["let there be light"], "en", "es"))

    user = client.calls[0][1]["content"]
    assert "let there be light" in user
    assert "sea la luz" in user


def test_the_index_is_rebuilt_when_the_checkpoint_is_gone(tmp_path: Path):
    # experiment.py deletes the run directory unless --save-checkpoints is passed, so the saved
    # index is only a cache; inference has to rebuild it from the training corpus.
    write_training_corpus(tmp_path)
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"retrieval": {"num_examples": 1}})

    list(model.translate(["let there be light"], "en", "es"))

    assert "sea la luz" in client.calls[0][1]["content"]


def test_full_corpus_mode_puts_the_whole_corpus_in_the_system_message(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"context_mode": CONTEXT_MODE_FULL_CORPUS})

    list(model.translate(["anything"], "en", "es"))

    system = client.calls[0][0]["content"]
    assert "en el principio" in system
    assert "sea la luz" in system


def test_save_effective_config_writes_the_merged_config(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola")
    path = tmp_path / "effective-config.yml"
    model.save_effective_config(path)

    with path.open(encoding="utf-8") as file:
        written = yaml.safe_load(file)
    assert written["model"] == "gpt-4o"
    assert written["infer"]["context_mode"] == CONTEXT_MODE_RAG


# --- confidence scores from log probabilities ---------------------------------------------


SCORES = [TokenLogprob("ho", -0.2), TokenLogprob("la", -0.4)]


def test_completion_mean_logprob():
    assert Completion("hola", SCORES).mean_logprob() == pytest.approx(-0.3)
    assert Completion("hola").mean_logprob() is None


def test_extract_token_logprobs_reads_the_openai_shape():
    choice = {"logprobs": {"content": [{"token": "ho", "logprob": -0.2}, {"token": "la", "logprob": -0.4}]}}
    assert extract_token_logprobs(choice) == SCORES


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_extract_token_logprobs_reads_pydantic_style_objects():
    # LiteLLM returns model objects rather than plain dicts for most providers.
    choice = _Obj(logprobs=_Obj(content=[_Obj(token="ho", logprob=-0.2), _Obj(token="la", logprob=-0.4)]))
    assert extract_token_logprobs(choice) == SCORES


@pytest.mark.parametrize(
    "choice",
    [
        {},
        {"logprobs": None},
        {"logprobs": {"content": None}},
        {"logprobs": {"content": []}},
    ],
)
def test_extract_token_logprobs_tolerates_a_provider_that_omits_them(choice: dict):
    # Providers that do not support logprobs silently omit them rather than failing.
    assert extract_token_logprobs(choice) == []


def test_extract_token_logprobs_skips_incomplete_entries():
    choice = {"logprobs": {"content": [{"token": "ho"}, {"token": "la", "logprob": -0.4}]}}
    assert extract_token_logprobs(choice) == [TokenLogprob("la", -0.4)]


def test_confidence_scores_come_from_the_token_logprobs(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\n", encoding="utf-8")
    model, client = make_model(
        tmp_path,
        lambda messages: "hola",
        logprobs_supported=True,
        token_logprobs=SCORES,
        infer={"concurrency": 1},
    )

    model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"], save_confidences=True)

    assert client.logprobs_requested == [True]
    confidences = tmp_path / "out.txt.confidences.tsv"
    assert confidences.is_file()
    rows = confidences.read_text(encoding="utf-8").splitlines()
    # Two header rows, then one token row and one score row for the single sentence.
    assert len(rows) == 4
    # The score row leads with the exponentiated mean log probability.
    assert float(rows[3].split("\t")[0]) == pytest.approx(math.exp(-0.3))


def test_confidence_scores_do_not_corrupt_the_predictions_file(tmp_path: Path):
    # The predictions file is raw text, so it must hold the translation, not the provider's
    # subword tokens.
    (tmp_path / "test.src.txt").write_text("one\n", encoding="utf-8")
    model, _ = make_model(
        tmp_path,
        lambda messages: "hola mundo",
        logprobs_supported=True,
        token_logprobs=SCORES,
        infer={"concurrency": 1},
    )

    model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"], save_confidences=True)

    assert (tmp_path / "out.txt").read_text(encoding="utf-8").splitlines() == ["hola mundo"]


def test_confidences_are_rejected_in_chapter_mode(tmp_path: Path):
    model, client = make_model(
        tmp_path,
        lambda messages: "hola",
        logprobs_supported=True,
        token_logprobs=SCORES,
        infer={"infer_batch_size": 4},
    )

    with pytest.raises(RuntimeError, match="single segment"):
        model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"], save_confidences=True)
    # It fails before spending anything on inference.
    assert client.calls == []


def test_confidences_are_rejected_when_batches_hold_several_segments(tmp_path: Path):
    model, _ = make_model(
        tmp_path,
        lambda messages: "hola",
        logprobs_supported=True,
        token_logprobs=SCORES,
        infer={"infer_batch_size": 4},
    )

    with pytest.raises(RuntimeError, match="single segment"):
        model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"], save_confidences=True)


def test_confidences_are_rejected_when_the_provider_has_no_logprobs(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", logprobs_supported=False)

    with pytest.raises(RuntimeError, match="does not return token log probabilities"):
        model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"], save_confidences=True)


def test_scores_are_dropped_when_a_code_fence_is_stripped(tmp_path: Path):
    # Stripping the fence changes the text, so the per-token scores no longer line up with it.
    model, _ = make_model(
        tmp_path,
        lambda messages: "```\nhola\n```",
        logprobs_supported=True,
        token_logprobs=SCORES,
        infer={"concurrency": 1},
    )

    translation = list(list(model.translate(["one"], "en", "es"))[0])[0]
    assert translation.get_translation() == "hola"
    assert not translation.has_sequence_confidence_score()


def test_an_empty_system_message_is_omitted_rather_than_sent_blank(tmp_path: Path):
    config = make_config(tmp_path, params={"prompt": {"system_message": ""}})
    messages = config.build_messages(["hello"], [], EN, ES)
    assert [message["role"] for message in messages] == ["user"]


def test_the_effective_config_records_the_prompts_actually_used(tmp_path: Path):
    # The prompt is the model here, so a run is only reproducible if the saved config has it.
    model, _ = make_model(tmp_path, lambda messages: "hola")
    path = tmp_path / "effective-config.yml"
    model.save_effective_config(path)

    prompt = yaml.safe_load(path.read_text(encoding="utf-8"))["params"]["prompt"]
    assert "Bible translation team" in prompt["system_message"]
    assert "{source}" in prompt["instruction_template"]
    assert "{num_segments}" in prompt["batch_instruction_template"]


# --- token counting -----------------------------------------------------------------------


def test_count_tokens_uses_the_provider_tokenizer():
    pytest.importorskip("litellm")
    text = "In the beginning God created the heavens and the earth. " * 20
    tokens = count_tokens("gpt-4o", text)
    assert tokens is not None
    # A real tokenization, not the character-count approximation it replaced.
    assert 150 < tokens < len(text) // 4


def test_count_tokens_falls_back_to_none_for_an_unusable_model():
    pytest.importorskip("litellm")
    # LiteLLM tokenizes unknown models with a default tokenizer rather than failing, so this
    # asserts the contract (an int or None) rather than a specific outcome.
    assert count_tokens("made-up/nonexistent", "hello") in (None, 1, 2)


def test_full_corpus_mode_warns_when_the_corpus_exceeds_the_context_limit(tmp_path: Path, caplog):
    pytest.importorskip("litellm")
    write_training_corpus(tmp_path)
    model, _ = make_model(
        tmp_path,
        lambda messages: "hola",
        infer={"context_mode": CONTEXT_MODE_FULL_CORPUS, "max_context_tokens": 1},
    )

    with caplog.at_level("WARNING"):
        list(model.translate(["anything"], "en", "es"))

    assert "exceeds infer.max_context_tokens" in caplog.text


def test_full_corpus_mode_is_quiet_when_the_corpus_fits(tmp_path: Path, caplog):
    pytest.importorskip("litellm")
    write_training_corpus(tmp_path)
    model, _ = make_model(
        tmp_path,
        lambda messages: "hola",
        infer={"context_mode": CONTEXT_MODE_FULL_CORPUS, "max_context_tokens": 100000},
    )

    with caplog.at_level("WARNING"):
        list(model.translate(["anything"], "en", "es"))

    assert "exceeds infer.max_context_tokens" not in caplog.text


def test_count_tokens_handles_an_unrecognized_model():
    pytest.importorskip("litellm")
    # LiteLLM tokenizes with a default tokenizer rather than failing on an unknown model.
    assert count_tokens("made-up/nonexistent", "In the beginning God created the heavens.") is not None


# --- usage and cost reporting ---------------------------------------------------------------


def test_usage_totals_accumulate():
    totals = UsageTotals()
    totals.add(Completion("a", [], prompt_tokens=100, completion_tokens=10, cost=0.01))
    totals.add(Completion("b", [], prompt_tokens=200, completion_tokens=20, cost=0.02))

    assert (totals.requests, totals.prompt_tokens, totals.completion_tokens) == (2, 300, 30)
    assert totals.cost == pytest.approx(0.03)
    assert "2 requests, 300 prompt + 30 completion tokens, $0.0300" == totals.describe()


def test_usage_totals_report_an_unpriced_model_rather_than_calling_it_free():
    totals = UsageTotals()
    totals.add(Completion("a", [], prompt_tokens=100, completion_tokens=10, cost=None))
    assert "cost unavailable" in totals.describe()
    assert "$" not in totals.describe()


def test_usage_totals_flag_a_partial_cost():
    totals = UsageTotals()
    totals.add(Completion("a", [], prompt_tokens=100, completion_tokens=10, cost=0.01))
    totals.add(Completion("b", [], prompt_tokens=100, completion_tokens=10, cost=None))

    assert "$0.0100 excluding 1 unpriced requests" in totals.describe()


def test_usage_totals_are_thread_safe():
    totals = UsageTotals()
    completion = Completion("a", [], prompt_tokens=1, completion_tokens=1, cost=0.001)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: totals.add(completion), range(2000)))

    assert totals.requests == 2000
    assert totals.prompt_tokens == 2000


def test_translating_logs_the_usage_and_cost(tmp_path: Path, caplog):
    def priced(messages):
        return "hola"

    client = ScriptedClient(priced)
    # The scripted client reports no usage, so patch in a priced reply.
    client.complete = lambda messages, logprobs=False: Completion(  # type: ignore[method-assign]
        "hola", [], prompt_tokens=500, completion_tokens=5, cost=0.002
    )
    config = make_config(tmp_path, infer={"concurrency": 1})
    model = RemoteLLMModel(config, ScriptedClientFactory(client))

    with caplog.at_level("INFO"):
        list(model.translate(["one", "two"], "en", "es"))

    assert "Translated 2 segments using 2 requests" in caplog.text
    assert "1,000 prompt + 10 completion tokens" in caplog.text
    assert "$0.0040" in caplog.text
