import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock

import pytest
import yaml

from silnlp.nmt.config import Language
from silnlp.nmt.config_utils import is_local_llm_config, is_remote_llm_config
from silnlp.nmt.example_retrieval import Example
from silnlp.nmt.remote_llm_config import (
    Completion,
    CompletionClient,
    CompletionClientFactory,
    CompletionSettings,
    RemoteLLMConfig,
    RemoteLLMModel,
    TokenLogprob,
    UsageTotals,
    LiteLLMCompletionClient,
    LiteLLMResponse,
    ModelReply,
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
    assert not is_local_llm_config(config)


# --- response parsing -------------------------------------------------------------------


def test_parse_numbered_response_reads_one_translation_per_line():
    assert ModelReply("1. uno\n2. dos\n3. tres").parse(3) == ["uno", "dos", "tres"]


@pytest.mark.parametrize("delimiter", [".", ")", ":", "]"])
def test_parse_numbered_response_accepts_common_delimiters(delimiter: str):
    assert ModelReply(f"1{delimiter} uno\n2{delimiter} dos").parse(2) == ["uno", "dos"]


def test_parse_numbered_response_ignores_preamble_and_reorders():
    assert ModelReply("Certainly! Here you go:\n\n2. dos\n1. uno").parse(2) == ["uno", "dos"]


def test_parse_numbered_response_strips_code_fences():
    assert ModelReply("```text\n1. uno\n2. dos\n```").parse(2) == ["uno", "dos"]


def test_parse_numbered_response_treats_unnumbered_lines_as_continuations():
    assert ModelReply("1. uno\nand more\n2. dos").parse(2) == ["uno and more", "dos"]


def test_parse_numbered_response_rejects_a_miscount():
    assert ModelReply("1. uno\n2. dos").parse(3) is None
    assert ModelReply("1. uno\n2. dos\n3. tres").parse(2) is None


def test_parse_numbered_response_rejects_gaps_and_duplicates():
    assert ModelReply("1. uno\n3. tres").parse(2) is None
    assert ModelReply("1. uno\n1. otro").parse(2) is None


def test_parse_numbered_response_rejects_unnumbered_prose():
    assert ModelReply("uno dos tres").parse(3) is None


def test_strip_code_fence_leaves_unfenced_text_alone():
    assert ModelReply("plain text").strip_code_fence() == "plain text"
    assert ModelReply("```\nfenced\n```").strip_code_fence() == "fenced"


# --- batching ---------------------------------------------------------------------------


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
    assert config.get_prompt()["example_selection"]["method"] == "tfidf"
    assert config.get_prompt()["num_examples"] == 10
    assert config.get_infer_batch_size() == 1
    # The hosted model tokenizes for itself, so preprocessing writes raw text.
    assert config.data["tokenize"] is False
    assert config.model_dir == tmp_path / "run"


def test_config_requires_a_model(tmp_path: Path):
    with pytest.raises(ValueError, match="LiteLLM format"):
        RemoteLLMConfig(tmp_path, {"model_type": "remote_llm", "model": "", "data": {"corpus_pairs": []}}, Mock())


@pytest.mark.parametrize(
    "infer, message",
    [
        ({"prompt": {"example_selection": {"method": "embeddings"}}}, "Unknown example_selection.method"),
        ({"infer_batch_size": 0}, "infer.get_infer_batch_size()"),
        ({"num_drafts": 0}, "infer.num_drafts"),
        ({"concurrency": 0}, "infer.concurrency"),
        ({"prompt": {"num_examples": -1}}, "num_examples"),
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
    user = config.build_messages(["hello"], [Example("greeting", "saludo")], EN, ES)[1]["content"]
    assert "The team has already translated these passages" in user


def test_full_corpus_block_is_presented_as_the_team_own_work(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1000000}})
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
    user = config.build_messages(["hello"], [Example("greeting", "saludo")], EN, ES)[1]["content"]

    assert "English: greeting" in user
    assert "Spanish: saludo" in user


def test_example_template_is_configurable(tmp_path: Path):
    config = make_config(
        tmp_path, infer={"prompt": {"example_format": {"type": "text", "template": "{source} => {target}"}}}
    )
    user = config.build_messages(["hello"], [Example("greeting", "saludo")], EN, ES)[1]["content"]
    assert "greeting => saludo" in user


def test_system_message_is_configurable(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"system_message": "Be terse."}})
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
        tokens_per_word: Optional[int] = 1,
    ) -> None:
        self._responder = responder
        self._logprobs_supported = logprobs_supported
        self._token_logprobs = token_logprobs
        self._tokens_per_word = tokens_per_word
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

    def count_tokens(self, text: str) -> Optional[int]:
        return None if self._tokens_per_word is None else len(text.split()) * self._tokens_per_word


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
    with_corpus: bool = True,
    tokens_per_word: Optional[int] = 1,
    **overrides,
) -> Tuple[RemoteLLMModel, ScriptedClient]:
    if with_corpus and not (tmp_path / "train.src.txt").is_file():
        write_training_corpus(tmp_path)
    client = ScriptedClient(responder, logprobs_supported, token_logprobs, tokens_per_word)
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
    (exp_dir / "train.src.txt").write_text("in the beginning\nlet there be light\n", encoding="utf-8")
    (exp_dir / "train.trg.txt").write_text("en el principio\nsea la luz\n", encoding="utf-8")


def test_train_records_the_retrieval_method_it_built(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1}})

    model.train()

    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    info = json.loads((checkpoint_dir / "remote_llm_model.json").read_text(encoding="utf-8"))
    assert info["retrieval_method"] == "tfidf"
    assert info["num_training_pairs"] == 2
    # A lexical index refits in seconds, so none is cached for inference to pick up.
    assert [path.name for path in checkpoint_dir.glob("retrieval*")] == []


def test_train_writes_a_checkpoint_so_the_last_checkpoint_resolves(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, _ = make_model(tmp_path, lambda messages: "hola")

    model.train()

    path, step = model.get_checkpoint_path("last")
    assert step == 1
    assert path == tmp_path / "run" / "checkpoint-1"


def test_train_in_full_corpus_mode_writes_a_checkpoint_but_no_index(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1000000}})

    model.train()

    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    assert checkpoint_dir.is_dir()
    info = json.loads((checkpoint_dir / "remote_llm_model.json").read_text(encoding="utf-8"))
    # Every example goes in the prompt, so no retrieval happens at all.
    assert "retrieval_method" not in info
    assert info["corpus_tokens"] is not None


def test_train_rejects_a_missing_training_corpus(tmp_path: Path):
    # Prompting with no examples when examples were asked for is a silent quality loss, so it
    # fails here rather than after a run's worth of requests has been paid for.
    model, _ = make_model(tmp_path, lambda messages: "hola", with_corpus=False)
    with pytest.raises(RuntimeError, match="Run preprocessing"):
        model.train()


def test_translate_rejects_a_missing_training_corpus(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", with_corpus=False)
    with pytest.raises(RuntimeError, match="Run preprocessing"):
        list(model.translate(["anything"], "en", "es"))


def test_a_missing_training_corpus_is_fine_without_examples(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", with_corpus=False, infer={"prompt": {"num_examples": 0}})
    model.train()
    assert (tmp_path / "run" / "checkpoint-1").is_dir()
    assert translations_of(model.translate(["anything"], "en", "es")) == ["hola"]


def test_retrieved_examples_reach_the_prompt(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1}})
    model.train()

    list(model.translate(["let there be light"], "en", "es"))

    user = client.calls[0][1]["content"]
    assert "let there be light" in user
    assert "sea la luz" in user


def test_the_index_is_rebuilt_when_the_checkpoint_is_gone(tmp_path: Path):
    # experiment.py deletes the run directory unless --save-checkpoints is passed, so the saved
    # index is only a cache; inference has to rebuild it from the training corpus.
    write_training_corpus(tmp_path)
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1}})

    list(model.translate(["let there be light"], "en", "es"))

    assert "sea la luz" in client.calls[0][1]["content"]


def test_full_corpus_mode_puts_the_whole_corpus_in_the_system_message(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1000000}})

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
    assert written["infer"]["prompt"]["num_examples"] == 10


# --- confidence scores from log probabilities ---------------------------------------------


SCORES = [TokenLogprob("ho", -0.2), TokenLogprob("la", -0.4)]


def test_completion_mean_logprob():
    assert Completion("hola", SCORES).mean_logprob() == pytest.approx(-0.3)
    assert Completion("hola").mean_logprob() is None


def _response_with(choice) -> LiteLLMResponse:
    return LiteLLMResponse({"choices": [choice]}, litellm=None)


def test_token_logprobs_read_the_openai_shape():
    choice = {"logprobs": {"content": [{"token": "ho", "logprob": -0.2}, {"token": "la", "logprob": -0.4}]}}
    assert _response_with(choice).token_logprobs() == SCORES


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_token_logprobs_read_pydantic_style_objects():
    # LiteLLM returns model objects rather than plain dicts for most providers.
    choice = _Obj(logprobs=_Obj(content=[_Obj(token="ho", logprob=-0.2), _Obj(token="la", logprob=-0.4)]))
    assert _response_with(choice).token_logprobs() == SCORES


@pytest.mark.parametrize(
    "choice",
    [
        {},
        {"logprobs": None},
        {"logprobs": {"content": None}},
        {"logprobs": {"content": []}},
    ],
)
def test_token_logprobs_tolerate_a_provider_that_omits_them(choice: dict):
    # Providers that do not support logprobs silently omit them rather than failing.
    assert _response_with(choice).token_logprobs() == []


def test_token_logprobs_skip_incomplete_entries():
    choice = {"logprobs": {"content": [{"token": "ho"}, {"token": "la", "logprob": -0.4}]}}
    assert _response_with(choice).token_logprobs() == [TokenLogprob("la", -0.4)]


def test_response_reads_the_text_and_usage_from_either_shape():
    raw = {"choices": [{"message": {"content": "hola"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    assert LiteLLMResponse(raw, litellm=None).text() == "hola"
    assert LiteLLMResponse(raw, litellm=None).prompt_tokens() == 7

    obj = _Obj(choices=[_Obj(message=_Obj(content="hola"))], usage=_Obj(prompt_tokens=7, completion_tokens=3))
    assert LiteLLMResponse(obj, litellm=None).completion_tokens() == 3


def test_response_tolerates_a_missing_usage_block():
    raw = {"choices": [{"message": {"content": "hola"}}]}
    assert LiteLLMResponse(raw, litellm=None).prompt_tokens() == 0


def test_response_reports_an_unknown_cost_when_litellm_has_no_pricing():
    class _NoPricing:
        def completion_cost(self, completion_response):
            raise Exception("no pricing")

    assert LiteLLMResponse({"choices": []}, _NoPricing()).cost() is None


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
    config = make_config(tmp_path, infer={"prompt": {"system_message": ""}})
    messages = config.build_messages(["hello"], [], EN, ES)
    assert [message["role"] for message in messages] == ["user"]


def test_the_effective_config_records_the_prompts_actually_used(tmp_path: Path):
    # The prompt is the model here, so a run is only reproducible if the saved config has it.
    model, _ = make_model(tmp_path, lambda messages: "hola")
    path = tmp_path / "effective-config.yml"
    model.save_effective_config(path)

    prompt = yaml.safe_load(path.read_text(encoding="utf-8"))["infer"]["prompt"]
    assert "Bible translation team" in prompt["system_message"]
    assert "{source}" in prompt["instruction_template"]
    assert "{num_segments}" in prompt["batch_instruction_template"]


# --- token counting -----------------------------------------------------------------------


@pytest.mark.slow
def test_count_tokens_uses_the_provider_tokenizer():
    # LiteLLM fetches the encoding over the network, which takes longer than the rest of the suite.
    pytest.importorskip("litellm")
    text = "In the beginning God created the heavens and the earth. " * 20
    tokens = LiteLLMCompletionClient("gpt-4o", CompletionSettings(0.0, 16, 0, 10)).count_tokens(text)
    assert tokens is not None
    # A real tokenization, not the character-count approximation it replaced.
    assert 150 < tokens < len(text) // 4


@pytest.mark.slow
def test_count_tokens_falls_back_to_none_for_an_unusable_model():
    pytest.importorskip("litellm")
    # LiteLLM tokenizes unknown models with a default tokenizer rather than failing, so this
    # asserts the contract (an int or None) rather than a specific outcome.
    client = LiteLLMCompletionClient("made-up/nonexistent", CompletionSettings(0.0, 16, 0, 10))
    assert client.count_tokens("hello") in (None, 1, 2)


def test_full_corpus_mode_warns_when_the_corpus_exceeds_the_context_limit(tmp_path: Path, caplog):
    write_training_corpus(tmp_path)
    model, _ = make_model(
        tmp_path,
        lambda messages: "hola",
        infer={"prompt": {"num_examples": 1000000}, "max_context_tokens": 1},
    )

    with caplog.at_level("WARNING"):
        list(model.translate(["anything"], "en", "es"))

    assert "exceeds infer.max_context_tokens" in caplog.text


def test_full_corpus_mode_is_quiet_when_the_corpus_fits(tmp_path: Path, caplog):
    write_training_corpus(tmp_path)
    model, _ = make_model(
        tmp_path,
        lambda messages: "hola",
        infer={"prompt": {"num_examples": 1000000}, "max_context_tokens": 100000},
    )

    with caplog.at_level("WARNING"):
        list(model.translate(["anything"], "en", "es"))

    assert "exceeds infer.max_context_tokens" not in caplog.text


@pytest.mark.slow
def test_count_tokens_handles_an_unrecognized_model():
    pytest.importorskip("litellm")
    # LiteLLM tokenizes with a default tokenizer rather than failing on an unknown model.
    client = LiteLLMCompletionClient("made-up/nonexistent", CompletionSettings(0.0, 16, 0, 10))
    assert client.count_tokens("In the beginning God created the heavens.") is not None


def test_a_client_that_cannot_count_tokens_makes_the_caller_skip_the_size_check(tmp_path: Path, caplog):
    model, _ = make_model(
        tmp_path,
        lambda messages: "hola",
        infer={"prompt": {"num_examples": 1000000}, "max_context_tokens": 1},
        tokens_per_word=None,
    )

    with caplog.at_level("WARNING"):
        list(model.translate(["anything"], "en", "es"))

    assert "exceeds infer.max_context_tokens" not in caplog.text


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
    write_training_corpus(tmp_path)
    config = make_config(tmp_path, infer={"concurrency": 1})
    model = RemoteLLMModel(config, ScriptedClientFactory(client))

    with caplog.at_level("INFO"):
        list(model.translate(["one", "two"], "en", "es"))

    assert "Translated 2 segments using 2 requests" in caplog.text
    assert "1,000 prompt + 10 completion tokens" in caplog.text
    assert "$0.0040" in caplog.text


def test_a_hoisted_corpus_does_not_leave_a_dangling_examples_heading(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1000000}})
    write_training_corpus(tmp_path)
    list(model.translate(["anything"], "en", "es"))

    system, user = client.calls[0][0]["content"], client.calls[0][1]["content"]
    assert "everything the team has translated so far" in system
    assert "The team has already translated these passages" not in user


def test_a_hoisted_corpus_keeps_a_custom_instruction_template(tmp_path: Path):
    model, client = make_model(
        tmp_path,
        lambda messages: "hola",
        infer={"prompt": {"num_examples": 1000000, "instruction_template": "Mine: {source}"}},
    )
    write_training_corpus(tmp_path)
    list(model.translate(["anything"], "en", "es"))

    assert client.calls[0][1]["content"] == "Mine: anything"


def test_retrieved_examples_keep_their_heading_when_the_corpus_is_not_hoisted(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1}})
    write_training_corpus(tmp_path)
    list(model.translate(["let there be light"], "en", "es"))

    assert "The team has already translated these passages" in client.calls[0][1]["content"]


def test_the_batch_prompt_drops_its_examples_heading_when_the_corpus_is_hoisted(tmp_path: Path):
    model, client = make_model(
        tmp_path,
        lambda messages: "1. hola\n2. adios",
        infer={"prompt": {"num_examples": 1000000}, "infer_batch_size": 2},
    )
    write_training_corpus(tmp_path)
    list(model.translate(["one", "two"], "en", "es"))

    user = client.calls[0][1]["content"]
    assert "The team has already translated these passages" not in user
    assert "consecutive" in user


def test_the_remote_model_prefers_the_detokenized_corpus(tmp_path: Path):
    # An experiment preprocessed for a tokenized model leaves readable text only in the detok files.
    (tmp_path / "train.src.txt").write_text("let ▁there ▁be ▁light\n", encoding="utf-8")
    (tmp_path / "train.trg.txt").write_text("sea ▁la ▁luz\n", encoding="utf-8")
    (tmp_path / "train.src.detok.txt").write_text("let there be light\n", encoding="utf-8")
    (tmp_path / "train.trg.detok.txt").write_text("sea la luz\n", encoding="utf-8")

    config = make_config(tmp_path, infer={"prompt": {"num_examples": 1}})
    selected = config.get_infer_prompt_builder().select_examples("let there be light")
    assert [example.target for example in selected] == ["sea la luz"]


def test_the_remote_model_falls_back_to_the_plain_corpus(tmp_path: Path):
    write_training_corpus(tmp_path)
    config = make_config(tmp_path, infer={"prompt": {"num_examples": 1}})
    selected = config.get_infer_prompt_builder().select_examples("let there be light")
    assert [example.target for example in selected] == ["sea la luz"]


def test_the_example_index_serves_batched_requests(tmp_path: Path):
    write_training_corpus(tmp_path)
    model, client = make_model(
        tmp_path, lambda messages: "1. hola\n2. adios", infer={"prompt": {"num_examples": 1}, "infer_batch_size": 2}
    )
    model.train()

    list(model.translate(["let there be light", "and so on"], "en", "es"))

    # A batched request draws on the same indexed corpus as a single-segment one.
    assert "sea la luz" in client.calls[0][1]["content"]


def test_the_batch_builder_numbers_the_segments_and_states_the_count(tmp_path: Path):
    model, client = make_model(
        tmp_path, lambda messages: "1. uno\n2. dos\n3. tres", infer={"infer_batch_size": 3}
    )
    list(model.translate(["one", "two", "three"], "en", "es"))

    user = client.calls[0][1]["content"]
    assert "1. one\n2. two\n3. three" in user
    # The count is the contract ModelReply.parse checks the reply against.
    assert "exactly 3 lines" in user


def test_a_single_segment_request_does_not_use_the_batch_wording(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"infer_batch_size": 1})
    list(model.translate(["one"], "en", "es"))

    assert "consecutive" not in client.calls[0][1]["content"]
