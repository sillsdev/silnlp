import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock

import pytest
import yaml

import silnlp.nmt.test as test_step
from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config import Language
from silnlp.nmt.config_utils import is_local_llm_config, is_remote_llm_config, load_config
from silnlp.nmt.example_retrieval import Example, TargetLanguageProfile
from silnlp.nmt.llm_usage import LLMUsageStatistics
from silnlp.nmt.reasoning_estimate import (
    MeasuredReasoning,
    NoReasoning,
    ReasoningBudget,
    ReasoningEstimate,
    UnknownReasoning,
)
from silnlp.nmt.remote_llm_config import (
    BatchReplyReader,
    CodeFenceRemover,
    Completion,
    CompletionClient,
    CompletionClientFactory,
    CompletionSettings,
    DryRunRemoteLLMModel,
    LiteLLMCompletionClient,
    LiteLLMResponse,
    ReadOnlySavedReplies,
    RemoteLLMConfig,
    RemoteLLMModel,
    ReplyReusingClient,
    RetryingCompletionClient,
    RetryPolicy,
    SavedReplies,
    SingleReplyReader,
    TokenLogprob,
    TranslationLabelRemover,
    UsageTotals,
)
from silnlp.nmt.translate import TranslationTask

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


def test_batch_reply_reader_reads_one_translation_per_line():
    assert BatchReplyReader(3).read("1. uno\n2. dos\n3. tres") == ["uno", "dos", "tres"]


def test_batch_reply_reader_accepts_numbers_followed_by_a_parenthesis():
    assert BatchReplyReader(2).read("1) uno\n2) dos") == ["uno", "dos"]


def test_batch_reply_reader_accepts_numbers_followed_by_a_colon():
    assert BatchReplyReader(2).read("1: uno\n2: dos") == ["uno", "dos"]


def test_batch_reply_reader_accepts_numbers_followed_by_a_bracket():
    assert BatchReplyReader(2).read("1] uno\n2] dos") == ["uno", "dos"]


def test_batch_reply_reader_ignores_preamble_and_reorders():
    assert BatchReplyReader(2).read("Certainly! Here you go:\n\n2. dos\n1. uno") == ["uno", "dos"]


def test_batch_reply_reader_strips_code_fences():
    assert BatchReplyReader(2).read("```text\n1. uno\n2. dos\n```") == ["uno", "dos"]


def test_batch_reply_reader_treats_unnumbered_lines_as_continuations():
    assert BatchReplyReader(2).read("1. uno\nand more\n2. dos") == ["uno and more", "dos"]


def test_batch_reply_reader_rejects_a_miscount():
    assert BatchReplyReader(3).read("1. uno\n2. dos") is None
    assert BatchReplyReader(2).read("1. uno\n2. dos\n3. tres") is None


def test_batch_reply_reader_rejects_gaps_and_duplicates():
    assert BatchReplyReader(2).read("1. uno\n3. tres") is None
    assert BatchReplyReader(2).read("1. uno\n1. otro") is None


def test_batch_reply_reader_rejects_unnumbered_prose():
    assert BatchReplyReader(3).read("uno dos tres") is None


def test_batch_reply_reader_rejects_a_blank_entry():
    assert BatchReplyReader(3).read("1. uno\n2.\n3. tres") is None


def test_code_fence_leaves_unfenced_text_alone():
    assert CodeFenceRemover().remove_from("plain text") == "plain text"
    assert CodeFenceRemover().remove_from("```\nfenced\n```") == "fenced"


def test_translation_label_naming_the_target_language_is_removed():
    assert TranslationLabelRemover("Spanish").remove_from("Spanish: sea la luz") == "sea la luz"


def test_translation_label_is_removed_whatever_its_case_and_when_a_dash_follows_it():
    assert TranslationLabelRemover("Spanish").remove_from("SPANISH - sea la luz") == "sea la luz"


def test_translation_label_saying_translation_is_removed():
    assert TranslationLabelRemover("Spanish").remove_from("Translation: sea la luz") == "sea la luz"


def test_translation_label_saying_target_is_removed():
    assert TranslationLabelRemover("Spanish").remove_from("Target: sea la luz") == "sea la luz"


def test_translation_label_saying_output_is_removed():
    assert TranslationLabelRemover("Spanish").remove_from("Output: sea la luz") == "sea la luz"


def test_translation_label_saying_answer_is_removed():
    assert TranslationLabelRemover("Spanish").remove_from("Answer: sea la luz") == "sea la luz"


def test_translation_label_leaves_a_leading_dash_when_the_language_has_no_name():
    assert TranslationLabelRemover("").remove_from("- sea la luz") == "- sea la luz"


@pytest.fixture
def spanish() -> TargetLanguageProfile:
    return TargetLanguageProfile(["en el principio creó Dios los cielos y la tierra", "sea la luz"])


@pytest.fixture
def spanish_reader(spanish: TargetLanguageProfile) -> SingleReplyReader:
    return SingleReplyReader(TranslationLabelRemover("Spanish"), spanish)


def test_single_reply_reader_strips_a_leading_label(spanish_reader: SingleReplyReader):
    assert spanish_reader.read("Spanish: sea la luz") == "sea la luz"


def test_single_reply_reader_keeps_the_line_most_like_the_target_corpus(spanish_reader: SingleReplyReader):
    assert spanish_reader.read("Here is the translation of the verse:\n\nsea la luz") == "sea la luz"


def test_single_reply_reader_without_a_target_corpus_skips_a_leading_aside():
    reader = SingleReplyReader(TranslationLabelRemover("Spanish"), TargetLanguageProfile([]))
    assert reader.read("Here is the translation of the verse:\n\nsea la luz") == "sea la luz"


def test_single_reply_reader_strips_code_fences_and_backticks(spanish_reader: SingleReplyReader):
    assert spanish_reader.read("```text\nsea la luz\n```") == "sea la luz"
    assert spanish_reader.read("`sea la luz`") == "sea la luz"


def test_single_reply_reader_keeps_quotation_marks_and_apostrophes(spanish_reader: SingleReplyReader):
    # A verse can open and close with a quotation mark, and an apostrophe is a letter in some orthographies.
    reply = '"ŋa\'a sea la luz."'
    assert spanish_reader.read(reply) == reply


def test_single_reply_reader_of_an_empty_reply_is_empty(spanish_reader: SingleReplyReader):
    assert spanish_reader.read("  \n ") == ""


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
    # Fewer examples than the default, so that a test's small training corpus is not sent whole.
    config.setdefault("infer", {}).setdefault("prompt", {}).setdefault("num_examples", 10)
    return RemoteLLMConfig(exp_dir, config, Mock())


def test_config_defaults(tmp_path: Path):
    config = RemoteLLMConfig(
        tmp_path, {"model_type": "remote_llm", "model": "gpt-4o", "data": {"corpus_pairs": []}}, Mock()
    )
    assert config.get_prompt()["example_selection"]["method"] == "coverage"
    assert config.get_prompt()["num_examples"] == 100
    assert config.get_infer_batch_size() == 1
    assert config.infer["num_retries"] == 8
    assert config.infer["reuse_saved_replies"] is True
    assert config.infer["copy_exact_matches"] is True
    assert config.get_prompt()["back_translation"] is False
    # The hosted model tokenizes for itself, so preprocessing writes raw text.
    assert config.data["tokenize"] is False
    assert config.model_dir == tmp_path / "run"


def test_config_requires_a_model(tmp_path: Path):
    with pytest.raises(ValueError, match="LiteLLM format"):
        RemoteLLMConfig(tmp_path, {"model_type": "remote_llm", "model": "", "data": {"corpus_pairs": []}}, Mock())


def test_config_rejects_an_unknown_example_selection_method(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown example_selection.method"):
        make_config(tmp_path, infer={"prompt": {"example_selection": {"method": "embeddings"}}})


def test_config_rejects_a_batch_size_below_one(tmp_path: Path):
    with pytest.raises(ValueError, match="infer.get_infer_batch_size()"):
        make_config(tmp_path, infer={"infer_batch_size": 0})


def test_config_rejects_fewer_than_one_draft(tmp_path: Path):
    with pytest.raises(ValueError, match="infer.num_drafts"):
        make_config(tmp_path, infer={"num_drafts": 0})


def test_config_rejects_a_concurrency_below_one(tmp_path: Path):
    with pytest.raises(ValueError, match="infer.concurrency"):
        make_config(tmp_path, infer={"concurrency": 0})


def test_config_rejects_a_negative_number_of_examples(tmp_path: Path):
    with pytest.raises(ValueError, match="num_examples"):
        make_config(tmp_path, infer={"prompt": {"num_examples": -1}})


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
    assert "<source_to_translate>\nhello\n</source_to_translate>" in messages[1]["content"]
    assert "1. hello" not in messages[1]["content"]


def test_the_default_prompt_is_the_benchmarked_prompt_word_for_word(tmp_path: Path):
    # The default rests on a benchmark of exactly this prompt, so any change to it is a new, unmeasured prompt.
    examples = [
        Example("God so loved the world.", "Porque de tal manera amó Dios al mundo."),
        Example("Love one another & <all>.", "Amaos 'unos' a otros."),
    ]
    system, user = (m["content"] for m in make_config(tmp_path).build_messages(["God is love."], examples, EN, ES))

    assert system == (
        "Translate the final English Bible verse into Spanish, following the translation conventions demonstrated "
        "by this project's parallel examples.\nUse the examples as evidence for Spanish vocabulary, grammar, "
        "inflection, names, spelling and punctuation. Retain shared words and established borrowings when the "
        "examples support them; use natural target constructions instead of mechanically copying source wording."
        "\nPreserve the full meaning of the source: participants, actions, negation, relationships and emphasis. "
        "Adapt an example's wording to the current sentence; similar examples may describe different people or "
        "events. Nearby translations provide context, not additional content to translate.\nReturn only the "
        "Spanish translation of the final source verse on one line, without a label, verse number, explanation or "
        "alternative translations."
    )
    assert user == (
        "<translation_examples>\nEnglish: God so loved the world.\nSpanish: Porque de tal manera amó Dios al "
        "mundo.\n\nEnglish: Love one another &amp; &lt;all&gt;.\nSpanish: Amaos 'unos' a otros.\n"
        "</translation_examples>\n\nFollowing the project's examples above, translate only this English verse "
        "into Spanish.\n<source_to_translate>\nGod is love.\n</source_to_translate>\nSpanish:"
    )


def test_markup_in_the_source_is_escaped_but_apostrophes_are_kept(tmp_path: Path):
    user = make_config(tmp_path).build_messages(["Bread & <fish> for 'all'"], [], EN, ES)[1]["content"]
    assert "<source_to_translate>\nBread &amp; &lt;fish&gt; for 'all'\n</source_to_translate>" in user


def test_a_zero_shot_prompt_has_no_examples_block(tmp_path: Path):
    user = make_config(tmp_path, infer={"prompt": {"num_examples": 0}}).build_messages(["hello"], [], EN, ES)[1]

    assert user["content"] == (
        "Translate only this English verse into Spanish.\n<source_to_translate>\nhello\n</source_to_translate>\n"
        "Spanish:"
    )


def test_batch_prompt_numbers_the_segments(tmp_path: Path):
    config = make_config(tmp_path)
    user = config.build_messages(["one", "two", "three"], [], EN, ES)[1]["content"]

    assert "1. one\n2. two\n3. three" in user
    assert "exactly 3 lines" in user


def test_a_batch_prompt_follows_the_examples_it_is_given(tmp_path: Path):
    user = make_config(tmp_path).build_messages(["one", "two"], [Example("greeting", "saludo")], EN, ES)[1]

    assert "<translation_examples>\nEnglish: greeting\nSpanish: saludo\n</translation_examples>" in user["content"]
    assert "Following the project's examples above, translate each of these 2 consecutive" in user["content"]


def test_a_batch_system_message_asks_for_numbered_lines_rather_than_one(tmp_path: Path):
    # The single-segment system message asks for one line, which would contradict a batched request.
    system = make_config(tmp_path).build_messages(["one", "two"], [], EN, ES)[0]["content"]

    assert "one numbered line per source passage" in system
    assert "on one line" not in system


def test_a_custom_system_message_also_governs_batches(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"system_message": "Be terse."}})
    assert config.build_messages(["one", "two"], [], EN, ES)[0]["content"] == "Be terse."


def test_a_batch_system_message_applies_only_to_batches(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"batch_system_message": "Number them."}})

    assert config.build_messages(["one", "two"], [], EN, ES)[0]["content"] == "Number them."
    assert config.build_messages(["one"], [], EN, ES)[0]["content"].startswith("Translate the final English")


def test_a_hoisted_corpus_is_wrapped_as_translation_examples(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1000000}})
    write_training_corpus(tmp_path)
    list(model.translate(["anything"], "en", "es"))

    system, user = client.calls[0][0]["content"], client.calls[0][1]["content"]
    assert system.endswith(
        "\n\n<translation_examples>\nEnglish: in the beginning\nSpanish: en el principio\n\n"
        "English: let there be light\nSpanish: sea la luz\n</translation_examples>"
    )
    assert user == (
        "Translate only this English verse into Spanish.\n<source_to_translate>\nanything\n</source_to_translate>\n"
        "Spanish:"
    )


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


def test_a_back_translation_prompt_asks_for_the_source_wording_rendered_faithfully(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"back_translation": True}})
    system = config.build_messages(["hola"], [], ES, EN)[0]["content"]

    assert system.startswith("Translate the final Spanish Bible verse into English")
    assert system.endswith(
        "alternative translations.\nA back-translation shows what the Spanish text ACTUALLY says, so a reviewer "
        "who does not read Spanish can check it. Render the Spanish wording faithfully, including anything "
        "unexpected, unusual, added, missing or apparently mistaken — do NOT smooth it toward a familiar or "
        "expected wording, and do not supply content that is not in the Spanish."
    )


def test_a_back_translation_batch_prompt_also_asks_for_the_source_wording_rendered_faithfully(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"back_translation": True}})
    system = config.build_messages(["uno", "dos"], [], ES, EN)[0]["content"]

    assert "alternative translations.\nA back-translation shows what the Spanish text ACTUALLY says" in system


def test_back_translation_guidance_follows_a_custom_system_message(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"system_message": "Be terse.", "back_translation": True}})
    system = config.build_messages(["hola"], [], ES, EN)[0]["content"]

    assert system.startswith("Be terse.\nA back-translation shows what the Spanish text ACTUALLY says")


def test_back_translation_guidance_follows_a_custom_batch_system_message(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"batch_system_message": "Number them.", "back_translation": True}})
    system = config.build_messages(["uno", "dos"], [], ES, EN)[0]["content"]

    assert system.startswith("Number them.\nA back-translation shows what the Spanish text ACTUALLY says")


def test_back_translation_guidance_alone_replaces_an_empty_system_message(tmp_path: Path):
    config = make_config(tmp_path, infer={"prompt": {"system_message": "", "back_translation": True}})
    system = config.build_messages(["hola"], [], ES, EN)[0]

    assert system["role"] == "system"
    assert system["content"].startswith("A back-translation shows what the Spanish text ACTUALLY says")


# --- model ------------------------------------------------------------------------------

_NUMBERED_SOURCE = re.compile(r"^\d+\. ")


def priced(prompt_tokens: int, completion_tokens: int, cost: float) -> LLMUsageStatistics:
    return LLMUsageStatistics.of_priced_request(prompt_tokens, completion_tokens, cost)


def unpriced(prompt_tokens: int = 0, completion_tokens: int = 0) -> LLMUsageStatistics:
    return LLMUsageStatistics.of_unpriced_request(prompt_tokens, completion_tokens)


class ScriptedClient(CompletionClient):
    """A completion client that answers with a scripted function instead of a network call."""

    def __init__(
        self,
        responder: Callable[[List[Dict[str, str]]], str],
        logprobs_supported: bool = False,
        token_logprobs: Optional[List[TokenLogprob]] = None,
        tokens_per_word: Optional[int] = 1,
        usage: Optional[LLMUsageStatistics] = None,
    ) -> None:
        self._responder = responder
        self._logprobs_supported = logprobs_supported
        self._token_logprobs = token_logprobs
        self._tokens_per_word = tokens_per_word
        self._usage = usage if usage is not None else unpriced()
        self.calls: List[List[Dict[str, str]]] = []
        self.logprobs_requested: List[bool] = []

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        self.calls.append(messages)
        self.logprobs_requested.append(logprobs)
        text = self._responder(messages)
        if logprobs and self._token_logprobs is not None:
            return Completion(text, list(self._token_logprobs), self._usage)
        return Completion(text, usage=self._usage)

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


def test_a_reply_with_an_aside_still_writes_one_line_per_source(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\ntwo\n", encoding="utf-8")
    model, _ = make_model(tmp_path, lambda messages: "Sure! Here it is:\nsea la luz", infer={"concurrency": 1})

    model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"])

    assert (tmp_path / "out.txt").read_text(encoding="utf-8").splitlines() == ["sea la luz", "sea la luz"]


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
    assert info["retrieval_method"] == "coverage"
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

    list(model.translate(["let there be more light"], "en", "es"))

    user = client.calls[0][1]["content"]
    assert "let there be light" in user
    assert "sea la luz" in user


def test_the_index_is_rebuilt_when_the_checkpoint_is_gone(tmp_path: Path):
    # experiment.py deletes the run directory unless --save-checkpoints is passed, so the saved
    # index is only a cache; inference has to rebuild it from the training corpus.
    write_training_corpus(tmp_path)
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1}})

    list(model.translate(["let there be more light"], "en", "es"))

    assert "sea la luz" in client.calls[0][1]["content"]


def test_a_segment_the_corpus_already_translates_is_copied_without_a_request(tmp_path: Path, caplog):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"concurrency": 1})

    with caplog.at_level("INFO"):
        groups = list(model.translate(["let there be light", "anything"], "en", "es"))

    assert translations_of(groups) == ["sea la luz", "hola"]
    assert len(client.calls) == 1
    assert "translations copied from the training corpus: 1" in caplog.text


def test_a_batch_leaves_out_the_segments_the_corpus_already_translates(tmp_path: Path):
    model, client = make_model(tmp_path, echo_translations, infer={"infer_batch_size": 3, "concurrency": 1})

    groups = model.translate(["one", "let there be light", "two"], "en", "es")

    assert translations_of(groups) == ["translated 1", "sea la luz", "translated 2"]
    assert len(client.calls) == 1


def test_a_segment_the_corpus_translates_two_ways_is_copied_too(tmp_path: Path):
    (tmp_path / "train.src.txt").write_text("amen\namen\n", encoding="utf-8")
    (tmp_path / "train.trg.txt").write_text("amén\nasí sea\n", encoding="utf-8")
    model, client = make_model(tmp_path, lambda messages: "hola")

    assert translations_of(model.translate(["amen"], "en", "es"))[0] in {"amén", "así sea"}
    assert len(client.calls) == 0


def test_exact_matches_go_to_the_model_when_copying_is_turned_off(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"copy_exact_matches": False})
    assert translations_of(model.translate(["let there be light"], "en", "es")) == ["hola"]


def test_a_zero_shot_run_never_copies_from_the_corpus(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 0}})
    assert translations_of(model.translate(["let there be light"], "en", "es")) == ["hola"]


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


def test_token_logprobs_tolerate_a_provider_that_leaves_out_the_logprobs_field():
    assert _response_with({}).token_logprobs() == []


def test_token_logprobs_tolerate_a_provider_that_sends_null_logprobs():
    assert _response_with({"logprobs": None}).token_logprobs() == []


def test_token_logprobs_tolerate_a_provider_that_sends_null_logprob_content():
    assert _response_with({"logprobs": {"content": None}}).token_logprobs() == []


def test_token_logprobs_tolerate_a_provider_that_sends_empty_logprob_content():
    assert _response_with({"logprobs": {"content": []}}).token_logprobs() == []


def test_token_logprobs_skip_incomplete_entries():
    choice = {"logprobs": {"content": [{"token": "ho"}, {"token": "la", "logprob": -0.4}]}}
    assert _response_with(choice).token_logprobs() == [TokenLogprob("la", -0.4)]


class _Pricing:
    def __init__(self, cost: Optional[float]) -> None:
        self._cost = cost

    def completion_cost(self, completion_response) -> float:
        if self._cost is None:
            raise Exception("no pricing")
        return self._cost


def test_response_reads_the_text_and_usage_from_either_shape():
    raw = {"choices": [{"message": {"content": "hola"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    assert LiteLLMResponse(raw, _Pricing(0.01)).text() == "hola"
    assert LiteLLMResponse(raw, _Pricing(0.01)).usage() == priced(7, 3, 0.01)

    obj = _Obj(choices=[_Obj(message=_Obj(content="hola"))], usage=_Obj(prompt_tokens=7, completion_tokens=3))
    assert LiteLLMResponse(obj, _Pricing(0.01)).usage() == priced(7, 3, 0.01)


def test_response_tolerates_a_missing_usage_block():
    raw = {"choices": [{"message": {"content": "hola"}}]}
    assert LiteLLMResponse(raw, _Pricing(0.0)).usage() == priced(0, 0, 0.0)


def test_response_reports_an_unknown_cost_when_litellm_has_no_pricing():
    raw = {"choices": [{"message": {"content": "hola"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    assert LiteLLMResponse(raw, _Pricing(None)).usage() == unpriced(7, 3)


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
    assert "Translate the final {src_lang} Bible verse" in prompt["system_message"]
    assert "one numbered line per source passage" in prompt["batch_system_message"]
    assert "{source}" in prompt["instruction_template"]
    assert "{num_segments}" in prompt["batch_instruction_template"]


def test_the_effective_config_records_the_back_translation_guidance(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"back_translation": True}})
    path = tmp_path / "effective-config.yml"
    model.save_effective_config(path)

    prompt = yaml.safe_load(path.read_text(encoding="utf-8"))["infer"]["prompt"]
    assert "A back-translation shows what the {src_lang} text ACTUALLY says" in prompt["system_message"]
    assert "A back-translation shows what the {src_lang} text ACTUALLY says" in prompt["batch_system_message"]


def test_an_effective_config_run_again_does_not_repeat_the_back_translation_guidance(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"back_translation": True}})
    path = tmp_path / "effective-config.yml"
    model.save_effective_config(path)

    rerun = RemoteLLMConfig(tmp_path, yaml.safe_load(path.read_text(encoding="utf-8")), Mock())
    assert rerun.build_messages(["hola"], [], ES, EN)[0]["content"].count("A back-translation shows") == 1
    assert rerun.build_messages(["uno", "dos"], [], ES, EN)[0]["content"].count("A back-translation shows") == 1


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
    totals.add(Completion("a", usage=priced(100, 10, 0.01)))
    totals.add(Completion("b", usage=priced(200, 20, 0.02)))

    assert "2 requests, 300 prompt + 30 completion tokens, $0.0300" == totals.describe()


def test_usage_totals_report_an_unpriced_model_rather_than_calling_it_free():
    totals = UsageTotals()
    totals.add(Completion("a", usage=unpriced(100, 10)))
    assert "cost unavailable" in totals.describe()
    assert "$" not in totals.describe()


def test_usage_totals_flag_a_partial_cost():
    totals = UsageTotals()
    totals.add(Completion("a", usage=priced(100, 10, 0.01)))
    totals.add(Completion("b", usage=unpriced(100, 10)))

    assert "$0.0100 excluding 1 unpriced requests" in totals.describe()


def test_usage_totals_are_thread_safe():
    totals = UsageTotals()
    completion = Completion("a", usage=priced(1, 1, 0.001))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: totals.add(completion), range(2000)))

    assert totals.describe().startswith("2,000 requests, 2,000 prompt + 2,000 completion tokens")


def test_translating_logs_the_usage_and_cost(tmp_path: Path, caplog):
    client = ScriptedClient(lambda messages: "hola", usage=priced(500, 5, 0.002))
    write_training_corpus(tmp_path)
    config = make_config(tmp_path, infer={"concurrency": 1})
    model = RemoteLLMModel(config, ScriptedClientFactory(client))

    with caplog.at_level("INFO"):
        list(model.translate(["one", "two"], "en", "es"))

    assert "Translated 2 segments using 2 requests" in caplog.text
    assert "1,000 prompt + 10 completion tokens" in caplog.text
    assert "$0.0040" in caplog.text


def test_translating_warns_of_translations_left_blank(tmp_path: Path, caplog):
    # Only a label is left once the reply is cleaned.
    model, _ = make_model(tmp_path, lambda messages: "Spanish:", infer={"concurrency": 1})

    with caplog.at_level("WARNING"):
        list(model.translate(["one", "", "two"], "en", "es"))

    assert "2 translations were left blank" in caplog.text


def test_an_empty_batch_reply_recovered_by_splitting_is_not_reported_as_blank(tmp_path: Path, caplog):
    def empty_for_a_batch(messages: List[Dict[str, str]]) -> str:
        return "hola" if echo_translations(messages) == "translated" else ""

    model, _ = make_model(tmp_path, empty_for_a_batch, infer={"infer_batch_size": 2, "concurrency": 1})

    with caplog.at_level("INFO"):
        assert translations_of(model.translate(["one", "two"], "en", "es")) == ["hola", "hola"]

    assert "left blank" not in caplog.text


def test_a_hoisted_corpus_does_not_leave_a_dangling_examples_block(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1000000}})
    write_training_corpus(tmp_path)
    list(model.translate(["anything"], "en", "es"))

    system, user = client.calls[0][0]["content"], client.calls[0][1]["content"]
    assert "<translation_examples>" in system
    assert "<translation_examples>" not in user
    assert "Following the project's examples above" not in user


def test_a_hoisted_corpus_keeps_a_custom_instruction_template(tmp_path: Path):
    model, client = make_model(
        tmp_path,
        lambda messages: "hola",
        infer={"prompt": {"num_examples": 1000000, "instruction_template": "Mine: {source}"}},
    )
    write_training_corpus(tmp_path)
    list(model.translate(["anything"], "en", "es"))

    assert client.calls[0][1]["content"] == "Mine: anything"


def test_retrieved_examples_keep_their_block_when_the_corpus_is_not_hoisted(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"prompt": {"num_examples": 1}})
    write_training_corpus(tmp_path)
    list(model.translate(["let there be more light"], "en", "es"))

    assert "<translation_examples>" in client.calls[0][1]["content"]


def test_the_batch_prompt_drops_its_examples_block_when_the_corpus_is_hoisted(tmp_path: Path):
    model, client = make_model(
        tmp_path,
        lambda messages: "1. hola\n2. adios",
        infer={"prompt": {"num_examples": 1000000}, "infer_batch_size": 2},
    )
    write_training_corpus(tmp_path)
    list(model.translate(["one", "two"], "en", "es"))

    user = client.calls[0][1]["content"]
    assert "<translation_examples>" not in user
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

    list(model.translate(["let there be more light", "and so on"], "en", "es"))

    # A batched request draws on the same indexed corpus as a single-segment one.
    assert "sea la luz" in client.calls[0][1]["content"]


def test_the_batch_builder_numbers_the_segments_and_states_the_count(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "1. uno\n2. dos\n3. tres", infer={"infer_batch_size": 3})
    list(model.translate(["one", "two", "three"], "en", "es"))

    user = client.calls[0][1]["content"]
    assert "1. one\n2. two\n3. three" in user
    # The count is the contract BatchReplyReader checks the reply against.
    assert "exactly 3 lines" in user


def test_a_single_segment_request_does_not_use_the_batch_wording(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"infer_batch_size": 1})
    list(model.translate(["one"], "en", "es"))

    assert "consecutive" not in client.calls[0][1]["content"]


# --- retries ------------------------------------------------------------------------------


class HttpError(Exception):
    def __init__(self, status_code: int, message: str = "request failed") -> None:
        super().__init__(message)
        self.status_code = status_code


class SequenceClient(CompletionClient):
    """Replies to, or fails, each request in turn from a fixed list of outcomes."""

    def __init__(self, outcomes: List[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, Completion)
        return outcome


def retrying(outcomes: List[object], max_retries: int = 2) -> Tuple[RetryingCompletionClient, SequenceClient]:
    inner = SequenceClient(outcomes)
    return RetryingCompletionClient(inner, RetryPolicy(max_retries, delay_seconds=0)), inner


def test_an_empty_reply_is_retried():
    client, inner = retrying([Completion(""), Completion("hola")])

    assert client.complete([]).text == "hola"
    assert inner.calls == 2


def test_a_retried_reply_is_billed_for_the_discarded_attempt():
    client, _ = retrying([Completion("", usage=priced(100, 5, 0.01)), Completion("hola", usage=priced(100, 2, 0.02))])

    assert client.complete([]).usage == priced(200, 7, 0.03)


def test_an_unpriced_attempt_leaves_the_cost_unknown():
    client, _ = retrying([Completion("", usage=unpriced(100, 5)), Completion("hola", usage=priced(100, 2, 0.01))])
    assert client.complete([]).usage == unpriced(200, 7)


def test_an_empty_reply_is_returned_once_the_retries_run_out():
    client, inner = retrying([Completion("")] * 3, max_retries=2)

    assert client.complete([]).is_empty()
    assert inner.calls == 3


def test_a_malformed_response_is_retried():
    # LiteLLM raises a plain exception, with no HTTP status, for a reply that has no choices.
    client, _ = retrying([Exception("Invalid response object"), Completion("hola")])
    assert client.complete([]).text == "hola"


def test_a_request_timeout_is_retried():
    client, _ = retrying([HttpError(408), Completion("hola")])
    assert client.complete([]).text == "hola"


def test_a_rate_limit_is_retried():
    client, _ = retrying([HttpError(429), Completion("hola")])
    assert client.complete([]).text == "hola"


def test_an_internal_server_error_is_retried():
    client, _ = retrying([HttpError(500), Completion("hola")])
    assert client.complete([]).text == "hola"


def test_an_unavailable_service_is_retried():
    client, _ = retrying([HttpError(503), Completion("hola")])
    assert client.complete([]).text == "hola"


def test_a_bad_request_is_not_retried():
    client, inner = retrying([HttpError(400), Completion("hola")])

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 1


def test_an_unauthenticated_request_is_not_retried():
    client, inner = retrying([HttpError(401), Completion("hola")])

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 1


def test_a_forbidden_request_is_not_retried():
    client, inner = retrying([HttpError(403), Completion("hola")])

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 1


def test_a_request_for_something_not_found_is_not_retried():
    client, inner = retrying([HttpError(404), Completion("hola")])

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 1


def test_insufficient_credits_reported_as_a_rate_limit_are_not_retried():
    client, inner = retrying([HttpError(429, "Insufficient credits"), Completion("hola")])

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 1


def test_an_openai_insufficient_quota_error_is_not_retried():
    message = (
        "Error code: 429 - {'error': {'message': 'You exceeded your current quota.', 'code': 'insufficient_quota'}}"
    )
    client, inner = retrying([HttpError(429, message), Completion("hola")])

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 1


def test_an_api_key_spending_limit_reported_as_a_rate_limit_is_not_retried():
    client, inner = retrying([HttpError(429, "Key limit exceeded"), Completion("hola")])

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 1


def test_a_vertex_rate_limit_worded_as_an_exhausted_quota_is_retried():
    message = "VertexAIException - Resource has been exhausted (e.g. check quota)."
    client, _ = retrying([HttpError(429, message), Completion("hola")])
    assert client.complete([]).text == "hola"


def test_a_gemini_rate_limit_that_mentions_quota_and_billing_is_retried():
    message = "You exceeded your current quota, please check your plan and billing details. Please retry in 41.9s."
    client, _ = retrying([HttpError(429, message), Completion("hola")])
    assert client.complete([]).text == "hola"


def test_a_persistent_failure_is_raised_once_the_retries_run_out():
    client, inner = retrying([HttpError(503)] * 3, max_retries=2)

    with pytest.raises(HttpError):
        client.complete([])
    assert inner.calls == 3


def test_the_retrying_client_reports_what_the_wrapped_client_supports():
    inner = ScriptedClient(lambda messages: "hola", logprobs_supported=True, tokens_per_word=2)
    client = RetryingCompletionClient(inner, RetryPolicy(0))

    assert client.supports_logprobs()
    assert client.count_tokens("two words") == 4


def test_the_retrying_client_estimates_with_the_wrapped_client():
    inner = PricedClient(reasoning=ReasoningBudget("a-model", 4096, 1024))
    client = RetryingCompletionClient(inner, RetryPolicy(0))
    messages = [{"role": "user", "content": "two words"}]

    assert client.estimate_usage(messages, 100, 1).describe() == inner.estimate_usage(messages, 100, 1).describe()


class FakeLiteLLM:
    """LiteLLM's module interface, narrowed to what LiteLLMCompletionClient calls."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome
        self.calls: List[dict] = []

    def completion(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        assert isinstance(self._outcome, dict)
        return self._outcome

    def completion_cost(self, completion_response) -> float:
        return 0.0


def test_litellm_is_asked_not_to_retry_on_its_own():
    # LiteLLM's retries need tenacity, and when it is missing they replace the provider's error with an import error.
    fake = FakeLiteLLM({"choices": [{"message": {"content": "hola"}}], "usage": {"prompt_tokens": 3}})
    LiteLLMCompletionClient("gpt-4o", CompletionSettings(0.0, 16, 5, 10), litellm=fake).complete([])
    assert fake.calls[0]["num_retries"] == 0


def test_a_provider_error_reaches_the_retry_policy_unchanged():
    fake = FakeLiteLLM(HttpError(429, "rate-limited upstream"))
    client = RetryingCompletionClient(
        LiteLLMCompletionClient("gpt-4o", CompletionSettings(0.0, 16, 1, 10), litellm=fake),
        RetryPolicy(1, delay_seconds=0),
    )

    with pytest.raises(HttpError, match="rate-limited upstream"):
        client.complete([])
    assert len(fake.calls) == 2


def test_importing_litellm_turns_off_its_feedback_banner():
    litellm = pytest.importorskip("litellm")
    LiteLLMCompletionClient("gpt-4o", CompletionSettings(0.0, 16, 0, 10))
    assert litellm.suppress_debug_info is True


# --- saved replies -------------------------------------------------------------------------


def reply(text: str) -> Completion:
    return Completion(text, usage=unpriced())


def save_reply(saved: SavedReplies, key: str, completion: Completion) -> None:
    occurrence, _ = saved.claim(key)
    saved.save(key, occurrence, completion)


def test_a_saved_reply_comes_back_whole_in_a_later_run(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    original = Completion("hola", [TokenLogprob("hola", -0.5)], priced(10, 2, 0.003))
    with saved.session():
        save_reply(saved, "request", original)

    resumed = SavedReplies(tmp_path / "replies.jsonl")
    with resumed.session():
        assert resumed.claim("request")[1] == original


def test_a_saved_reply_comes_back_in_a_later_session_of_the_same_run(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    with saved.session():
        save_reply(saved, "request", Completion("hola"))

    with saved.session():
        assert saved.claim("request")[1] == Completion("hola")


def test_a_reply_with_a_unicode_line_separator_comes_back_whole(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    with saved.session():
        save_reply(saved, "request", reply("one\u2028two\u0085three"))

    resumed = SavedReplies(tmp_path / "replies.jsonl")
    with resumed.session():
        assert resumed.claim("request")[1] == reply("one\u2028two\u0085three")


def test_identical_requests_in_one_session_are_told_apart(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    with saved.session():
        save_reply(saved, "request", reply("first"))
        save_reply(saved, "request", reply("second"))

    resumed = SavedReplies(tmp_path / "replies.jsonl")
    with resumed.session():
        assert [resumed.claim("request")[1] for _ in range(3)] == [reply("first"), reply("second"), None]


def test_nothing_is_saved_outside_a_session(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    save_reply(saved, "request", Completion("hola"))

    assert not (tmp_path / "replies.jsonl").exists()
    with saved.session():
        assert saved.claim("request")[1] is None


def test_a_partly_written_last_line_does_not_spoil_the_next_reply(tmp_path: Path):
    path = tmp_path / "replies.jsonl"
    saved = SavedReplies(path)
    with saved.session():
        save_reply(saved, "kept", reply("hola"))
    with path.open("a", encoding="utf-8") as file:
        file.write('{"key": "interrupted", "occ')

    with saved.session():
        save_reply(saved, "after", reply("adios"))

    resumed = SavedReplies(path)
    with resumed.session():
        assert resumed.claim("kept")[1] == reply("hola")
        assert resumed.claim("interrupted")[1] is None
        assert resumed.claim("after")[1] == reply("adios")


def test_a_last_line_cut_inside_a_character_does_not_spoil_the_earlier_replies(tmp_path: Path):
    path = tmp_path / "replies.jsonl"
    saved = SavedReplies(path)
    with saved.session():
        save_reply(saved, "kept", reply("ŋa'a"))
    with path.open("ab") as file:
        file.write('{"key": "interrupted", "text": "ŋ'.encode("utf-8")[:-1])

    resumed = SavedReplies(path)
    with resumed.session():
        assert resumed.claim("kept")[1] == reply("ŋa'a")


def test_a_failed_run_resumes_without_repeating_the_finished_requests(tmp_path: Path):
    calls: List[List[Dict[str, str]]] = []

    def fail_on_the_third(messages: List[Dict[str, str]]) -> str:
        calls.append(messages)
        if len(calls) == 3:
            raise RuntimeError("provider unavailable")
        return f"reply {len(calls)}"

    model, _ = make_model(tmp_path, fail_on_the_third, infer={"concurrency": 1})
    with pytest.raises(RuntimeError):
        list(model.translate(["one", "two", "three", "four"], "en", "es"))

    resumed, client = make_model(tmp_path, lambda messages: "fresh", infer={"concurrency": 1})
    groups = list(resumed.translate(["one", "two", "three", "four"], "en", "es"))

    assert translations_of(groups) == ["reply 1", "reply 2", "fresh", "fresh"]
    assert len(client.calls) == 2


def test_a_failed_run_says_that_rerunning_resumes(tmp_path: Path, caplog):
    def unavailable(messages: List[Dict[str, str]]) -> str:
        raise RuntimeError("provider unavailable")

    model, _ = make_model(tmp_path, unavailable, infer={"concurrency": 1})
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        list(model.translate(["one"], "en", "es"))

    assert "rerunning resumes" in caplog.text


def test_each_draft_of_a_segment_resumes_with_its_own_reply(tmp_path: Path):
    replies = iter(["draft a", "draft b", "draft c"])
    infer = {"num_drafts": 3, "temperature": 0.8, "concurrency": 1}
    model, _ = make_model(tmp_path, lambda messages: next(replies), infer=dict(infer))
    list(model.translate(["one"], "en", "es", produce_multiple_translations=True))

    resumed, client = make_model(tmp_path, lambda messages: "fresh", infer=dict(infer))
    group = list(resumed.translate(["one"], "en", "es", produce_multiple_translations=True))[0]

    assert sorted(translation.get_translation() for translation in group) == ["draft a", "draft b", "draft c"]
    assert client.calls == []


def test_an_empty_reply_is_asked_for_again_when_resuming(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "", infer={"concurrency": 1})
    list(model.translate(["one"], "en", "es"))

    resumed, client = make_model(tmp_path, lambda messages: "hola", infer={"concurrency": 1})
    assert translations_of(list(resumed.translate(["one"], "en", "es"))) == ["hola"]
    assert len(client.calls) == 1


def test_a_reply_that_cleans_to_nothing_is_asked_for_again_when_resuming(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "Spanish:", infer={"concurrency": 1})
    assert translations_of(list(model.translate(["one"], "en", "es"))) == [""]

    resumed, client = make_model(tmp_path, lambda messages: "hola", infer={"concurrency": 1})
    assert translations_of(list(resumed.translate(["one"], "en", "es"))) == ["hola"]
    assert len(client.calls) == 1


def test_a_batch_reply_with_a_blank_entry_is_recovered_by_splitting(tmp_path: Path):
    def blank_second(messages: List[Dict[str, str]]) -> str:
        return "hola" if echo_translations(messages) == "translated" else "1. uno\n2."

    model, _ = make_model(tmp_path, blank_second, infer={"infer_batch_size": 2, "concurrency": 1})
    assert translations_of(model.translate(["one", "two"], "en", "es")) == ["hola", "hola"]


def test_a_saved_reply_the_caller_cannot_use_is_asked_for_again(tmp_path: Path):
    path = tmp_path / "replies.jsonl"
    first = ReplyReusingClient(SavedReplies(path), "settings")
    with first.session():
        first.complete(ScriptedClient(lambda messages: "Spanish:"), [], False, lambda reply: True)

    resumed = ReplyReusingClient(SavedReplies(path), "settings")
    client = ScriptedClient(lambda messages: "hola")
    with resumed.session():
        assert resumed.complete(client, [], False, lambda reply: reply.text != "Spanish:").text == "hola"
    assert len(client.calls) == 1


def test_a_saved_reply_is_not_reused_under_different_settings(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "cool", infer={"concurrency": 1, "temperature": 0.2})
    list(model.translate(["one"], "en", "es"))

    warmer, _ = make_model(tmp_path, lambda messages: "warm", infer={"concurrency": 1, "temperature": 0.9})
    assert translations_of(list(warmer.translate(["one"], "en", "es"))) == ["warm"]


def test_saved_replies_are_neither_kept_nor_reused_when_reuse_is_off(tmp_path: Path):
    infer = {"concurrency": 1, "reuse_saved_replies": False}
    model, _ = make_model(tmp_path, lambda messages: "first", infer=dict(infer))
    list(model.translate(["one"], "en", "es"))

    again, _ = make_model(tmp_path, lambda messages: "second", infer=dict(infer))
    assert translations_of(list(again.translate(["one"], "en", "es"))) == ["second"]
    assert list(tmp_path.glob("*.jsonl")) == []


def test_forcing_inference_sends_every_request_again(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "first", infer={"concurrency": 1})
    list(model.translate(["one"], "en", "es"))

    forced, client = make_model(tmp_path, lambda messages: "second", infer={"concurrency": 1})
    forced.discard_saved_inference()
    assert translations_of(list(forced.translate(["one"], "en", "es"))) == ["second"]
    assert len(client.calls) == 1


def test_forcing_inference_again_in_the_same_run_keeps_the_replies_it_has_made(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "fresh", infer={"concurrency": 1})
    model.discard_saved_inference()
    list(model.translate(["one"], "en", "es"))

    model.discard_saved_inference()
    list(model.translate(["one"], "en", "es"))
    assert len(client.calls) == 1


def test_a_forced_run_that_failed_resumes_from_its_own_replies(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "old", infer={"concurrency": 1})
    list(model.translate(["one", "two"], "en", "es"))

    def fail_on_two(messages: List[Dict[str, str]]) -> str:
        if "two" in messages[-1]["content"]:
            raise RuntimeError("provider unavailable")
        return "new"

    forced, _ = make_model(tmp_path, fail_on_two, infer={"concurrency": 1})
    forced.discard_saved_inference()
    with pytest.raises(RuntimeError):
        list(forced.translate(["one", "two"], "en", "es"))

    resumed, _ = make_model(tmp_path, lambda messages: "newer", infer={"concurrency": 1})
    assert translations_of(list(resumed.translate(["one", "two"], "en", "es"))) == ["new", "newer"]


def test_forcing_inference_still_copies_what_the_corpus_translates(tmp_path: Path):
    model, client = make_model(tmp_path, lambda messages: "hola", infer={"concurrency": 1})
    model.discard_saved_inference()
    assert translations_of(model.translate(["let there be light"], "en", "es")) == ["sea la luz"]
    assert client.calls == []


def test_forcing_inference_without_saved_replies_does_nothing(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola", infer={"reuse_saved_replies": False})
    model.discard_saved_inference()
    assert translations_of(model.translate(["one"], "en", "es")) == ["hola"]


def test_test_predictions_are_complete_when_only_blank_sources_have_blank_translations(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\n\nthree\n", encoding="utf-8")
    (tmp_path / "predictions.txt").write_text("uno\n\ntres\n", encoding="utf-8")
    model, _ = make_model(tmp_path, lambda messages: "hola")

    assert model.has_completed_translation(tmp_path / "test.src.txt", tmp_path / "predictions.txt")


def test_test_predictions_are_incomplete_while_a_source_has_a_blank_translation(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\n\nthree\n", encoding="utf-8")
    (tmp_path / "predictions.txt").write_text("uno\ndos\n\n", encoding="utf-8")
    model, _ = make_model(tmp_path, lambda messages: "hola")

    assert not model.has_completed_translation(tmp_path / "test.src.txt", tmp_path / "predictions.txt")


def test_test_predictions_are_incomplete_when_they_have_fewer_lines_than_the_source(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\n\nthree\n", encoding="utf-8")
    (tmp_path / "predictions.txt").write_text("uno\n", encoding="utf-8")
    model, _ = make_model(tmp_path, lambda messages: "hola")

    assert not model.has_completed_translation(tmp_path / "test.src.txt", tmp_path / "predictions.txt")


def test_missing_test_predictions_are_unfinished(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\n", encoding="utf-8")
    model, _ = make_model(tmp_path, lambda messages: "hola")
    assert not model.has_completed_translation(tmp_path / "test.src.txt", tmp_path / "predictions.txt")


def test_usage_totals_count_reused_replies_without_billing_them_again():
    totals = UsageTotals()
    totals.add(Completion("hola", usage=priced(100, 0, 0.01), reused=True))
    totals.add(Completion("adios", usage=priced(100, 0, 0.01)))

    described = totals.describe()
    assert "1 requests, 100 prompt" in described
    assert "$0.0100" in described
    assert "replies reused from an earlier run: 1" in described


def test_usage_totals_include_the_totals_of_another_translation():
    totals = UsageTotals()
    totals.add(Completion("a", usage=priced(100, 10, 0.01)))
    other = UsageTotals()
    other.add(Completion("b", usage=unpriced(200, 20)))
    other.add(Completion("c", usage=priced(300, 0, 0.03), reused=True))
    other.add_copied_translation()

    totals.include(other)

    assert "2 requests, 300 prompt + 30 completion tokens, $0.0100 excluding 1 unpriced requests" in totals.describe()
    assert "replies reused from an earlier run: 1" in totals.describe()
    assert "translations copied from the training corpus: 1" in totals.describe()


# --- read-only saved replies ---------------------------------------------------------------


def test_read_only_saved_replies_find_a_reply_an_earlier_run_saved(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    with saved.session():
        save_reply(saved, "request", reply("hola"))

    read_only = ReadOnlySavedReplies(tmp_path / "replies.jsonl")
    with read_only.session():
        assert read_only.claim("request")[1] == reply("hola")


def test_read_only_saved_replies_save_nothing(tmp_path: Path):
    read_only = ReadOnlySavedReplies(tmp_path / "replies.jsonl")
    with read_only.session():
        save_reply(read_only, "request", Completion("hola"))
        assert read_only.claim("request")[1] is None

    assert not (tmp_path / "replies.jsonl").exists()


def test_read_only_saved_replies_leave_a_partly_written_last_line_alone(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    with saved.session():
        save_reply(saved, "request", Completion("hola"))
    with (tmp_path / "replies.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"key": "cut sh')
    before = (tmp_path / "replies.jsonl").read_bytes()

    read_only = ReadOnlySavedReplies(tmp_path / "replies.jsonl")
    with read_only.session():
        read_only.claim("request")

    assert (tmp_path / "replies.jsonl").read_bytes() == before


def test_read_only_saved_replies_forget_but_keep_the_file_when_discarded(tmp_path: Path):
    saved = SavedReplies(tmp_path / "replies.jsonl")
    with saved.session():
        save_reply(saved, "request", Completion("hola"))

    read_only = ReadOnlySavedReplies(tmp_path / "replies.jsonl")
    read_only.discard()
    with read_only.session():
        assert read_only.claim("request")[1] is None
    assert (tmp_path / "replies.jsonl").is_file()


# --- estimating with LiteLLM ------------------------------------------------------------------


class FakeLiteLLMPrices:
    """LiteLLM's token counting and price list, for a model it prices per token, or for one it does not know."""

    def __init__(self, model_info: Optional[dict] = None, supports_reasoning: bool = False) -> None:
        self._model_info = model_info
        self._supports_reasoning = supports_reasoning
        self.counted: List[List[Dict[str, str]]] = []

    def token_counter(self, model: str, messages: List[Dict[str, str]]) -> int:
        self.counted.append(messages)
        return 1000

    def get_model_info(self, model: str) -> dict:
        if self._model_info is None:
            raise Exception("This model isn't mapped yet.")
        return self._model_info

    def cost_per_token(
        self, model: str, prompt_tokens: int, completion_tokens: int, cache_read_input_tokens: int = 0
    ) -> Tuple[float, float]:
        info = self.get_model_info(model)
        # Like LiteLLM, a model with no cache price has its cached tokens billed at nothing.
        cache_price = info.get("cache_read_input_token_cost") or 0.0
        uncached_tokens = prompt_tokens - cache_read_input_tokens
        prompt_cost = uncached_tokens * info["input_cost_per_token"] + cache_read_input_tokens * cache_price
        return prompt_cost, completion_tokens * info["output_cost_per_token"]

    def supports_reasoning(self, model: str) -> bool:
        return self._supports_reasoning


def _priced_litellm(cache_price: Optional[float] = 0.0001, supports_reasoning: bool = False) -> FakeLiteLLMPrices:
    info = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
    if cache_price is not None:
        info["cache_read_input_token_cost"] = cache_price
    return FakeLiteLLMPrices(info, supports_reasoning)


def _litellm_client(fake: FakeLiteLLMPrices, extra_kwargs: Optional[dict] = None) -> LiteLLMCompletionClient:
    return LiteLLMCompletionClient("gpt-4o", CompletionSettings(0.0, 16, 0, 10), extra_kwargs, litellm=fake)


def _prompt() -> List[Dict[str, str]]:
    return [{"role": "system", "content": "translate"}, {"role": "user", "content": "hello"}]


def test_estimated_usage_adds_the_prompt_and_completion_prices():
    usage = _litellm_client(_priced_litellm()).estimate_usage(_prompt(), 100)
    assert usage.describe() == "1 requests, 1,000 prompt + 100 completion tokens, $1.2000"


def test_estimated_usage_bills_cached_prompt_tokens_at_the_cache_price():
    usage = _litellm_client(_priced_litellm()).estimate_usage(_prompt(), 100, 800)
    assert usage.describe().endswith("; $0.4800 if the provider caches the system message the requests share")


def test_estimated_usage_says_nothing_of_caching_for_a_model_without_a_cache_price():
    usage = _litellm_client(_priced_litellm(cache_price=None)).estimate_usage(_prompt(), 100, 800)
    assert usage.describe() == "1 requests, 1,000 prompt + 100 completion tokens, $1.2000"


def test_estimated_usage_is_unpriced_for_a_model_litellm_has_no_price_for():
    usage = _litellm_client(FakeLiteLLMPrices()).estimate_usage(_prompt(), 100)
    assert usage.describe().endswith("; cost unavailable (no pricing for this model)")


def test_estimated_usage_counts_the_tokens_of_the_whole_chat():
    fake = _priced_litellm()

    _litellm_client(fake).estimate_usage(_prompt(), 100)

    assert fake.counted == [_prompt()]


def _warns_that_reasoning_is_left_out(client: LiteLLMCompletionClient, caplog) -> bool:
    with caplog.at_level("WARNING"):
        client.estimate_usage(_prompt(), 100)
    return "leaves out" in caplog.text


def test_reasoning_is_left_out_with_a_warning_for_a_model_that_litellm_says_supports_it(caplog):
    assert _warns_that_reasoning_is_left_out(_litellm_client(_priced_litellm(supports_reasoning=True)), caplog)


def test_reasoning_is_left_out_with_a_warning_for_a_model_asked_to_reason_by_its_litellm_params(caplog):
    client = _litellm_client(_priced_litellm(), {"reasoning": {"effort": "minimal"}})
    assert _warns_that_reasoning_is_left_out(client, caplog)


def test_there_is_no_reasoning_to_leave_out_for_a_model_that_neither_supports_nor_is_asked_to_reason(caplog):
    assert not _warns_that_reasoning_is_left_out(_litellm_client(_priced_litellm()), caplog)


class UncountingLiteLLM:
    """LiteLLM failing to count a model's tokens."""

    def token_counter(self, model: str, messages: List[Dict[str, str]]) -> int:
        raise ValueError("no tokenizer")


def test_estimated_usage_fails_for_a_model_litellm_cannot_count():
    client = LiteLLMCompletionClient("made-up", CompletionSettings(0.0, 16, 0, 10), litellm=UncountingLiteLLM())

    with pytest.raises(RuntimeError, match="could not be counted"):
        client.estimate_usage(_prompt(), 100)


class UnrecognizingLiteLLM(FakeLiteLLMPrices):
    """LiteLLM failing to look up a model it does not recognize, though it can still count the model's tokens."""

    def supports_reasoning(self, model: str) -> bool:
        raise ValueError("unrecognized model")


def test_a_model_litellm_cannot_look_up_has_no_reasoning_to_leave_out(caplog):
    client = LiteLLMCompletionClient("made-up", CompletionSettings(0.0, 16, 0, 10), litellm=UnrecognizingLiteLLM())
    assert not _warns_that_reasoning_is_left_out(client, caplog)


class FakeLiteLLMReasoning(FakeLiteLLMPrices):
    """LiteLLM translating reasoning settings into the parameters it would send, or failing to."""

    def __init__(self, optional_params: Optional[dict] = None, supports_reasoning: bool = False) -> None:
        super().__init__(supports_reasoning=supports_reasoning)
        self._optional_params = optional_params
        self.translated: List[dict] = []

    def get_llm_provider(self, model: str) -> Tuple[str, str, None, None]:
        provider, _, name = model.partition("/")
        return name, provider, None, None

    def get_optional_params(self, **settings) -> dict:
        self.translated.append(settings)
        if self._optional_params is None:
            raise ValueError("unsupported parameters")
        return self._optional_params


def _reasoning_client(model: str, litellm_options: dict, fake: FakeLiteLLMReasoning) -> LiteLLMCompletionClient:
    return LiteLLMCompletionClient(model, CompletionSettings(0.0, 4096, 0, 10), litellm_options, litellm=fake)


def _reasoning_tokens(model: str, litellm_options: dict, fake: FakeLiteLLMReasoning, reply_tokens: int = 100) -> int:
    usage = _reasoning_client(model, litellm_options, fake).estimate_usage(_prompt(), reply_tokens)
    return usage.to_record()["completion_tokens"] - reply_tokens


def test_a_thinking_budget_litellm_would_send_is_estimated_in_full():
    fake = FakeLiteLLMReasoning({"thinking": {"type": "enabled", "budget_tokens": 1024}})

    assert _reasoning_tokens("anthropic/claude-sonnet-4-5", {"reasoning_effort": "low"}, fake) == 1024


def test_a_gemini_thinking_budget_litellm_would_send_is_estimated_in_full():
    fake = FakeLiteLLMReasoning({"thinkingConfig": {"thinkingBudget": 2048, "includeThoughts": True}})

    assert _reasoning_tokens("gemini/gemini-2.5-pro", {"reasoning_effort": "medium"}, fake) == 2048


def test_litellm_is_asked_about_only_the_reasoning_settings_and_max_new_tokens():
    fake = FakeLiteLLMReasoning({"thinking": {"type": "enabled", "budget_tokens": 3000}})
    options = {"thinking": {"type": "enabled", "budget_tokens": 3000}, "api_base": "https://example.org"}

    _reasoning_tokens("anthropic/claude-sonnet-4-5", options, fake)

    assert fake.translated == [
        {
            "model": "claude-sonnet-4-5",
            "custom_llm_provider": "anthropic",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 3000},
        }
    ]


def test_a_budget_the_model_decides_for_itself_is_left_out_with_a_warning(caplog):
    fake = FakeLiteLLMReasoning({"thinkingConfig": {"thinkingBudget": -1}}, supports_reasoning=True)

    with caplog.at_level("WARNING"):
        tokens = _reasoning_tokens("gemini/gemini-2.5-pro", {"reasoning_effort": "high"}, fake)

    assert tokens == 0
    assert "leaves out" in caplog.text


def test_adaptive_thinking_without_a_budget_is_left_out_with_a_warning(caplog):
    fake = FakeLiteLLMReasoning({"thinking": {"type": "adaptive"}}, supports_reasoning=True)

    with caplog.at_level("WARNING"):
        tokens = _reasoning_tokens("anthropic/claude-opus-4-6", {"reasoning_effort": "high"}, fake)

    assert tokens == 0
    assert "leaves out" in caplog.text


def test_gpt_6_without_a_litellm_budget_reasons_at_its_measured_rate():
    fake = FakeLiteLLMReasoning({"reasoning_effort": "high"}, supports_reasoning=True)

    assert _reasoning_tokens("openrouter/openai/gpt-6.1-sol", {"reasoning_effort": "high"}, fake) == 570


def test_gpt_6_whose_reasoning_settings_litellm_rejects_reasons_at_its_measured_rate():
    fake = FakeLiteLLMReasoning()

    assert _reasoning_tokens("openrouter/openai/gpt-6.1-sol", {"reasoning_effort": "high"}, fake) == 570


def test_gpt_6_reasons_at_its_measured_rate_within_max_new_tokens():
    tokens = _reasoning_tokens("openai/gpt-6.1-sol", {"reasoning_effort": "xhigh"}, FakeLiteLLMReasoning(), 600)

    assert tokens == 4096 - 600


def test_litellm_is_not_asked_about_a_model_without_reasoning_settings():
    fake = FakeLiteLLMReasoning({"thinkingConfig": {"thinkingBudget": 2048}}, supports_reasoning=True)

    tokens = _reasoning_tokens("gemini/gemini-2.5-pro", {}, fake)

    assert fake.translated == []
    assert tokens == 0


def test_a_model_that_cannot_reason_has_no_reasoning_to_estimate(caplog):
    with caplog.at_level("INFO"):
        tokens = _reasoning_tokens("gpt-4o", {}, FakeLiteLLMReasoning())

    assert tokens == 0
    assert caplog.text == ""


def test_a_client_works_out_its_reasoning_estimate_once_for_all_its_requests(caplog):
    fake = FakeLiteLLMReasoning({"thinking": {"type": "enabled", "budget_tokens": 1024}})
    client = _reasoning_client("anthropic/claude-sonnet-4-5", {"reasoning_effort": "low"}, fake)

    with caplog.at_level("INFO"):
        client.estimate_usage(_prompt(), 100)
        client.estimate_usage(_prompt(), 100)

    assert len(fake.translated) == 1
    assert caplog.text.count("thinking budget") == 1


@pytest.mark.slow
def test_estimated_usage_prices_a_known_model_with_litellm_and_bills_cached_tokens_for_less():
    pytest.importorskip("litellm")
    client = LiteLLMCompletionClient("gpt-4o", CompletionSettings(0.0, 16, 0, 10))

    described = client.estimate_usage([{"role": "user", "content": "word " * 10000}], 1000, 8000).describe()

    assert "cost unavailable" not in described
    assert "if the provider caches" in described


@pytest.mark.slow
def test_estimated_usage_is_unpriced_for_a_model_missing_from_litellms_price_list():
    pytest.importorskip("litellm")
    client = LiteLLMCompletionClient("openrouter/made-up/nonexistent", CompletionSettings(0.0, 16, 0, 10))

    assert client.estimate_usage(_prompt(), 1000).to_record()["cost"] is None


@pytest.mark.slow
def test_litellm_gives_claude_a_thinking_budget_for_a_reasoning_effort():
    pytest.importorskip("litellm")
    settings = CompletionSettings(0.0, 8192, 0, 10)
    client = LiteLLMCompletionClient("anthropic/claude-sonnet-4-5", settings, {"reasoning_effort": "low"})

    assert client.estimate_usage(_prompt(), 100).to_record()["completion_tokens"] == 100 + 1024


@pytest.mark.slow
def test_litellm_gives_gemini_a_thinking_budget_for_a_reasoning_effort():
    pytest.importorskip("litellm")
    settings = CompletionSettings(0.0, 8192, 0, 10)
    client = LiteLLMCompletionClient("gemini/gemini-2.5-pro", settings, {"reasoning_effort": "high"})

    assert client.estimate_usage(_prompt(), 100).to_record()["completion_tokens"] == 100 + 4096


# --- dry runs -------------------------------------------------------------------------------


class PricedClient(ScriptedClient):
    """A scripted client with flat per-token prices, which bills cached prompt tokens at a tenth of the price."""

    def __init__(self, prices_cached_prompts: bool = True, reasoning: Optional[ReasoningEstimate] = None) -> None:
        super().__init__(lambda messages: "hola")
        self._prices_cached_prompts = prices_cached_prompts
        self._reasoning = reasoning if reasoning is not None else NoReasoning()

    def _price(self, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0) -> LLMUsageStatistics:
        if cached_prompt_tokens > 0 and not self._prices_cached_prompts:
            return unpriced(prompt_tokens, completion_tokens)
        uncached_tokens = prompt_tokens - cached_prompt_tokens
        cost = uncached_tokens * 0.001 + cached_prompt_tokens * 0.0001 + completion_tokens * 0.002
        return priced(prompt_tokens, completion_tokens, cost)

    def _reasoning_estimate(self) -> ReasoningEstimate:
        return self._reasoning


def make_dry_run_model(
    tmp_path: Path, client: Optional[ScriptedClient] = None, with_corpus: bool = True, **overrides
) -> Tuple[DryRunRemoteLLMModel, ScriptedClient]:
    if with_corpus and not (tmp_path / "train.src.txt").is_file():
        write_training_corpus(tmp_path)
    if client is None:
        client = ScriptedClient(lambda messages: "hola")
    return DryRunRemoteLLMModel(make_config(tmp_path, **overrides), ScriptedClientFactory(client)), client


def write_large_training_corpus(exp_dir: Path) -> None:
    """Enough examples that, sent whole, they make a system message long enough for a provider to cache."""
    (exp_dir / "train.src.txt").write_text("".join(f"verse {i} of the source\n" for i in range(300)), encoding="utf-8")
    (exp_dir / "train.trg.txt").write_text("".join(f"versículo {i} del texto\n" for i in range(300)), encoding="utf-8")


def words_in_the_first_request(exp_dir: Path, sources: List[str], **overrides) -> Tuple[int, int]:
    """The words a real run sends in its first request, and in that request's system message."""
    exp_dir.mkdir(exist_ok=True)
    model, client = make_model(exp_dir, lambda messages: "hola", **overrides)
    list(model.translate(sources, "en", "es"))
    messages = client.calls[0]
    system_message_words = len(messages[0]["content"].split()) if messages[0]["role"] == "system" else 0
    return sum(len(message["content"].split()) for message in messages), system_message_words


def dollars_in(text: str) -> List[float]:
    return [float(amount) for amount in re.findall(r"\$(\d+\.\d{4})", text)]


def test_a_dry_run_sends_no_request(tmp_path: Path):
    model, client = make_dry_run_model(tmp_path)

    list(model.translate(["one", "two"], "en", "es"))

    assert client.calls == []


def test_a_dry_run_yields_a_blank_translation_for_each_segment(tmp_path: Path):
    model, _ = make_dry_run_model(tmp_path)

    assert translations_of(list(model.translate(["one", "two"], "en", "es"))) == ["", ""]


def test_a_dry_run_model_produces_no_translations(tmp_path: Path):
    model, _ = make_dry_run_model(tmp_path)
    assert not model.produces_translations()


def test_a_remote_model_produces_translations(tmp_path: Path):
    model, _ = make_model(tmp_path, lambda messages: "hola")
    assert model.produces_translations()


def test_a_remote_llm_config_creates_a_dry_run_model(tmp_path: Path):
    client = ScriptedClient(lambda messages: "hola")
    model = make_config(tmp_path).create_dry_run_model(ScriptedClientFactory(client))
    assert not model.produces_translations()


def test_a_dry_run_counts_the_tokens_of_the_prompt_a_real_run_would_send(tmp_path: Path, caplog):
    prompt_words, _ = words_in_the_first_request(tmp_path / "real", ["one"])
    model, _ = make_dry_run_model(tmp_path)

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    assert f"1 requests, {prompt_words:,} prompt" in caplog.text


def test_a_dry_run_estimates_a_reply_from_the_corpus_length_ratio(tmp_path: Path, caplog):
    # The corpus has 6 target words for its 7 source words.
    model, _ = make_dry_run_model(tmp_path)

    with caplog.at_level("INFO"):
        list(model.translate(["one two three four five six seven"], "en", "es"))

    assert "+ 6 completion tokens" in caplog.text
    assert "0.86 target tokens per source token" in caplog.text


def test_a_dry_run_estimates_a_batched_reply_with_its_line_numbers(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, infer={"infer_batch_size": 2})

    with caplog.at_level("INFO"):
        list(model.translate(["one two three four five six seven", "a b c d e f g"], "en", "es"))

    # 12 words for the 14 source words, and 2 for the line numbers.
    assert "1 requests" in caplog.text
    assert "+ 14 completion tokens" in caplog.text


def test_a_dry_run_without_a_training_corpus_estimates_a_reply_as_long_as_its_source(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, with_corpus=False, infer={"prompt": {"num_examples": 0}})

    with caplog.at_level("INFO"):
        list(model.translate(["one two three"], "en", "es"))

    assert "+ 3 completion tokens" in caplog.text
    assert "no training corpus" in caplog.text


def test_a_dry_run_reports_the_estimated_cost(tmp_path: Path, caplog):
    prompt_words, _ = words_in_the_first_request(tmp_path / "real", ["one"])
    model, _ = make_dry_run_model(tmp_path, PricedClient())

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    # "one" is estimated to need one completion token.
    assert f"${prompt_words * 0.001 + 0.002:.4f}" in caplog.text


def test_a_dry_run_of_a_model_without_a_price_reports_no_cost(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path)

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    assert "cost unavailable" in caplog.text


def test_a_dry_run_reports_what_caching_the_shared_system_message_would_save(tmp_path: Path, caplog):
    settings = {"infer": {"prompt": {"num_examples": 1000}, "concurrency": 1}}
    (tmp_path / "real").mkdir()
    write_large_training_corpus(tmp_path / "real")
    _, system_message_words = words_in_the_first_request(tmp_path / "real", ["one"], **settings)
    write_large_training_corpus(tmp_path)
    model, _ = make_dry_run_model(tmp_path, PricedClient(), **settings)

    with caplog.at_level("INFO"):
        list(model.translate(["one", "two"], "en", "es"))

    total_line = [line for line in caplog.text.splitlines() if "total so far" in line][-1]
    uncached, cached = dollars_in(total_line)
    assert system_message_words >= 1024
    # The second request reads the system message from the cache at a tenth of the price.
    assert uncached - cached == pytest.approx(system_message_words * 0.0009, abs=0.0002)
    assert "if the provider caches the system message" in total_line


def test_a_dry_run_reports_no_cached_cost_for_a_system_message_too_short_to_cache(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, PricedClient())

    with caplog.at_level("INFO"):
        list(model.translate(["one", "two"], "en", "es"))

    assert "caches" not in caplog.text


def test_a_dry_run_reports_no_cached_cost_for_a_model_without_a_cache_price(tmp_path: Path, caplog):
    write_large_training_corpus(tmp_path)
    model, _ = make_dry_run_model(
        tmp_path, PricedClient(prices_cached_prompts=False), infer={"prompt": {"num_examples": 1000}}
    )

    with caplog.at_level("INFO"):
        list(model.translate(["one", "two"], "en", "es"))

    assert "caches" not in caplog.text


def test_a_dry_run_counts_a_request_for_each_draft(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, infer={"num_drafts": 2, "temperature": 0.8})

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es", produce_multiple_translations=True))

    assert "2 requests" in caplog.text


def test_a_dry_run_bills_nothing_for_a_segment_the_corpus_already_translates(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path)

    with caplog.at_level("INFO"):
        list(model.translate(["let there be light"], "en", "es"))

    assert "0 requests" in caplog.text
    assert "translations copied from the training corpus: 1" in caplog.text


def test_a_dry_run_keeps_a_running_total_across_translations(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path)

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))
        list(model.translate(["two"], "en", "es"))

    assert "the estimated total so far is 2 requests" in caplog.text


def test_a_dry_run_does_not_warn_of_blank_translations(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path)

    with caplog.at_level("WARNING"):
        list(model.translate(["one"], "en", "es"))

    assert "left blank" not in caplog.text


def test_a_dry_run_warns_that_its_estimate_leaves_out_reasoning(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, PricedClient(reasoning=UnknownReasoning("a-model")))

    with caplog.at_level("WARNING"):
        list(model.translate(["one"], "en", "es"))

    assert "reasoning" in caplog.text


def test_a_dry_run_of_a_model_that_cannot_reason_says_nothing_of_reasoning(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, PricedClient())

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    assert "reasoning" not in caplog.text


def test_a_dry_run_adds_estimated_reasoning_to_the_completion_tokens(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, PricedClient(reasoning=MeasuredReasoning("a-model", 4096, 2.0, "a test")))

    with caplog.at_level("INFO"):
        list(model.translate(["one two three four five six seven"], "en", "es"))

    # A 6-token reply, and twice that in reasoning.
    assert "+ 18 completion tokens" in caplog.text


def test_a_dry_run_bills_estimated_reasoning_as_output(tmp_path: Path, caplog):
    prompt_words, _ = words_in_the_first_request(tmp_path / "real", ["one"])
    model, _ = make_dry_run_model(tmp_path, PricedClient(reasoning=MeasuredReasoning("a-model", 4096, 2.0, "a test")))

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    # A 1-token reply and 2 tokens of reasoning.
    assert f"${prompt_words * 0.001 + 3 * 0.002:.4f}" in caplog.text


def test_a_dry_run_reports_how_much_of_its_total_is_measured_reasoning(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, PricedClient(reasoning=MeasuredReasoning("a-model", 4096, 2.0, "a test")))

    with caplog.at_level("INFO"):
        list(model.translate(["one two three four five six seven"], "en", "es"))
        list(model.translate(["one"], "en", "es"))

    total_line = [line for line in caplog.text.splitlines() if "total so far" in line][-1]
    assert "includes about 14 reasoning tokens" in total_line


def test_a_dry_run_with_a_thinking_budget_reports_an_upper_bound(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, PricedClient(reasoning=ReasoningBudget("a-model", 4096, 100)))

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    total_line = [line for line in caplog.text.splitlines() if "total so far" in line][-1]
    assert "+ 101 completion tokens" in total_line
    assert "up to 100 reasoning tokens, so the cost is an upper bound" in total_line


def test_a_dry_run_announces_how_it_estimates_reasoning(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(tmp_path, PricedClient(reasoning=MeasuredReasoning("a-model", 4096, 2.0, "a test")))

    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    assert "2 tokens per reply token, as measured for a test" in caplog.text


def test_a_dry_run_warns_once_that_reasoning_would_crowd_out_replies(tmp_path: Path, caplog):
    model, _ = make_dry_run_model(
        tmp_path, PricedClient(reasoning=ReasoningBudget("a-model", 50, 100)), infer={"concurrency": 1}
    )

    with caplog.at_level("WARNING"):
        list(model.translate(["one", "two"], "en", "es"))

    assert caplog.text.count("cut short or blank") == 1


def test_a_dry_run_fails_when_the_model_tokens_cannot_be_counted(tmp_path: Path):
    model, _ = make_dry_run_model(tmp_path, ScriptedClient(lambda messages: "hola", tokens_per_word=None))

    with pytest.raises(RuntimeError, match="could not be counted"):
        list(model.translate(["one"], "en", "es"))


def test_estimated_usage_fails_for_a_client_that_cannot_count_tokens():
    client = ScriptedClient(lambda messages: "hola", tokens_per_word=None)

    with pytest.raises(RuntimeError, match="could not be counted"):
        client.estimate_usage([{"role": "user", "content": "hello"}], 10)


def test_a_dry_run_bills_nothing_for_a_reply_an_earlier_run_saved(tmp_path: Path, caplog):
    real, _ = make_model(tmp_path, lambda messages: "hola", infer={"concurrency": 1})
    list(real.translate(["one"], "en", "es"))
    model, _ = make_dry_run_model(tmp_path, infer={"concurrency": 1})

    with caplog.at_level("INFO"):
        list(model.translate(["one", "two"], "en", "es"))

    assert "1 requests" in caplog.text
    assert "replies reused from an earlier run: 1" in caplog.text


def test_a_dry_run_leaves_an_earlier_runs_saved_replies_as_it_found_them(tmp_path: Path):
    real, _ = make_model(tmp_path, lambda messages: "hola")
    list(real.translate(["one"], "en", "es"))
    before = (tmp_path / "remote_llm_replies.jsonl").read_bytes()
    model, _ = make_dry_run_model(tmp_path)

    list(model.translate(["one", "two"], "en", "es"))

    assert (tmp_path / "remote_llm_replies.jsonl").read_bytes() == before


def test_a_dry_run_saves_no_replies(tmp_path: Path):
    model, _ = make_dry_run_model(tmp_path)

    list(model.translate(["one"], "en", "es"))

    assert list(tmp_path.glob("*.jsonl")) == []


def test_a_forced_dry_run_bills_every_request_but_keeps_the_saved_replies(tmp_path: Path, caplog):
    real, _ = make_model(tmp_path, lambda messages: "hola")
    list(real.translate(["one"], "en", "es"))
    model, _ = make_dry_run_model(tmp_path)

    model.discard_saved_inference()
    with caplog.at_level("INFO"):
        list(model.translate(["one"], "en", "es"))

    assert "1 requests" in caplog.text
    assert "reused" not in caplog.text
    assert (tmp_path / "remote_llm_replies.jsonl").is_file()


def test_a_dry_run_of_test_files_writes_no_predictions(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\ntwo\n", encoding="utf-8")
    model, client = make_dry_run_model(tmp_path)

    model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"])

    assert not (tmp_path / "out.txt").exists()
    assert client.calls == []


def test_a_dry_run_still_rejects_confidences_the_provider_cannot_give(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\n", encoding="utf-8")
    model, _ = make_dry_run_model(tmp_path)

    with pytest.raises(RuntimeError, match="log probabilities"):
        model.translate_test_files([tmp_path / "test.src.txt"], [tmp_path / "out.txt"], save_confidences=True)


def test_a_dry_run_of_the_test_step_writes_no_predictions_or_scores(tmp_path: Path):
    (tmp_path / "test.src.txt").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "test.trg.detok.txt").write_text("uno\ndos\n", encoding="utf-8")
    (tmp_path / "test.vref.txt").write_text("MAT 1:1\nMAT 1:2\n", encoding="utf-8")
    write_training_corpus(tmp_path)
    config = make_config(tmp_path)
    client = ScriptedClient(lambda messages: "hola")

    test_step.test(config, scorers={"bleu"}, model=DryRunRemoteLLMModel(config, ScriptedClientFactory(client)))

    assert list(tmp_path.glob("test.trg-predictions*")) == []
    assert list(tmp_path.glob("scores-*")) == []
    assert client.calls == []


def write_experiment(mt_experiments_dir: Path) -> Path:
    exp_dir = mt_experiments_dir / "exp"
    exp_dir.mkdir()
    config = {
        "model_type": "remote_llm",
        "model": "gpt-4o",
        "data": {"corpus_pairs": [], "lang_codes": {"en": "English", "es": "Spanish"}},
        "infer": {"prompt": {"num_examples": 10}},
    }
    (exp_dir / "config.yml").write_text(yaml.dump(config), encoding="utf-8")
    write_training_corpus(exp_dir)
    for side in ("src", "trg"):
        (exp_dir / f"train.{side}.detok.txt").write_text((exp_dir / f"train.{side}.txt").read_text(encoding="utf-8"))
    return exp_dir


def dry_run_translation_task(mt_experiments_dir: Path, client: ScriptedClient) -> TranslationTask:
    environment = SilNlpEnv.create_environment_with_mt_experiments_dir(mt_experiments_dir)
    config = load_config("exp", environment)
    assert isinstance(config, RemoteLLMConfig)
    model = DryRunRemoteLLMModel(config, ScriptedClientFactory(client))
    return TranslationTask(name="exp", model=model, environment=environment)


def test_a_dry_run_translation_of_a_text_file_writes_nothing(tmp_path: Path):
    exp_dir = write_experiment(tmp_path)
    (exp_dir / "source.txt").write_text("one\ntwo\n", encoding="utf-8")
    client = ScriptedClient(lambda messages: "hola")

    dry_run_translation_task(tmp_path, client).translate_files("source.txt", None, "en", "es")

    assert not (exp_dir / "infer").exists()
    assert client.calls == []


def test_a_dry_run_translation_of_a_usfm_file_writes_nothing(tmp_path: Path):
    exp_dir = write_experiment(tmp_path)
    (exp_dir / "source.SFM").write_text("\\id MAT\n\\c 1\n\\v 1 one\n\\v 2 two\n", encoding="utf-8")
    client = ScriptedClient(lambda messages: "hola")

    dry_run_translation_task(tmp_path, client).translate_files("source.SFM", "out.SFM", "en", "es", vref=True)

    assert not (exp_dir / "out.SFM").exists()
    assert not (exp_dir / "out.txt").exists()
    assert client.calls == []


def test_a_dry_run_translation_of_a_docx_file_writes_nothing(tmp_path: Path):
    docx = pytest.importorskip("docx")
    exp_dir = write_experiment(tmp_path)
    document = docx.Document()
    document.add_paragraph("One. Two.")
    document.save(str(exp_dir / "source.docx"))
    client = ScriptedClient(lambda messages: "hola")

    dry_run_translation_task(tmp_path, client).translate_files("source.docx", "out.docx", "en", "es")

    assert not (exp_dir / "out.docx").exists()
    assert client.calls == []


def test_a_dry_run_translation_of_a_folder_creates_no_output_folders(tmp_path: Path):
    exp_dir = write_experiment(tmp_path)
    (exp_dir / "sources" / "part").mkdir(parents=True)
    (exp_dir / "sources" / "part" / "source.txt").write_text("one\n", encoding="utf-8")
    client = ScriptedClient(lambda messages: "hola")

    dry_run_translation_task(tmp_path, client).translate_files("sources", "out", "en", "es")

    assert not (exp_dir / "out").exists()
    assert client.calls == []


def test_a_dry_run_translation_does_not_estimate_quality(tmp_path: Path):
    exp_dir = write_experiment(tmp_path)
    (exp_dir / "source.txt").write_text("one\n", encoding="utf-8")
    client = ScriptedClient(lambda messages: "hola")

    # Quality estimation would fail on the missing linregress file and confidences.
    dry_run_translation_task(tmp_path, client).translate_files(
        "source.txt", None, "en", "es", linregress_path=exp_dir / "linregress.json"
    )

    assert not (exp_dir / "infer").exists()
