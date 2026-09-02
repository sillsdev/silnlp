import json
import logging

import pytest

from silnlp.nmt.config import Language
from silnlp.nmt.example_retrieval import Example, TextExampleFormatter, create_example_formatter
from silnlp.nmt.llm_config import (
    LLMConfig,
    PromptBuilder,
    PromptMessages,
    PromptTemplate,
    parse_example_selection,
    parse_num_examples,
    read_prompt_template_file,
    resolve_prompt_defaults,
    warn_about_examples_placeholder,
)

EN = Language("en", "English")
FR = Language("fr", "French")


def _template(instruction_template="{source}", system_message="", example_format="text"):
    return PromptTemplate(system_message, instruction_template, create_example_formatter(example_format))


class _FakePool:
    def __init__(self, examples=(), whole=False):
        self._examples = list(examples)
        self._whole = whole
        self.calls = []

    def __len__(self):
        return len(self._examples)

    def covers_whole_pool(self, k):
        return self._whole

    def select(self, query, k, pool_index=None):
        self.calls.append((query, k, pool_index))
        return self._examples


# --- PromptMessages ----------------------------------------------------------------------


def test_prompt_messages_include_the_system_and_assistant_turns():
    prompt = PromptMessages("Be terse.", "Translate: hello", "bonjour")
    assert prompt.to_chat_messages() == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Translate: hello"},
        {"role": "assistant", "content": "bonjour"},
    ]


def test_prompt_messages_fold_the_system_message_into_the_user_turn():
    prompt = PromptMessages("Be terse.", "Translate: hello")
    assert prompt.to_folded_chat_messages() == [{"role": "user", "content": "Be terse.\n\nTranslate: hello"}]


# --- PromptBuilder -----------------------------------------------------------------------


def test_prompt_builder_fills_the_instruction_template():
    builder = PromptBuilder([_template("Translate {src_lang} to {trg_lang}: {source}")], 0, None)
    assert builder.build("hello", EN, FR).instruction == "Translate English to French: hello"


def test_prompt_builder_formats_the_system_message_with_the_languages():
    builder = PromptBuilder([_template(system_message="{src_lang} into {trg_lang}")], 0, None)
    assert builder.build("hello", EN, FR).system_message == "English into French"


def test_prompt_builder_renders_examples_into_the_placeholder():
    pool = _FakePool([Example("cat", "chat")])
    builder = PromptBuilder(
        [_template("{examples}{source}", example_format={"type": "text", "template": "{source}->{target}\n"})], 1, pool
    )
    assert builder.build("hello", EN, FR).instruction == "cat->chat\nhello"
    assert pool.calls == [("hello", 1, None)]


def test_prompt_builder_skips_the_pool_when_no_examples_are_asked_for():
    pool = _FakePool([Example("cat", "chat")])
    builder = PromptBuilder([_template("{examples}{source}")], 0, pool)
    assert builder.build("hello", EN, FR).instruction == "hello"
    assert pool.calls == []


def test_prompt_builder_requires_at_least_one_template():
    with pytest.raises(ValueError, match="at least one template"):
        PromptBuilder([], 0, None)


def test_prompt_builder_rotates_templates_by_pool_index():
    builder = PromptBuilder([_template("A: {source}"), _template("B: {source}")], 0, None)
    assert [builder.build("x", EN, FR, pool_index=i).instruction for i in range(4)] == [
        "A: x",
        "B: x",
        "A: x",
        "B: x",
    ]


def test_prompt_builder_uses_the_first_template_when_there_is_no_pool_index():
    builder = PromptBuilder([_template("A: {source}"), _template("B: {source}")], 0, None)
    assert builder.build("x", EN, FR).instruction == "A: x"


def test_prompt_builder_reports_whether_it_covers_the_whole_pool():
    assert not PromptBuilder([_template()], 5, None).covers_whole_pool()
    assert PromptBuilder([_template()], 5, _FakePool(whole=True)).covers_whole_pool()


def test_prompt_template_reports_its_examples_placeholder():
    assert _template("{examples}{source}").has_examples_placeholder
    assert not _template("{source}").has_examples_placeholder


# --- config parsing ----------------------------------------------------------------------


def test_parse_example_selection_lowercases_the_method_and_reads_the_model():
    assert parse_example_selection({"example_selection": {"method": "TFIDF", "model": "m"}}) == ("tfidf", "m")


def test_parse_example_selection_accepts_a_bare_string():
    # merge_dict() replaces rather than merges when a bare-string override lands on a dict default.
    assert parse_example_selection({"example_selection": "embedding"}) == ("embedding", None)


def test_parse_example_selection_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown example_selection.method"):
        parse_example_selection({"example_selection": {"method": "bogus"}})


def test_parse_num_examples_rejects_a_negative_count():
    with pytest.raises(ValueError, match="non-negative"):
        parse_num_examples({"num_examples": -1}, "train.prompt")


# --- default resolution ------------------------------------------------------------------


def test_resolve_prompt_defaults_uses_the_zero_shot_template_without_examples():
    prompt = {"system_message": None, "instruction_template": None, "example_format": None, "num_examples": 0}
    resolve_prompt_defaults(prompt, LLMConfig)
    assert prompt["instruction_template"] == LLMConfig.DEFAULT_INSTRUCTION_TEMPLATE
    assert "{examples}" not in prompt["instruction_template"]
    assert prompt["system_message"] == LLMConfig.DEFAULT_SYSTEM_MESSAGE
    assert prompt["example_format"] == LLMConfig.DEFAULT_EXAMPLE_FORMAT


def test_resolve_prompt_defaults_uses_the_few_shot_template_with_examples():
    prompt = {"system_message": None, "instruction_template": None, "example_format": None, "num_examples": 3}
    resolve_prompt_defaults(prompt, LLMConfig)
    assert prompt["instruction_template"] == LLMConfig.DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE
    assert "{examples}" in prompt["instruction_template"]


def test_resolve_prompt_defaults_leaves_the_user_wording_alone():
    prompt = {"system_message": "mine", "instruction_template": "{source}", "example_format": "json", "num_examples": 3}
    resolve_prompt_defaults(prompt, LLMConfig)
    assert prompt == {
        "system_message": "mine",
        "instruction_template": "{source}",
        "example_format": "json",
        "num_examples": 3,
    }


def test_the_two_default_templates_agree_apart_from_the_examples_block():
    zero_shot = LLMConfig.DEFAULT_INSTRUCTION_TEMPLATE.format(src_lang="English", trg_lang="French", source="hello")
    few_shot = LLMConfig.DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE.format(
        src_lang="English", trg_lang="French", source="hello", examples="EXAMPLES\n\n"
    )
    assert zero_shot.startswith("Translate the following text from English to French.")
    assert "EXAMPLES" in few_shot
    assert few_shot.endswith("hello")


# --- placeholder warnings ----------------------------------------------------------------


def test_warns_when_examples_are_requested_but_the_template_has_no_placeholder(caplog):
    with caplog.at_level(logging.WARNING):
        warn_about_examples_placeholder([_template("{source}")], 2, "train.prompt.instruction_template")
    assert any("silently discarded" in record.message for record in caplog.records)


def test_warns_when_the_template_has_a_placeholder_but_no_examples_are_requested(caplog):
    with caplog.at_level(logging.WARNING):
        warn_about_examples_placeholder([_template("{examples}{source}")], 0, "train.prompt.instruction_template")
    assert any("always renders as nothing" in record.message for record in caplog.records)


def test_does_not_warn_when_the_template_and_the_count_agree(caplog):
    with caplog.at_level(logging.WARNING):
        warn_about_examples_placeholder([_template("{examples}{source}")], 2, "x")
        warn_about_examples_placeholder([_template("{source}")], 0, "x")
    assert caplog.records == []


def test_warning_names_the_offending_template_when_there_are_several(caplog):
    with caplog.at_level(logging.WARNING):
        warn_about_examples_placeholder([_template("{examples}{source}"), _template("{source}")], 2, "templates.jsonl")
    assert any("templates.jsonl[1]" in record.message for record in caplog.records)


# --- template files ----------------------------------------------------------------------


DEFAULTS = {"system_message": "default system", "instruction_template": "default {source}", "example_format": "text"}


def _write_templates(path, entries):
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def test_read_prompt_template_file_reads_one_template_per_line(tmp_path):
    path = _write_templates(
        tmp_path / "templates.jsonl",
        [
            {"system_message": "one", "instruction_template": "A: {source}", "example_format": "json"},
            {"system_message": "two", "instruction_template": "B: {source}", "example_format": "xml"},
        ],
    )
    templates = read_prompt_template_file(path, DEFAULTS)
    assert [t.system_message for t in templates] == ["one", "two"]
    assert [t.instruction_template for t in templates] == ["A: {source}", "B: {source}"]


def test_read_prompt_template_file_falls_back_to_the_defaults_per_field(tmp_path):
    path = _write_templates(tmp_path / "templates.jsonl", [{"instruction_template": "A: {source}"}])
    template = read_prompt_template_file(path, DEFAULTS)[0]
    assert template.system_message == "default system"
    assert isinstance(template.formatter, TextExampleFormatter)


def test_read_prompt_template_file_ignores_blank_lines(tmp_path):
    path = tmp_path / "templates.jsonl"
    path.write_text('\n{"instruction_template": "A: {source}"}\n\n', encoding="utf-8")
    assert len(read_prompt_template_file(path, DEFAULTS)) == 1


def test_read_prompt_template_file_rejects_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        read_prompt_template_file(tmp_path / "nowhere.jsonl", DEFAULTS)


def test_read_prompt_template_file_rejects_an_empty_file(tmp_path):
    path = tmp_path / "templates.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no templates"):
        read_prompt_template_file(path, DEFAULTS)


def test_read_prompt_template_file_reports_the_line_of_a_syntax_error(tmp_path):
    path = tmp_path / "templates.jsonl"
    path.write_text('{"instruction_template": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="line 2 is not valid JSON"):
        read_prompt_template_file(path, DEFAULTS)


def test_read_prompt_template_file_rejects_a_non_object_line(tmp_path):
    path = tmp_path / "templates.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be a JSON object"):
        read_prompt_template_file(path, DEFAULTS)


def test_read_prompt_template_file_rejects_unknown_fields(tmp_path):
    path = _write_templates(tmp_path / "templates.jsonl", [{"num_examples": 3}])
    with pytest.raises(RuntimeError, match="unknown field"):
        read_prompt_template_file(path, DEFAULTS)
