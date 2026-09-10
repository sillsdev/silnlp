import json
import logging

import pytest

from silnlp.nmt.config import Language
from silnlp.nmt.example_retrieval import Example, ExampleFormatterFactory
from silnlp.nmt.llm_config import (
    LLMConfig,
    PromptBuilder,
    PromptMessages,
    PromptTemplate,
    PlainPromptMessagesFactory,
    PromptConfig,
    PromptDefaults,
    PromptTemplateCollection,
)

EN = Language("en", "English")
FR = Language("fr", "French")


def _template(instruction_template="{source}", system_message="", example_format="text"):
    return PromptTemplate(system_message, instruction_template, ExampleFormatterFactory.create(example_format))


def _collection(*templates):
    return PromptTemplateCollection(list(templates))


def _builder(templates, num_examples=0, pool=None):
    return PromptBuilder(templates, num_examples, pool, PlainPromptMessagesFactory())


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
    builder = _builder(_collection(_template("Translate {src_lang} to {trg_lang}: {source}")), 0, None)
    assert builder.build("hello", EN, FR).instruction == "Translate English to French: hello"


def test_prompt_builder_formats_the_system_message_with_the_languages():
    builder = _builder(_collection(_template(system_message="{src_lang} into {trg_lang}")), 0, None)
    assert builder.build("hello", EN, FR).system_message == "English into French"


def test_prompt_builder_renders_examples_into_the_placeholder():
    pool = _FakePool([Example("cat", "chat")])
    builder = _builder(
        _collection(_template("{examples}{source}", example_format={"template": "{source}->{target}\n"})),
        1,
        pool,
    )
    assert builder.build("hello", EN, FR).instruction == "cat->chat\nhello"
    assert pool.calls == [("hello", 1, None)]


def test_prompt_builder_skips_the_pool_when_no_examples_are_asked_for():
    pool = _FakePool([Example("cat", "chat")])
    builder = _builder(_collection(_template("{examples}{source}")), 0, pool)
    assert builder.build("hello", EN, FR).instruction == "hello"
    assert pool.calls == []


def test_prompt_builder_requires_at_least_one_template():
    with pytest.raises(ValueError, match="No valid prompt templates"):
        _builder(PromptTemplateCollection([]), 0, None)


def test_prompt_builder_rotates_templates_by_pool_index():
    builder = _builder(_collection(_template("A: {source}"), _template("B: {source}")), 0, None)
    assert [builder.build("x", EN, FR, pool_index=i).instruction for i in range(4)] == [
        "A: x",
        "B: x",
        "A: x",
        "B: x",
    ]


def test_prompt_builder_uses_the_first_template_when_there_is_no_pool_index():
    builder = _builder(_collection(_template("A: {source}"), _template("B: {source}")), 0, None)
    assert builder.build("x", EN, FR).instruction == "A: x"


def test_prompt_builder_reports_whether_it_covers_the_whole_pool():
    assert not _builder(_collection(_template()), 5, None).covers_whole_pool()
    assert _builder(_collection(_template()), 5, _FakePool(whole=True)).covers_whole_pool()


def test_prompt_template_reports_its_examples_placeholder():
    assert _template("{examples}{source}").describe_examples_mismatch(2) is None
    assert _template("{source}").describe_examples_mismatch(2) is not None


# --- config parsing ----------------------------------------------------------------------


def _prompt_config(**settings):
    base = {
        "num_examples": 0,
        "example_selection": {"method": "tfidf", "model": None},
        "system_message": "",
        "instruction_template": "{source}",
        "example_format": "text",
    }
    base.update(settings)
    return PromptConfig(base, "train.prompt")


def test_prompt_config_creates_the_configured_retriever():
    retriever = _prompt_config(example_selection={"method": "bm25", "model": None}).create_retriever()
    assert retriever.method == "bm25"


def test_prompt_config_accepts_a_bare_string_selection():
    # merge_dict() replaces rather than merges when a bare-string override lands on a dict default.
    assert _prompt_config(example_selection="embedding").create_retriever().method == "embedding"


def test_prompt_config_rejects_a_negative_example_count():
    with pytest.raises(ValueError, match="non-negative"):
        _prompt_config(num_examples=-1).get_num_examples()


# --- default resolution ------------------------------------------------------------------


DEFAULTS = PromptDefaults(
    LLMConfig.DEFAULT_SYSTEM_MESSAGE,
    LLMConfig.DEFAULT_INSTRUCTION_TEMPLATE,
    LLMConfig.DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE,
    LLMConfig.DEFAULT_EXAMPLE_FORMAT,
)


def _resolved(**settings):
    prompt = {"system_message": None, "instruction_template": None, "example_format": None, **settings}
    PromptConfig(prompt, "train.prompt").resolve_defaults(DEFAULTS)
    return prompt


def test_resolve_prompt_defaults_uses_the_zero_shot_template_without_examples():
    prompt = _resolved(num_examples=0)
    assert prompt["instruction_template"] == LLMConfig.DEFAULT_INSTRUCTION_TEMPLATE
    assert "{examples}" not in prompt["instruction_template"]
    assert prompt["system_message"] == LLMConfig.DEFAULT_SYSTEM_MESSAGE
    assert prompt["example_format"] == LLMConfig.DEFAULT_EXAMPLE_FORMAT


def test_resolve_prompt_defaults_uses_the_few_shot_template_with_examples():
    prompt = _resolved(num_examples=3)
    assert prompt["instruction_template"] == LLMConfig.DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE
    assert "{examples}" in prompt["instruction_template"]


def test_resolve_prompt_defaults_leaves_the_user_wording_alone():
    prompt = {"system_message": "mine", "instruction_template": "{source}", "example_format": "json", "num_examples": 3}
    PromptConfig(prompt, "train.prompt").resolve_defaults(DEFAULTS)
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
        PromptTemplateCollection([_template("{source}")]).validate_for_icl(2, "train.prompt.instruction_template")
    assert any("silently discarded" in record.message for record in caplog.records)


def test_warns_when_the_template_has_a_placeholder_but_no_examples_are_requested(caplog):
    with caplog.at_level(logging.WARNING):
        collection = PromptTemplateCollection([_template("{examples}{source}")])
        collection.validate_for_icl(0, "train.prompt.instruction_template")
    assert any("always renders as nothing" in record.message for record in caplog.records)


def test_does_not_warn_when_the_template_and_the_count_agree(caplog):
    with caplog.at_level(logging.WARNING):
        PromptTemplateCollection([_template("{examples}{source}")]).validate_for_icl(2, "x")
        PromptTemplateCollection([_template("{source}")]).validate_for_icl(0, "x")
    assert caplog.records == []


def test_warning_names_the_offending_template_when_there_are_several(caplog):
    with caplog.at_level(logging.WARNING):
        collection = PromptTemplateCollection([_template("{examples}{source}"), _template("{source}")])
        collection.validate_for_icl(2, "templates.jsonl")
    assert any("templates.jsonl[1]" in record.message for record in caplog.records)


# --- template files ----------------------------------------------------------------------


def _entry(**overrides):
    entry = {"system_message": "sys", "instruction_template": "{source}", "example_format": "text"}
    entry.update(overrides)
    return entry


def _write_templates(path, entries):
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def _templates_from(path, count):
    collection = PromptTemplateCollection.from_file(path)
    return [collection.template_for(i) for i in range(count)]


def test_prompt_template_file_reads_one_template_per_line(tmp_path):
    path = _write_templates(
        tmp_path / "templates.jsonl",
        [
            _entry(system_message="one", instruction_template="A: {source}", example_format="json"),
            _entry(system_message="two", instruction_template="B: {source}", example_format="xml"),
        ],
    )
    templates = _templates_from(path, 2)
    assert [t.system_message for t in templates] == ["one", "two"]
    assert [t.instruction_template for t in templates] == ["A: {source}", "B: {source}"]


def test_prompt_template_file_keeps_an_explicit_empty_field(tmp_path):
    path = _write_templates(tmp_path / "templates.jsonl", [_entry(system_message="")])
    assert PromptTemplateCollection.from_file(path).template_for(None).system_message == ""


def test_prompt_template_file_ignores_blank_lines(tmp_path):
    path = tmp_path / "templates.jsonl"
    path.write_text("\n" + json.dumps(_entry(instruction_template="A: {source}")) + "\n\n", encoding="utf-8")
    assert _templates_from(path, 1)[0].instruction_template == "A: {source}"


def test_prompt_template_file_rejects_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        PromptTemplateCollection.from_file(tmp_path / "nowhere.jsonl")


def test_prompt_template_file_rejects_an_empty_file(tmp_path):
    path = tmp_path / "templates.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no templates"):
        PromptTemplateCollection.from_file(path)


@pytest.mark.parametrize(
    "bad_line, message",
    [
        ("not json", "not valid JSON"),
        ('["not", "an", "object"]', "must be a JSON object"),
        ('{"num_examples": 3}', "unknown field"),
        (json.dumps({"system_message": "s", "instruction_template": "t"}), "missing required field"),
        (json.dumps({"system_message": "s", "example_format": "text"}), "missing required field"),
        (json.dumps(_entry(example_format="bogus")), "invalid example_format"),
    ],
)
def test_prompt_template_file_skips_a_bad_line_and_says_why(tmp_path, caplog, bad_line, message):
    path = tmp_path / "templates.jsonl"
    path.write_text(bad_line + "\n" + json.dumps(_entry(instruction_template="ok")) + "\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        collection = PromptTemplateCollection.from_file(path)

    assert collection.template_for(None).instruction_template == "ok"
    assert any(message in record.message for record in caplog.records)


def test_prompt_template_file_names_every_missing_field(tmp_path, caplog):
    path = _write_templates(tmp_path / "templates.jsonl", [{"system_message": "s"}, _entry()])
    with caplog.at_level(logging.WARNING):
        PromptTemplateCollection.from_file(path)
    warning = next(r.message for r in caplog.records if "missing required field" in r.message)
    assert "instruction_template" in warning and "example_format" in warning


def test_prompt_builder_rotation_index_overrides_the_pool_index():
    pool = _FakePool([Example("cat", "chat")])
    builder = _builder(_collection(_template("A: {source}"), _template("B: {source}")), 1, pool)
    # The pool entry is still excluded from its own examples even though another template is used.
    assert builder.build("x", EN, FR, pool_index=0, rotation_index=1).instruction == "B: x"
    assert pool.calls == [("x", 1, 0)]


def test_prompt_builder_rotates_rows_outside_the_pool():
    builder = _builder(_collection(_template("A: {source}"), _template("B: {source}")), 0, None)
    assert [builder.build("x", EN, FR, rotation_index=i).instruction for i in range(3)] == ["A: x", "B: x", "A: x"]


def test_prompt_builder_rotation_index_zero_is_not_treated_as_unset():
    builder = _builder(_collection(_template("A: {source}"), _template("B: {source}")), 0, None)
    assert builder.build("x", EN, FR, pool_index=1, rotation_index=0).instruction == "A: x"
