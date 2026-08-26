import json
import logging

import numpy as np
import pytest

from silnlp.nmt.example_retrieval import (
    EmbeddingExampleRetriever,
    Example,
    ExamplePromptBuilder,
    JsonExampleFormatter,
    PromptExampleConfig,
    TextExampleFormatter,
    TfidfExampleRetriever,
    XmlExampleFormatter,
    create_example_formatter,
    create_example_retriever,
)


def _examples(*pairs):
    return [Example(source=s, target=t) for s, t in pairs]


def test_tfidf_retriever_ranks_most_similar_source_first():
    examples = _examples(
        ("the cat sat on the mat", "1"), ("completely unrelated sentence", "2"), ("a cat sat here", "3")
    )
    retriever = TfidfExampleRetriever(examples)
    results = retriever.retrieve("the cat sat", k=2)
    assert [ex.target for ex in results] == ["1", "3"]


def test_tfidf_retriever_respects_k():
    examples = _examples(("apple pie", "1"), ("apple pie", "2"), ("apple pie", "3"))
    retriever = TfidfExampleRetriever(examples)
    assert len(retriever.retrieve("apple pie", k=2)) == 2


def test_tfidf_retriever_retrieve_for_pool_index_excludes_self():
    examples = _examples(("the cat sat", "1"), ("the cat sat", "2"), ("totally different words", "3"))
    retriever = TfidfExampleRetriever(examples)
    results = retriever.retrieve_for_pool_index(0, k=3)
    assert examples[0] not in results
    assert len(results) == 2


def test_tfidf_retriever_k_larger_than_pool_returns_whole_pool():
    examples = _examples(("apple pie", "1"), ("banana split", "2"))
    retriever = TfidfExampleRetriever(examples)
    assert len(retriever.retrieve("apple pie", k=10)) == 2


def test_tfidf_retriever_empty_pool_returns_empty_list():
    retriever = TfidfExampleRetriever([])
    assert retriever.retrieve("anything", k=5) == []
    assert retriever.retrieve_for_pool_index(0, k=5) == []


class _StubEmbeddingModel:
    """A minimal stand-in for sentence_transformers.SentenceTransformer, keyed by exact text."""

    def __init__(self, vectors: dict):
        self._vectors = vectors

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        return np.array([self._vectors[t] for t in texts], dtype=np.float32)


def test_embedding_retriever_uses_injected_model_stub():
    vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "kitten": [0.9, 0.1]}
    examples = _examples(("cat", "chat"), ("dog", "chien"))
    retriever = EmbeddingExampleRetriever(examples, model=_StubEmbeddingModel(vectors))
    results = retriever.retrieve("kitten", k=1)
    assert [ex.target for ex in results] == ["chat"]


def test_embedding_retriever_retrieve_for_pool_index_excludes_self():
    vectors = {"cat": [1.0, 0.0], "kitten": [0.9, 0.1], "dog": [0.0, 1.0]}
    examples = _examples(("cat", "1"), ("kitten", "2"), ("dog", "3"))
    retriever = EmbeddingExampleRetriever(examples, model=_StubEmbeddingModel(vectors))
    results = retriever.retrieve_for_pool_index(0, k=2)
    assert [ex.target for ex in results] == ["2", "3"]


def test_create_example_retriever_dispatches_lexical():
    retriever = create_example_retriever("lexical", _examples(("cat", "chat")))
    assert isinstance(retriever, TfidfExampleRetriever)


def test_create_example_retriever_embedding_without_dependency_raises_clear_error():
    # sentence-transformers is optional; missing it should fail with actionable guidance.
    try:
        import sentence_transformers  # noqa: F401

        return  # dependency is installed in this environment; nothing to assert here
    except ImportError:
        pass

    with pytest.raises(ImportError, match="poetry install -E llm"):
        create_example_retriever("embedding", _examples(("cat", "chat")))


def test_create_example_retriever_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown example_selection.method"):
        create_example_retriever("bogus", _examples(("cat", "chat")))


def test_text_example_formatter_renders_each_example_through_the_template():
    formatter = TextExampleFormatter("Source: {source}\nTarget: {target}\n\n")
    examples = _examples(("cat", "chat"), ("dog", "chien"))
    assert formatter.format(examples, "English", "French") == (
        "Source: cat\nTarget: chat\n\nSource: dog\nTarget: chien\n\n"
    )


def test_json_example_formatter_produces_valid_escaped_json():
    formatter = JsonExampleFormatter()
    examples = _examples(('say "hi"', "chat \\ chien"), ("café", "日本語"))

    text = formatter.format(examples, "English", "French")
    parsed = json.loads(text)

    assert parsed == [{"source": 'say "hi"', "target": "chat \\ chien"}, {"source": "café", "target": "日本語"}]
    # non-ASCII text is left as-is (ensure_ascii=False), not \uXXXX-escaped
    assert "café" in text and "日本語" in text


def test_json_example_formatter_empty_examples_returns_empty_string():
    assert JsonExampleFormatter().format([], "English", "French") == ""


def test_xml_example_formatter_escapes_special_characters():
    formatter = XmlExampleFormatter()
    examples = _examples(("A < B & C > D", "target"))
    text = formatter.format(examples, "English", "French")
    assert text == "<example>\n<source>A &lt; B &amp; C &gt; D</source>\n<target>target</target>\n</example>\n"


def test_xml_example_formatter_concatenates_multiple_examples():
    formatter = XmlExampleFormatter()
    examples = _examples(("cat", "chat"), ("dog", "chien"))
    text = formatter.format(examples, "English", "French")
    assert text == (
        "<example>\n<source>cat</source>\n<target>chat</target>\n</example>\n"
        "<example>\n<source>dog</source>\n<target>chien</target>\n</example>\n"
    )


def test_create_example_formatter_dispatches_text_json_xml():
    assert isinstance(create_example_formatter({"type": "text", "template": "{source}"}), TextExampleFormatter)
    assert isinstance(create_example_formatter({"type": "json"}), JsonExampleFormatter)
    assert isinstance(create_example_formatter({"type": "xml"}), XmlExampleFormatter)


def test_create_example_formatter_defaults_to_text():
    assert isinstance(create_example_formatter({"template": "{source}"}), TextExampleFormatter)


def test_create_example_formatter_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown params.prompt.example_format.type"):
        create_example_formatter({"type": "bogus"})


def _make_example_config(
    num_examples=0,
    formatter=None,
    selection_method="lexical",
    selection_model=None,
    instruction_template="{examples}{source}",
    model="google/gemma-2-2b-it",
):
    return PromptExampleConfig(
        num_examples=num_examples,
        formatter=formatter if formatter is not None else TextExampleFormatter("{source}->{target}\n"),
        selection_method=selection_method,
        selection_model=selection_model,
        instruction_template=instruction_template,
        model=model,
    )


def test_prompt_example_config_from_params_parses_and_lowercases_method():
    config = PromptExampleConfig.from_params(
        {
            "num_examples": 3,
            "example_format": {"type": "text", "template": "{source}->{target}\n"},
            "example_selection": {"method": "LEXICAL", "model": None},
            "instruction_template": "{examples}{source}",
        },
        model="google/gemma-2-2b-it",
    )
    assert config.num_examples == 3
    assert isinstance(config.formatter, TextExampleFormatter)
    assert config.selection_method == "lexical"
    assert config.selection_model is None


def test_prompt_example_config_from_params_supports_json_format():
    config = PromptExampleConfig.from_params(
        {
            "num_examples": 3,
            "example_format": {"type": "json"},
            "example_selection": {"method": "lexical"},
            "instruction_template": "{examples}{source}",
        },
        model="google/gemma-2-2b-it",
    )
    assert isinstance(config.formatter, JsonExampleFormatter)


def test_prompt_example_config_rejects_negative_num_examples():
    with pytest.raises(ValueError, match="non-negative"):
        _make_example_config(num_examples=-1)


def test_prompt_example_config_rejects_unknown_selection_method():
    with pytest.raises(ValueError, match="Unknown params.prompt.example_selection.method"):
        _make_example_config(num_examples=1, selection_method="bogus")


def test_prompt_example_config_warns_when_placeholder_missing(caplog):
    with caplog.at_level(logging.WARNING):
        _make_example_config(num_examples=2, instruction_template="Translate: {source}")
    assert any("{examples}" in record.message for record in caplog.records)


def test_prompt_example_config_no_warning_when_placeholder_present(caplog):
    with caplog.at_level(logging.WARNING):
        _make_example_config(num_examples=2, instruction_template="{examples}{source}")
    assert caplog.records == []


def test_prompt_example_config_no_warning_when_disabled(caplog):
    with caplog.at_level(logging.WARNING):
        _make_example_config(num_examples=0, instruction_template="Translate: {source}")
    assert caplog.records == []


def test_prompt_example_config_rejects_translate_gemma_when_enabled():
    with pytest.raises(RuntimeError, match="TranslateGemma"):
        _make_example_config(num_examples=2, model="google/translategemma-4b-it")


def test_prompt_example_config_allows_translate_gemma_when_disabled():
    _make_example_config(num_examples=0, model="google/translategemma-4b-it")  # no raise


def test_example_prompt_builder_returns_empty_string_and_touches_no_files_when_disabled(tmp_path):
    config = _make_example_config(num_examples=0)
    builder = ExamplePromptBuilder(config, tmp_path / "missing.src.txt", tmp_path / "missing.trg.txt")
    assert builder.render("hello", "English", "French") == ""


def test_example_prompt_builder_renders_retrieved_examples(tmp_path):
    src_path = tmp_path / "train.src.txt"
    trg_path = tmp_path / "train.trg.txt"
    src_path.write_text("the cat sat\nsomething unrelated\n", encoding="utf-8")
    trg_path.write_text("le chat assis\nquelque chose\n", encoding="utf-8")
    config = _make_example_config(
        num_examples=1, formatter=TextExampleFormatter("Source: {source}\nTarget: {target}\n\n")
    )
    builder = ExamplePromptBuilder(config, src_path, trg_path)

    text = builder.render("the cat sat here", "English", "French")

    assert text == "Source: the cat sat\nTarget: le chat assis\n\n"


def test_example_prompt_builder_supports_json_formatter(tmp_path):
    src_path = tmp_path / "train.src.txt"
    trg_path = tmp_path / "train.trg.txt"
    src_path.write_text("the cat sat\n", encoding="utf-8")
    trg_path.write_text("le chat assis\n", encoding="utf-8")
    config = _make_example_config(num_examples=1, formatter=JsonExampleFormatter())
    builder = ExamplePromptBuilder(config, src_path, trg_path)

    text = builder.render("the cat sat here", "English", "French")

    assert json.loads(text) == [{"source": "the cat sat", "target": "le chat assis"}]


def test_example_prompt_builder_leave_one_out_by_pool_index(tmp_path):
    src_path = tmp_path / "train.src.txt"
    trg_path = tmp_path / "train.trg.txt"
    src_path.write_text("the cat sat\nthe cat sat\ntotally different words\n", encoding="utf-8")
    trg_path.write_text("1\n2\n3\n", encoding="utf-8")
    config = _make_example_config(num_examples=2)
    builder = ExamplePromptBuilder(config, src_path, trg_path)

    text = builder.render("unused", "English", "French", pool_index=0)

    assert text == "the cat sat->2\ntotally different words->3\n"


def test_example_prompt_builder_raises_clear_error_when_corpus_missing(tmp_path):
    config = _make_example_config(num_examples=1, formatter=TextExampleFormatter("{source}"))
    builder = ExamplePromptBuilder(config, tmp_path / "missing.src.txt", tmp_path / "missing.trg.txt")

    with pytest.raises(RuntimeError, match="preprocessing"):
        builder.render("hello", "English", "French")
