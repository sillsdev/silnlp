import json
import logging

import numpy as np
import pytest

from silnlp.nmt.example_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    BM25ExampleRetriever,
    EmbeddingExampleRetriever,
    Example,
    ExamplePromptBuilder,
    ExampleRetriever,
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


def _fitted(retriever, examples):
    retriever.fit(examples)
    return retriever


def test_tfidf_retriever_ranks_most_similar_source_first():
    examples = _examples(
        ("the cat sat on the mat", "1"), ("completely unrelated sentence", "2"), ("a cat sat here", "3")
    )
    retriever = _fitted(TfidfExampleRetriever(), examples)
    results = retriever.retrieve("the cat sat", k=2)
    assert [ex.target for ex in results] == ["1", "3"]


def test_tfidf_retriever_respects_k():
    examples = _examples(("apple pie", "1"), ("apple pie", "2"), ("apple pie", "3"))
    retriever = _fitted(TfidfExampleRetriever(), examples)
    assert len(retriever.retrieve("apple pie", k=2)) == 2


def test_tfidf_retriever_retrieve_for_pool_index_excludes_self():
    examples = _examples(("the cat sat", "1"), ("the cat sat", "2"), ("totally different words", "3"))
    retriever = _fitted(TfidfExampleRetriever(), examples)
    results = retriever.retrieve_for_pool_index(0, k=3)
    assert examples[0] not in results
    assert len(results) == 2


def test_tfidf_retriever_k_larger_than_pool_returns_whole_pool():
    examples = _examples(("apple pie", "1"), ("banana split", "2"))
    retriever = _fitted(TfidfExampleRetriever(), examples)
    assert len(retriever.retrieve("apple pie", k=10)) == 2


def test_tfidf_retriever_empty_pool_returns_empty_list():
    retriever = _fitted(TfidfExampleRetriever(), [])
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
    retriever = _fitted(EmbeddingExampleRetriever(model=_StubEmbeddingModel(vectors)), examples)
    results = retriever.retrieve("kitten", k=1)
    assert [ex.target for ex in results] == ["chat"]


def test_embedding_retriever_retrieve_for_pool_index_excludes_self():
    vectors = {"cat": [1.0, 0.0], "kitten": [0.9, 0.1], "dog": [0.0, 1.0]}
    examples = _examples(("cat", "1"), ("kitten", "2"), ("dog", "3"))
    retriever = _fitted(EmbeddingExampleRetriever(model=_StubEmbeddingModel(vectors)), examples)
    results = retriever.retrieve_for_pool_index(0, k=2)
    assert [ex.target for ex in results] == ["2", "3"]


def test_create_example_retriever_dispatches_each_method():
    assert isinstance(create_example_retriever("tfidf"), TfidfExampleRetriever)
    assert isinstance(create_example_retriever("bm25"), BM25ExampleRetriever)
    assert isinstance(create_example_retriever("embedding"), EmbeddingExampleRetriever)


def test_create_example_retriever_is_case_insensitive():
    assert isinstance(create_example_retriever("TFIDF"), TfidfExampleRetriever)


def test_create_example_retriever_passes_the_model_name_through():
    assert create_example_retriever("embedding", "some/model").model_name == "some/model"


def test_create_example_retriever_embedding_defaults_the_model_name():
    assert create_example_retriever("embedding").model_name == DEFAULT_EMBEDDING_MODEL


def test_create_example_retriever_embedding_without_dependency_raises_clear_error():
    # sentence-transformers is optional; missing it should fail with actionable guidance.
    try:
        import sentence_transformers  # noqa: F401

        return  # dependency is installed in this environment; nothing to assert here
    except ImportError:
        pass

    with pytest.raises(ImportError, match="poetry install -E llm"):
        create_example_retriever("embedding").fit(_examples(("cat", "chat")))


def test_create_example_retriever_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown example_selection.method"):
        create_example_retriever("bogus")


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
    with pytest.raises(ValueError, match="Unknown example_format.type"):
        create_example_formatter({"type": "bogus"})


def test_create_example_formatter_accepts_bare_string_shorthand():
    assert isinstance(create_example_formatter("json"), JsonExampleFormatter)
    assert isinstance(create_example_formatter("xml"), XmlExampleFormatter)


def test_create_example_formatter_bare_text_string_matches_explicit_default():
    examples = _examples(("cat", "chat"))
    from_shorthand = create_example_formatter("text").format(examples, "English", "French")
    explicit_default = TextExampleFormatter().format(examples, "English", "French")
    assert from_shorthand == explicit_default


def _make_example_config(
    num_examples=0,
    formatter=None,
    selection_method="tfidf",
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
            "example_selection": {"method": "TFIDF", "model": None},
            "instruction_template": "{examples}{source}",
        },
        model="google/gemma-2-2b-it",
    )
    assert config.num_examples == 3
    assert isinstance(config.formatter, TextExampleFormatter)
    assert config.selection_method == "tfidf"
    assert config.selection_model is None


def test_prompt_example_config_from_params_accepts_bare_string_example_format_and_selection():
    # merge_dict() replaces rather than merges when a bare-string override lands on a dict
    # default, so both keys must accept a bare string, not only the nested-dict form.
    config = PromptExampleConfig.from_params(
        {
            "num_examples": 3,
            "example_format": "json",
            "example_selection": "embedding",
            "instruction_template": "{examples}{source}",
        },
        model="google/gemma-2-2b-it",
    )
    assert isinstance(config.formatter, JsonExampleFormatter)
    assert config.selection_method == "embedding"
    assert config.selection_model is None


def test_prompt_example_config_from_params_supports_json_format():
    config = PromptExampleConfig.from_params(
        {
            "num_examples": 3,
            "example_format": {"type": "json"},
            "example_selection": {"method": "tfidf"},
            "instruction_template": "{examples}{source}",
        },
        model="google/gemma-2-2b-it",
    )
    assert isinstance(config.formatter, JsonExampleFormatter)


def test_prompt_example_config_rejects_negative_num_examples():
    with pytest.raises(ValueError, match="non-negative"):
        _make_example_config(num_examples=-1)


def test_prompt_example_config_rejects_unknown_selection_method():
    with pytest.raises(ValueError, match="Unknown example_selection.method"):
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


class _RankedStubRetriever(ExampleRetriever):
    """Exercises the base class's generic leave-one-out, which BM25 relies on."""

    method = "stub"

    def _fit_index(self, sources):
        self._sources = sources

    def _top_indices_for_query(self, query, k):
        return list(range(len(self._sources)))[:k]


def test_generic_leave_one_out_drops_the_pool_entry_and_still_returns_k():
    retriever = _fitted(_RankedStubRetriever(), _examples(("a", "1"), ("b", "2"), ("c", "3")))
    assert [ex.target for ex in retriever.retrieve_for_pool_index(0, k=2)] == ["2", "3"]
    assert [ex.target for ex in retriever.retrieve_for_pool_index(1, k=2)] == ["1", "3"]


def test_generic_leave_one_out_returns_fewer_than_k_when_the_pool_is_too_small():
    retriever = _fitted(_RankedStubRetriever(), _examples(("a", "1"), ("b", "2")))
    assert [ex.target for ex in retriever.retrieve_for_pool_index(0, k=5)] == ["2"]


def test_embedding_retriever_pickles_without_the_model_but_keeps_the_embeddings(tmp_path):
    vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "kitten": [0.9, 0.1]}
    retriever = _fitted(
        EmbeddingExampleRetriever(model=_StubEmbeddingModel(vectors)), _examples(("cat", "chat"), ("dog", "chien"))
    )
    retriever.save(tmp_path)

    loaded = ExampleRetriever.load(tmp_path)
    assert loaded is not None
    assert loaded.examples == retriever.examples
    assert loaded._model is None
    # A query still needs the model, but ranking against the cached embeddings does not.
    loaded._model = _StubEmbeddingModel(vectors)
    assert [ex.target for ex in loaded.retrieve("kitten", k=1)] == ["chat"]


def test_retriever_meta_records_the_method_and_model_name(tmp_path):
    retriever = EmbeddingExampleRetriever("some/model", model=_StubEmbeddingModel({"a": [1.0]}))
    _fitted(retriever, _examples(("a", "1"))).save(tmp_path)
    meta = json.loads((tmp_path / "retrieval_meta.json").read_text(encoding="utf-8"))
    assert meta == {"method": "embedding", "model_name": "some/model", "num_examples": 1}
