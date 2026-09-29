import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from silnlp.nmt.example_retrieval import (
    BM25ExampleRetriever,
    CorpusPair,
    CorpusPairProvider,
    EmbeddingExampleRetriever,
    Example,
    ExampleFormatterFactory,
    ExamplePool,
    ExampleRetriever,
    ExampleRetrieverFactory,
    FixedCorpusPairProvider,
    JsonExampleFormatter,
    PreferredCorpusPairProvider,
    RetrievalTokenizer,
    TextExampleFormatter,
    TfidfExampleRetriever,
    XmlExampleFormatter,
)


def _examples(*pairs):
    return [Example(source=s, target=t) for s, t in pairs]


def _fitted(retriever, sources):
    retriever.fit(sources)
    return retriever


def test_tfidf_retriever_ranks_most_similar_source_first():
    sources = ["the cat sat on the mat", "completely unrelated sentence", "a cat sat here"]
    assert _fitted(TfidfExampleRetriever(), sources).rank("the cat sat", k=2) == [0, 2]


def test_tfidf_retriever_respects_k():
    retriever = _fitted(TfidfExampleRetriever(), ["apple pie"] * 3)
    assert len(retriever.rank("apple pie", k=2)) == 2


def test_tfidf_retriever_rank_excluding_leaves_out_its_own_position():
    sources = ["the cat sat", "the cat sat", "totally different words"]
    ranked = _fitted(TfidfExampleRetriever(), sources).rank_excluding(sources[0], 0, k=3)
    assert ranked == [1, 2]


def test_tfidf_retriever_k_larger_than_the_corpus_returns_every_position():
    retriever = _fitted(TfidfExampleRetriever(), ["apple pie", "banana split"])
    assert len(retriever.rank("apple pie", k=10)) == 2


def test_tfidf_retriever_empty_corpus_returns_no_positions():
    retriever = _fitted(TfidfExampleRetriever(), [])
    assert retriever.rank("anything", k=5) == []
    assert retriever.rank_excluding("anything", 0, k=5) == []


class _StubEmbeddingModel:
    """A minimal stand-in for sentence_transformers.SentenceTransformer, keyed by exact text."""

    def __init__(self, vectors: dict):
        self._vectors = vectors
        self.encode_calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        self.encode_calls += 1
        return np.array([self._vectors[t] for t in texts], dtype=np.float32)


def test_embedding_retriever_uses_injected_model_stub():
    vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "kitten": [0.9, 0.1]}
    retriever = _fitted(EmbeddingExampleRetriever(model=_StubEmbeddingModel(vectors)), ["cat", "dog"])
    assert retriever.rank("kitten", k=1) == [0]


def test_embedding_retriever_rank_excluding_leaves_out_its_own_position():
    vectors = {"cat": [1.0, 0.0], "kitten": [0.9, 0.1], "dog": [0.0, 1.0]}
    sources = ["cat", "kitten", "dog"]
    retriever = _fitted(EmbeddingExampleRetriever(model=_StubEmbeddingModel(vectors)), sources)
    assert retriever.rank_excluding(sources[0], 0, k=2) == [1, 2]


def test_create_example_retriever_dispatches_each_method():
    assert isinstance(ExampleRetrieverFactory.create("tfidf"), TfidfExampleRetriever)
    assert isinstance(ExampleRetrieverFactory.create("bm25"), BM25ExampleRetriever)
    assert isinstance(ExampleRetrieverFactory.create("embedding"), EmbeddingExampleRetriever)


def test_create_example_retriever_is_case_insensitive():
    assert isinstance(ExampleRetrieverFactory.create("TFIDF"), TfidfExampleRetriever)


def test_create_example_retriever_passes_the_model_name_through(tmp_path):
    retriever = EmbeddingExampleRetriever("some/model", model=_StubEmbeddingModel({"a": [1.0]}))
    _fitted(retriever, ["a"]).save(tmp_path)
    meta = json.loads((tmp_path / "example_retrieval_meta.json").read_text(encoding="utf-8"))
    assert meta["model_name"] == "some/model"


def test_embedding_retriever_does_not_touch_its_model_until_it_is_fitted():
    # Loading the model downloads it, so a config that never draws examples must not trigger one.
    stub = _StubEmbeddingModel({"cat": [1.0]})
    retriever = EmbeddingExampleRetriever(model=stub)
    assert stub.encode_calls == 0

    retriever.fit(["cat"])
    assert stub.encode_calls == 1


def test_create_example_retriever_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown example_selection.method"):
        ExampleRetrieverFactory.create("bogus")


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
    assert isinstance(ExampleFormatterFactory.create({"type": "text", "template": "{source}"}), TextExampleFormatter)
    assert isinstance(ExampleFormatterFactory.create({"type": "json"}), JsonExampleFormatter)
    assert isinstance(ExampleFormatterFactory.create({"type": "xml"}), XmlExampleFormatter)


def test_create_example_formatter_defaults_to_text():
    assert isinstance(ExampleFormatterFactory.create({"template": "{source}"}), TextExampleFormatter)


def test_create_example_formatter_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown example_format.type"):
        ExampleFormatterFactory.create({"type": "bogus"})


def test_create_example_formatter_accepts_bare_string_shorthand():
    assert isinstance(ExampleFormatterFactory.create("json"), JsonExampleFormatter)
    assert isinstance(ExampleFormatterFactory.create("xml"), XmlExampleFormatter)


def test_create_example_formatter_bare_text_string_matches_explicit_default():
    examples = _examples(("cat", "chat"))
    from_shorthand = ExampleFormatterFactory.create("text").format(examples, "English", "French")
    explicit_default = TextExampleFormatter().format(examples, "English", "French")
    assert from_shorthand == explicit_default


class _NeverFitRetriever(ExampleRetriever):
    method = "never-fit"

    def _fit_index(self, sources):
        raise AssertionError("the pool built an index it did not need")

    def _top_indices_for_query(self, query, k):
        raise AssertionError("the pool queried an index it did not need")


def _write_pool(tmp_path, sources, targets, retriever=None):
    src_path = tmp_path / "train.src.txt"
    trg_path = tmp_path / "train.trg.txt"
    src_path.write_text("".join(line + "\n" for line in sources), encoding="utf-8")
    trg_path.write_text("".join(line + "\n" for line in targets), encoding="utf-8")
    return ExamplePool(
        FixedCorpusPairProvider(CorpusPair(src_path, trg_path)),
        retriever if retriever is not None else ExampleRetrieverFactory.create("tfidf"),
    )


def test_example_pool_selects_the_most_relevant_example_last(tmp_path):
    pool = _write_pool(
        tmp_path,
        ["the cat sat", "a cat sat here", "something unrelated"],
        ["le chat assis", "un chat assis", "quelque chose"],
    )
    assert [ex.target for ex in pool.select("the cat sat", k=2)] == ["un chat assis", "le chat assis"]


def test_example_pool_returns_nothing_and_touches_no_files_when_k_is_zero(tmp_path):
    pool = ExamplePool(
        FixedCorpusPairProvider(CorpusPair(tmp_path / "missing.src.txt", tmp_path / "missing.trg.txt")),
        ExampleRetrieverFactory.create("tfidf"),
    )
    assert pool.select("hello", k=0) == []


def test_example_pool_leave_one_out_excludes_the_pool_entry(tmp_path):
    pool = _write_pool(tmp_path, ["the cat sat", "the cat sat", "totally different words"], ["1", "2", "3"])
    assert [ex.target for ex in pool.select("unused", k=2, pool_index=0)] == ["3", "2"]


def test_example_pool_uses_the_whole_pool_in_corpus_order_when_k_covers_it(tmp_path):
    pool = _write_pool(tmp_path, ["b sentence", "a sentence", "c sentence"], ["1", "2", "3"])
    assert [ex.target for ex in pool.select("a sentence", k=99)] == ["1", "2", "3"]
    assert pool.covers_whole_pool(99)


def test_example_pool_whole_pool_path_still_leaves_out_the_pool_entry(tmp_path):
    pool = _write_pool(tmp_path, ["one", "two", "three"], ["1", "2", "3"])
    assert [ex.target for ex in pool.select("unused", k=99, pool_index=1)] == ["1", "3"]


def test_example_pool_whole_pool_path_builds_no_index(tmp_path):
    pool = _write_pool(tmp_path, ["one", "two"], ["1", "2"], retriever=_NeverFitRetriever())
    assert len(pool.select("one", k=99)) == 2


def test_example_pool_raises_a_clear_error_when_the_corpus_is_missing(tmp_path):
    pool = ExamplePool(
        FixedCorpusPairProvider(CorpusPair(tmp_path / "missing.src.txt", tmp_path / "missing.trg.txt")),
        ExampleRetrieverFactory.create("tfidf"),
    )
    with pytest.raises(RuntimeError, match="preprocessing"):
        pool.select("hello", k=1)


_EMBEDDING_VECTORS = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "kitten": [0.9, 0.1]}


def _embedding_pool(tmp_path, model_name=None, model=None, sources=("cat", "dog")):
    retriever = EmbeddingExampleRetriever(model_name, model or _StubEmbeddingModel(_EMBEDDING_VECTORS))
    return _write_pool(tmp_path, list(sources), [str(i) for i in range(1, len(sources) + 1)], retriever=retriever)


def test_example_pool_saves_and_reloads_its_index(tmp_path):
    _embedding_pool(tmp_path).save_index(tmp_path)

    model = _StubEmbeddingModel(_EMBEDDING_VECTORS)
    reloaded = _embedding_pool(tmp_path, model=model)
    assert reloaded.load_index(tmp_path)
    assert [ex.target for ex in reloaded.select("kitten", k=1)] == ["1"]
    # Only the query was encoded: the corpus came back from the saved vectors.
    assert model.encode_calls == 1


def test_example_pool_rejects_a_saved_index_built_with_another_method(tmp_path):
    _embedding_pool(tmp_path).save_index(tmp_path)
    assert not _write_pool(tmp_path, ["cat", "dog"], ["1", "2"]).load_index(tmp_path)


def test_example_pool_rejects_a_saved_index_built_with_another_embedding_model(tmp_path):
    _embedding_pool(tmp_path, model_name="some/model").save_index(tmp_path)
    assert not _embedding_pool(tmp_path, model_name="another/model").load_index(tmp_path)


def test_example_pool_rejects_a_saved_index_whose_corpus_has_changed_size(tmp_path):
    _embedding_pool(tmp_path).save_index(tmp_path)
    assert not _embedding_pool(tmp_path, sources=("cat", "dog", "kitten")).load_index(tmp_path)


def test_example_pool_reports_a_missing_index(tmp_path):
    assert not _embedding_pool(tmp_path).load_index(tmp_path / "nowhere")


def test_example_pool_rebuilds_rather_than_fail_on_an_unreadable_index(tmp_path):
    _embedding_pool(tmp_path).save_index(tmp_path)
    (tmp_path / "example_retrieval_embeddings.npy").write_bytes(b"not an npy file")
    assert not _embedding_pool(tmp_path).load_index(tmp_path)


# Long enough that a thread doing the lazy work gives up the GIL and the others reach the same
# check. Without it they finish inside one slice and an unguarded lazy init still looks correct.
_LAZY_WORK_SECONDS = 0.1


class _CountingFitRetriever(ExampleRetriever):
    """Counts indexing runs, so a lazy fit repeated per thread is visible."""

    method = "counting"

    def __init__(self):
        super().__init__()
        self.fit_calls = 0
        self._sources = []
        self._counter = threading.Lock()

    def _fit_index(self, sources):
        with self._counter:
            self.fit_calls += 1
        time.sleep(_LAZY_WORK_SECONDS)
        self._sources = sources

    def _top_indices_for_query(self, query, k):
        return list(range(len(self._sources)))[:k]


class _CountingCorpusPairProvider(CorpusPairProvider):
    """Counts corpus reads, so a lazy read repeated per thread is visible."""

    def __init__(self, corpus_pair: CorpusPair) -> None:
        self._corpus_pair = corpus_pair
        self.reads = 0
        self._counter = threading.Lock()

    def get_corpus_pair(self) -> CorpusPair:
        with self._counter:
            self.reads += 1
        time.sleep(_LAZY_WORK_SECONDS)
        return self._corpus_pair


class _CountingModelLoadRetriever(EmbeddingExampleRetriever):
    """Counts model loads, so one load per thread is visible."""

    def __init__(self):
        super().__init__()
        self.model_loads = 0
        self._counter = threading.Lock()

    def _create_model(self):
        with self._counter:
            self.model_loads += 1
        time.sleep(_LAZY_WORK_SECONDS)
        return _StubEmbeddingModel(_EMBEDDING_VECTORS)


def _run_together(work, workers=4):
    """Runs `work` on `workers` threads, all released at the same moment."""
    barrier = threading.Barrier(workers)

    def task(_):
        barrier.wait()
        return work()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(task, range(workers)))


def test_example_pool_fits_its_index_once_when_threads_select_together(tmp_path):
    # Inference selects examples on a thread pool, so an unguarded lazy fit runs once per worker.
    retriever = _CountingFitRetriever()
    pool = _write_pool(tmp_path, ["a b", "c d", "e f"], ["1", "2", "3"], retriever=retriever)
    pool.ensure_available()

    selected = _run_together(lambda: pool.select("a b", k=1))

    assert retriever.fit_calls == 1
    assert [len(examples) for examples in selected] == [1] * 4


def test_example_pool_reads_its_corpus_once_when_threads_select_together(tmp_path):
    src_path = tmp_path / "train.src.txt"
    trg_path = tmp_path / "train.trg.txt"
    src_path.write_text("a b\nc d\n", encoding="utf-8")
    trg_path.write_text("1\n2\n", encoding="utf-8")
    provider = _CountingCorpusPairProvider(CorpusPair(src_path, trg_path))
    pool = ExamplePool(provider, _CountingFitRetriever())

    selected = _run_together(lambda: pool.select("a b", k=1))

    assert provider.reads == 1
    assert [len(examples) for examples in selected] == [1] * 4


def test_embedding_retriever_loads_its_model_once_when_threads_rank_together(tmp_path):
    # A pool that adopted a cached index has never touched the model, so every worker's first
    # query reaches the lazy load at the same moment.
    _fitted(EmbeddingExampleRetriever(model=_StubEmbeddingModel(_EMBEDDING_VECTORS)), ["cat", "dog"]).save(tmp_path)
    retriever = _CountingModelLoadRetriever()
    assert retriever.load(tmp_path, corpus_size=2)

    ranked = _run_together(lambda: retriever.rank("kitten", k=1))

    assert retriever.model_loads == 1
    assert ranked == [[0]] * 4


def test_lexical_retrievers_cache_nothing_and_rebuild_every_run(tmp_path):
    # Refitting tfidf or bm25 takes seconds, so nothing is written that could go stale or
    # tie the experiment directory to the installed scikit-learn.
    pool = _write_pool(tmp_path, ["the cat sat", "something unrelated"], ["1", "2"])
    pool.save_index(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["train.src.txt", "train.trg.txt"]
    assert not _write_pool(tmp_path, ["the cat sat", "something unrelated"], ["1", "2"]).load_index(tmp_path)


class _RankedStubRetriever(ExampleRetriever):
    """Exercises the base class's generic leave-one-out, which BM25 relies on."""

    method = "stub"

    def _fit_index(self, sources):
        self._sources = sources

    def _top_indices_for_query(self, query, k):
        return list(range(len(self._sources)))[:k]


def test_generic_leave_one_out_drops_its_own_position_and_still_returns_k():
    retriever = _fitted(_RankedStubRetriever(), ["a", "b", "c"])
    assert retriever.rank_excluding("a", 0, k=2) == [1, 2]
    assert retriever.rank_excluding("b", 1, k=2) == [0, 2]


def test_generic_leave_one_out_returns_fewer_than_k_when_the_corpus_is_too_small():
    retriever = _fitted(_RankedStubRetriever(), ["a", "b"])
    assert retriever.rank_excluding("a", 0, k=5) == [1]


def test_embedding_retriever_saves_its_vectors_but_not_its_model(tmp_path):
    vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "kitten": [0.9, 0.1]}
    retriever = _fitted(EmbeddingExampleRetriever(model=_StubEmbeddingModel(vectors)), ["cat", "dog"])
    retriever.save(tmp_path)

    assert (tmp_path / "example_retrieval_embeddings.npy").is_file()
    # Ranking a known position uses the saved vectors, so it works without any model at all.
    loaded = EmbeddingExampleRetriever()
    assert loaded.load(tmp_path, corpus_size=2)
    assert loaded.rank_excluding("cat", 0, k=1) == [1]


def test_retriever_meta_records_the_method_and_model_name(tmp_path):
    retriever = EmbeddingExampleRetriever("some/model", model=_StubEmbeddingModel({"a": [1.0]}))
    _fitted(retriever, ["a"]).save(tmp_path)
    meta = json.loads((tmp_path / "example_retrieval_meta.json").read_text(encoding="utf-8"))
    assert meta == {"method": "embedding", "model_name": "some/model", "num_sources": 1}


def _write_pair(tmp_path, name, source) -> CorpusPair:
    pair = CorpusPair(tmp_path / f"{name}.src.txt", tmp_path / f"{name}.trg.txt")
    pair.src_path.write_text(source + "\n", encoding="utf-8")
    pair.trg_path.write_text("target\n", encoding="utf-8")
    return pair


def test_preferred_corpus_pair_is_used_when_it_exists(tmp_path):
    preferred = _write_pair(tmp_path, "preferred", "preferred")
    fallback = _write_pair(tmp_path, "fallback", "fallback")
    assert PreferredCorpusPairProvider(preferred, fallback).get_corpus_pair() == preferred


def test_preferred_corpus_pair_gives_way_to_the_fallback_when_missing(tmp_path):
    fallback = _write_pair(tmp_path, "fallback", "fallback")
    missing = CorpusPair(tmp_path / "missing.src.txt", tmp_path / "missing.trg.txt")
    assert PreferredCorpusPairProvider(missing, fallback).get_corpus_pair() == fallback


def test_a_half_written_preferred_pair_gives_way_to_the_fallback(tmp_path):
    # Taking one side from each pair would mix a detokenized source with a tokenized target.
    fallback = _write_pair(tmp_path, "fallback", "fallback")
    half = CorpusPair(tmp_path / "half.src.txt", tmp_path / "half.trg.txt")
    half.src_path.write_text("half\n", encoding="utf-8")
    assert PreferredCorpusPairProvider(half, fallback).get_corpus_pair() == fallback


def test_the_pool_draws_examples_from_the_pair_its_provider_names(tmp_path):
    preferred = _write_pair(tmp_path, "preferred", "preferred")
    fallback = _write_pair(tmp_path, "fallback", "fallback")
    pool = ExamplePool(PreferredCorpusPairProvider(preferred, fallback), ExampleRetrieverFactory.create("tfidf"))
    assert [ex.source for ex in pool.all_examples()] == ["preferred"]


def test_example_pool_names_the_corpus_it_could_not_read(tmp_path):
    pool = ExamplePool(
        FixedCorpusPairProvider(CorpusPair(tmp_path / "a.src.txt", tmp_path / "a.trg.txt")),
        ExampleRetrieverFactory.create("tfidf"),
    )
    with pytest.raises(RuntimeError, match="a.src.txt and .*a.trg.txt"):
        pool.all_examples()


def test_example_pool_ensure_available_reads_the_corpus_up_front(tmp_path):
    with pytest.raises(RuntimeError, match="Run preprocessing"):
        ExamplePool(
            FixedCorpusPairProvider(CorpusPair(tmp_path / "a.src.txt", tmp_path / "a.trg.txt")),
            ExampleRetrieverFactory.create("tfidf"),
        ).ensure_available()


def test_tokenize_for_retrieval_drops_punctuation_only_tokens():
    assert RetrievalTokenizer().tokenize("Let there be LIGHT!") == ["let", "there", "be", "light"]
    assert RetrievalTokenizer().tokenize('"Come," he said -- and went...') == ["come", "he", "said", "and", "went"]


def test_tokenize_for_retrieval_keeps_words_that_contain_punctuation():
    assert RetrievalTokenizer().tokenize("Don't stop, Jesus-like 12,345.") == ["don't", "stop", "jesus-like", "12,345"]


def test_tokenize_for_retrieval_keeps_non_latin_words():
    tokens = RetrievalTokenizer().tokenize("Se dijo: \u00abvengan\u00bb \u0663\u0664")
    assert tokens == ["se", "dijo", "vengan", "\u0663\u0664"]


def test_tokenize_for_retrieval_yields_nothing_for_punctuation_only_text():
    assert RetrievalTokenizer().tokenize("!!! ???") == []


def test_tfidf_retriever_handles_a_corpus_with_no_word_tokens():
    # Ordinarily TfidfVectorizer raises "empty vocabulary" rather than ranking nothing.
    retriever = _fitted(TfidfExampleRetriever(), ["!!!", "..."])
    assert retriever.rank("anything", k=1) == []
    assert retriever.rank_excluding("!!!", 0, k=1) == []


def test_bm25_retriever_handles_a_corpus_with_no_word_tokens():
    # Ordinarily BM25Okapi divides by zero on an average document length of zero.
    pytest.importorskip("rank_bm25")
    retriever = _fitted(BM25ExampleRetriever(), ["!!!", "..."])
    assert retriever.rank("anything", k=1) == []
    assert retriever.rank_excluding("!!!", 0, k=1) == []


def test_tfidf_retriever_indexes_a_corpus_where_only_some_sources_have_words():
    retriever = _fitted(TfidfExampleRetriever(), ["!!!", "let there be light"])
    assert retriever.rank("light", k=1) == [1]


def test_bm25_retriever_indexes_a_corpus_where_only_some_sources_have_words():
    # Three sources because BM25 idf is zero for every term of a two-document corpus
    pytest.importorskip("rank_bm25")
    retriever = _fitted(BM25ExampleRetriever(), ["!!!", "let there be light", "something else entirely"])
    assert retriever.rank("light", k=1) == [1]


def test_punctuation_carries_no_retrieval_signal():
    # If not filtered out, the query's three "!!!" would outweigh its one real word
    retriever = _fitted(TfidfExampleRetriever(), ["the cat sat", "!!! !!! !!! ???", "something else entirely"])
    assert retriever.rank("!!! !!! !!! cat", k=1) == [0]


def test_tfidf_and_bm25_agree_on_what_counts_as_a_word():
    # Switching selection method should change the ranking, not how the text is split.
    pytest.importorskip("rank_bm25")
    sources = ["do not stop", "don't stop now", "something else entirely"]
    tfidf = _fitted(TfidfExampleRetriever(), sources).rank("don't", k=3)
    bm25 = _fitted(BM25ExampleRetriever(), sources).rank("don't", k=3)
    assert tfidf[0] == bm25[0] == 1
