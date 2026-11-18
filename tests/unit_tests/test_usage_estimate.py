import re
from typing import Dict, List, Optional, Tuple

import pytest

from silnlp.nmt.example_retrieval import Example, LengthRatio
from silnlp.nmt.llm_usage import RequestUsage
from silnlp.nmt.prompt_caching import PrefixPromptCachingPolicy, PromptTokenCount, SharedSystemMessageCacheStatus
from silnlp.nmt.reasoning_estimate import MeasuredReasoning, NoReasoning, ReasoningBudget, ReasoningEstimate
from silnlp.nmt.token_counter import TokenCounter
from silnlp.nmt.usage_estimate import (
    EstimateTokenCounter,
    EstimateTotals,
    RequestCostEstimator,
    PriceList,
    RequestEstimate,
    SharedSystemMessageCacheTracker,
    UnknownPrices,
    UsageEstimator,
)
from silnlp.nmt.usage_summary_table import UsageSummaryTable


class WordCounter(TokenCounter):
    """Counts a word as a token, or counts nothing at all."""

    def __init__(self, counts: bool = True) -> None:
        self._counts = counts
        self.counted: List[str] = []

    def count_tokens(self, text: str) -> Optional[int]:
        self.counted.append(text)
        return len(text.split()) if self._counts else None


class OneCentPerToken(PriceList):
    """A cent per token, and a tenth of a cent per token read from the cache."""

    def price(self, prompt_token_count: PromptTokenCount, completion_tokens: int) -> RequestUsage:
        cost = (
            prompt_token_count.total - prompt_token_count.read_from_cache + completion_tokens
        ) * 0.01 + prompt_token_count.read_from_cache * 0.001
        return RequestUsage.of_priced_request(prompt_token_count.total, completion_tokens, cost)


def _cost_estimator(reasoning: Optional[ReasoningEstimate] = None) -> RequestCostEstimator:
    return RequestCostEstimator(
        OneCentPerToken(), reasoning if reasoning is not None else NoReasoning(), PrefixPromptCachingPolicy()
    )


def _rows(table: UsageSummaryTable) -> Dict[str, Tuple[str, str]]:
    rows = {}
    for line in table.render().splitlines()[1:]:
        label, value, *note = re.split(r"\s{2,}", line.strip(), maxsplit=2)
        rows[label] = (value, note[0] if note else "")
    return rows


def _table_with(usage: RequestUsage) -> UsageSummaryTable:
    table = UsageSummaryTable("Estimate")
    usage.add_requests_row_to(table)
    usage.add_tokens_and_cost_rows_to(table)
    return table


def _not_cached() -> SharedSystemMessageCacheStatus:
    return SharedSystemMessageCacheStatus(0, already_cached=False)


# --- prices and cost estimates ----------------------------------------------------------------


def test_a_model_without_prices_bills_its_requests_as_unpriced():
    assert UnknownPrices().price(PromptTokenCount(100), 10).to_record() == {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cost": None,
    }


def test_an_estimate_bills_the_prompt_and_the_reply():
    estimate = _cost_estimator().estimate(100, 10, _not_cached())
    assert estimate.usage.describe() == "1 requests, 100 prompt + 10 completion tokens, $1.1000"


def test_an_estimate_bills_reasoning_as_part_of_the_completion():
    estimate = _cost_estimator(MeasuredReasoning("a-model", 4096, 2.0, "a test")).estimate(100, 10, _not_cached())

    assert estimate.reasoning_tokens == 20
    assert estimate.usage.describe() == "1 requests, 100 prompt + 30 completion tokens, $1.3000"


def test_an_estimate_prices_what_the_prompt_would_cost_if_the_provider_cached_its_system_message():
    estimate = _cost_estimator().estimate(2000, 10, SharedSystemMessageCacheStatus(1500, already_cached=True))

    # 500 uncached prompt tokens and 10 completion tokens at a cent, and 1,500 cached tokens at a tenth of a cent.
    assert estimate.usage_if_cached.describe_cost() == "$6.6000"


def test_an_estimate_that_caching_cannot_change_costs_the_same_if_cached():
    estimate = _cost_estimator().estimate(2000, 10, _not_cached())
    assert estimate.usage_if_cached == estimate.usage


def test_a_model_the_run_knows_nothing_of_is_billed_as_unpriced_without_reasoning():
    estimate = RequestCostEstimator.of_unknown_model().estimate(100, 10, _not_cached())

    assert estimate.reasoning_tokens == 0
    assert estimate.usage.describe().endswith("; cost unavailable (no pricing for this model)")


# --- counting -----------------------------------------------------------------------------------


def test_an_estimate_counts_tokens_as_the_model_does():
    assert EstimateTokenCounter(WordCounter()).count("three short words") == 3


def test_an_estimate_counts_a_chats_tokens_as_the_model_does():
    messages = [{"role": "system", "content": "translate this"}, {"role": "user", "content": "hello"}]
    assert EstimateTokenCounter(WordCounter()).count_messages(messages) == 3


def test_an_estimate_refuses_to_guess_tokens_the_model_cannot_count():
    with pytest.raises(RuntimeError, match="could not be counted"):
        EstimateTokenCounter(WordCounter(counts=False)).count("hello")


def test_an_estimate_refuses_to_guess_a_chats_tokens_the_model_cannot_count():
    with pytest.raises(RuntimeError, match="could not be counted"):
        EstimateTokenCounter(WordCounter(counts=False)).count_messages([{"role": "user", "content": "hello"}])


# --- a shared system message ----------------------------------------------------------------------


def _chat(system_message: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_message}, {"role": "user", "content": "hello"}]


def test_the_requests_sent_together_find_their_shared_system_message_not_yet_cached():
    cache_tracker = SharedSystemMessageCacheTracker(EstimateTokenCounter(WordCounter()), concurrency=2)

    first = cache_tracker.next_request(_chat("translate this"))
    second = cache_tracker.next_request(_chat("translate this"))

    assert (first, second) == (SharedSystemMessageCacheStatus(2, False), SharedSystemMessageCacheStatus(2, False))


def test_a_request_after_those_sent_together_finds_their_shared_system_message_cached():
    cache_tracker = SharedSystemMessageCacheTracker(EstimateTokenCounter(WordCounter()), concurrency=2)
    cache_tracker.next_request(_chat("translate this"))
    cache_tracker.next_request(_chat("translate this"))

    assert cache_tracker.next_request(_chat("translate this")) == SharedSystemMessageCacheStatus(2, already_cached=True)


def test_a_different_system_message_is_not_yet_cached():
    cache_tracker = SharedSystemMessageCacheTracker(EstimateTokenCounter(WordCounter()), concurrency=1)
    cache_tracker.next_request(_chat("translate this"))

    assert cache_tracker.next_request(_chat("back-translate this")) == SharedSystemMessageCacheStatus(
        2, already_cached=False
    )


def test_a_request_without_a_system_message_shares_none():
    cache_tracker = SharedSystemMessageCacheTracker(EstimateTokenCounter(WordCounter()), concurrency=1)
    assert cache_tracker.next_request([{"role": "user", "content": "hello"}]).num_tokens == 0


def test_a_shared_system_message_is_counted_once():
    counter = WordCounter()
    cache_tracker = SharedSystemMessageCacheTracker(EstimateTokenCounter(counter), concurrency=1)

    cache_tracker.next_request(_chat("translate this"))
    cache_tracker.next_request(_chat("translate this"))

    assert counter.counted == ["translate this"]


# --- what only an estimate says ---------------------------------------------------------------


def _priced(prompt_tokens: int, completion_tokens: int, cost: float) -> RequestUsage:
    return RequestUsage.of_priced_request(prompt_tokens, completion_tokens, cost)


def _unpriced(prompt_tokens: int, completion_tokens: int) -> RequestUsage:
    return RequestUsage.of_unpriced_request(prompt_tokens, completion_tokens)


def _noted(
    estimates: List[RequestEstimate], reasoning: Optional[ReasoningEstimate] = None
) -> Dict[str, Tuple[str, str]]:
    totals = EstimateTotals()
    usage = RequestUsage()
    for estimate in estimates:
        totals.add(estimate)
        usage = usage + estimate.usage
    table = _table_with(usage)
    totals.add_notes_to(table, _cost_estimator(reasoning))
    return _rows(table)


def test_an_estimate_notes_its_measured_reasoning_beside_its_completion_tokens():
    rows = _noted(
        [RequestEstimate(_priced(100, 30, 0.01), _priced(100, 30, 0.01), 20)],
        MeasuredReasoning("a-model", 4096, 2.0, "a test"),
    )
    assert rows["Completion tokens"] == ("30", "including about 20 reasoning tokens")


def test_an_estimate_from_a_thinking_budget_notes_that_its_cost_is_an_upper_bound():
    rows = _noted(
        [RequestEstimate(_priced(100, 110, 0.01), _priced(100, 110, 0.01), 100)], ReasoningBudget("a-model", 4096, 100)
    )
    assert rows["Completion tokens"] == ("110", "including up to 100 reasoning tokens, so the cost is an upper bound")


def test_an_estimate_without_reasoning_has_no_note_beside_its_completion_tokens():
    rows = _noted([RequestEstimate(_priced(100, 10, 0.01), _priced(100, 10, 0.01), 0)])
    assert rows["Completion tokens"] == ("10", "")


def test_an_estimate_notes_what_caching_would_save_beside_its_cost():
    rows = _noted([RequestEstimate(_priced(2000, 10, 0.03), _priced(2000, 10, 0.01), 0)])
    assert rows["Cost"] == ("$0.0300", "$0.0100 if the provider caches the system message the requests share")


def test_an_estimate_with_nothing_to_read_from_a_cache_says_nothing_of_caching():
    rows = _noted([RequestEstimate(_priced(2000, 10, 0.03), _priced(2000, 10, 0.03), 0)])
    assert rows["Cost"] == ("$0.0300", "")


def test_an_estimate_without_a_cache_price_says_nothing_of_caching():
    rows = _noted([RequestEstimate(_priced(2000, 10, 0.03), _unpriced(2000, 10), 0)])
    assert rows["Cost"] == ("$0.0300", "")


def test_an_estimate_notes_both_its_unpriced_requests_and_what_caching_would_save_beside_its_cost():
    rows = _noted(
        [
            RequestEstimate(_priced(2000, 10, 0.03), _priced(2000, 10, 0.01), 0),
            RequestEstimate(_unpriced(100, 10), _priced(100, 10, 0.0), 0),
        ]
    )
    assert rows["Cost"] == (
        "$0.0300",
        "excluding 1 unpriced requests; $0.0100 if the provider caches the system message the requests share",
    )


def test_estimates_add_up_their_reasoning_and_what_caching_would_save():
    rows = _noted(
        [
            RequestEstimate(_priced(2000, 30, 0.03), _priced(2000, 30, 0.03), 20),
            RequestEstimate(_priced(2000, 30, 0.03), _priced(2000, 30, 0.01), 20),
        ],
        MeasuredReasoning("a-model", 4096, 2.0, "a test"),
    )
    assert rows["Completion tokens"] == ("60", "including about 40 reasoning tokens")
    assert rows["Cost"] == ("$0.0600", "$0.0400 if the provider caches the system message the requests share")


def test_a_run_that_estimated_no_request_notes_nothing():
    rows = _noted([], MeasuredReasoning("a-model", 4096, 2.0, "a test"))
    assert rows["Completion tokens"] == ("0", "")


# --- estimating a request ----------------------------------------------------------------------


def _estimator(
    reasoning: Optional[ReasoningEstimate] = None, examples: Optional[List[Example]] = None
) -> UsageEstimator:
    counter = WordCounter()
    return UsageEstimator(counter, LengthRatio(examples or [], counter), _cost_estimator(reasoning), concurrency=1)


def test_a_request_is_estimated_from_its_counted_prompt_and_a_reply_as_long_as_its_source():
    usage = _estimator().estimate(_chat("translate this"), ["two words"])
    assert usage.describe() == "1 requests, 3 prompt + 2 completion tokens, $0.0500"


def test_a_reply_is_estimated_at_the_training_corpus_length_ratio():
    estimator = _estimator(examples=[Example("one two", "uno dos tres cuatro")])
    assert (
        estimator.estimate(_chat("translate this"), ["two words"])
        .describe()
        .startswith("1 requests, 3 prompt + 4 completion tokens")
    )


def test_a_batched_reply_also_numbers_its_lines():
    usage = _estimator().estimate(_chat("translate this"), ["one", "two"])
    # Two words of translation and two line numbers.
    assert usage.describe().startswith("1 requests, 3 prompt + 4 completion tokens")


def test_an_estimator_notes_what_its_estimates_add_up_to():
    estimator = _estimator(MeasuredReasoning("a-model", 4096, 2.0, "a test"))
    estimator.estimate(_chat("translate this"), ["two words"])
    estimator.estimate(_chat("translate this"), ["two words"])
    table = _table_with(_priced(6, 12, 0.0))

    estimator.add_notes_to(table)

    assert _rows(table)["Completion tokens"] == ("12", "including about 8 reasoning tokens")
