import json

from silnlp.nmt.llm_usage import EstimatedLLMUsageStatistics, LLMUsageStatistics
from silnlp.nmt.reasoning_estimate import MeasuredReasoning, NoReasoning, ReasoningBudget


def _priced(prompt_tokens: int, completion_tokens: int, cost: float) -> LLMUsageStatistics:
    return LLMUsageStatistics.of_priced_request(prompt_tokens, completion_tokens, cost)


def _unpriced(prompt_tokens: int, completion_tokens: int) -> LLMUsageStatistics:
    return LLMUsageStatistics.of_unpriced_request(prompt_tokens, completion_tokens)


def _measured_reasoning() -> MeasuredReasoning:
    return MeasuredReasoning("a-model", 4096, 2.0, "a test")


def _estimated(usage: LLMUsageStatistics, usage_if_cached=None, reasoning_tokens=0, reasoning=None):
    return EstimatedLLMUsageStatistics(
        usage,
        usage if usage_if_cached is None else usage_if_cached,
        reasoning_tokens,
        NoReasoning() if reasoning is None else reasoning,
    )


def test_a_priced_request_is_described_with_its_cost():
    assert _priced(100, 10, 0.01).describe() == "1 requests, 100 prompt + 10 completion tokens, $0.0100"


def test_a_free_request_costs_nothing_rather_than_an_unknown_amount():
    assert _priced(100, 10, 0.0).describe().endswith(", $0.0000")


def test_an_unpriced_request_has_no_cost_rather_than_a_zero_one():
    described = _unpriced(100, 10).describe()

    assert described.endswith("; cost unavailable (no pricing for this model)")
    assert "$" not in described


def test_no_usage_is_no_requests_at_no_cost():
    assert LLMUsageStatistics().describe() == "0 requests, 0 prompt + 0 completion tokens, $0.0000"


def test_priced_requests_add_up():
    total = _priced(100, 10, 0.01) + _priced(200, 20, 0.02)

    assert total.describe() == "2 requests, 300 prompt + 30 completion tokens, $0.0300"


def test_a_sum_with_an_unpriced_request_gives_the_cost_of_the_priced_ones():
    total = _priced(100, 10, 0.01) + _unpriced(200, 20)

    assert total.describe() == "2 requests, 300 prompt + 30 completion tokens, $0.0100 excluding 1 unpriced requests"


def test_a_sum_of_unpriced_requests_has_no_cost():
    assert "cost unavailable" in (_unpriced(100, 10) + _unpriced(200, 20)).describe()


def test_a_retried_request_is_one_request_billed_for_both_attempts():
    usage = _priced(100, 2, 0.01).including_retried_attempt(_priced(100, 5, 0.02))

    assert usage.describe() == "1 requests, 200 prompt + 7 completion tokens, $0.0300"


def test_a_retried_request_with_an_unpriced_attempt_has_no_cost():
    usage = _priced(100, 2, 0.01).including_retried_attempt(_unpriced(100, 5))

    assert usage.describe() == (
        "1 requests, 200 prompt + 7 completion tokens; cost unavailable (no pricing for this model)"
    )


def test_a_priced_request_comes_back_from_its_record():
    usage = _priced(100, 10, 0.003)

    assert LLMUsageStatistics.from_record(json.loads(json.dumps(usage.to_record()))) == usage


def test_an_unpriced_request_comes_back_from_its_record():
    usage = _unpriced(100, 10)

    assert LLMUsageStatistics.from_record(json.loads(json.dumps(usage.to_record()))) == usage


def test_the_record_of_an_unpriced_request_has_a_null_cost():
    assert _unpriced(100, 10).to_record() == {"prompt_tokens": 100, "completion_tokens": 10, "cost": None}


def test_requests_with_different_costs_are_not_equal():
    assert _priced(100, 10, 0.01) != _priced(100, 10, 0.02)


def test_a_priced_and_an_unpriced_request_are_not_equal():
    assert _priced(100, 10, 0.0) != _unpriced(100, 10)


def test_an_estimate_is_described_like_real_usage_when_it_has_no_reasoning_or_caching():
    assert _estimated(_priced(100, 10, 0.01)).describe() == _priced(100, 10, 0.01).describe()


def test_an_estimate_says_how_much_of_it_is_measured_reasoning():
    estimate = _estimated(_priced(100, 30, 0.01), reasoning_tokens=20, reasoning=_measured_reasoning())

    assert estimate.describe().endswith("; this includes about 20 reasoning tokens")


def test_an_estimate_from_a_thinking_budget_is_an_upper_bound():
    budget = ReasoningBudget("a-model", 4096, 100)
    estimate = _estimated(_priced(100, 110, 0.01), reasoning_tokens=100, reasoning=budget)

    assert estimate.describe().endswith("; this includes up to 100 reasoning tokens, so the cost is an upper bound")


def test_an_estimate_says_what_caching_would_save():
    estimate = _estimated(_priced(2000, 10, 0.03), usage_if_cached=_priced(2000, 10, 0.01))

    assert estimate.describe().endswith("; $0.0100 if the provider caches the system message the requests share")


def test_an_estimate_with_nothing_to_read_from_a_cache_says_nothing_of_caching():
    assert "caches" not in _estimated(_priced(2000, 10, 0.03)).describe()


def test_an_estimate_without_a_cache_price_says_nothing_of_caching():
    estimate = _estimated(_priced(2000, 10, 0.03), usage_if_cached=_unpriced(2000, 10))

    assert "caches" not in estimate.describe()


def test_estimates_add_up_with_their_reasoning_and_caching():
    reasoning = _measured_reasoning()
    first = _estimated(_priced(2000, 30, 0.03), _priced(2000, 30, 0.03), 20, reasoning)
    second = _estimated(_priced(2000, 30, 0.03), _priced(2000, 30, 0.01), 20, reasoning)

    assert (first + second).describe() == (
        "2 requests, 4,000 prompt + 60 completion tokens, $0.0600; this includes about 40 reasoning tokens; "
        "$0.0400 if the provider caches the system message the requests share"
    )


def test_a_sum_that_starts_from_no_usage_is_still_an_estimate():
    estimate = _estimated(_priced(100, 30, 0.01), reasoning_tokens=20, reasoning=_measured_reasoning())

    assert (LLMUsageStatistics() + estimate).describe() == estimate.describe()


def test_real_usage_added_to_an_estimate_counts_without_reasoning():
    estimate = _estimated(_priced(100, 30, 0.01), reasoning_tokens=20, reasoning=_measured_reasoning())

    assert (estimate + _priced(100, 10, 0.01)).describe() == (
        "2 requests, 200 prompt + 40 completion tokens, $0.0200; this includes about 20 reasoning tokens"
    )
