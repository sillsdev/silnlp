from typing import List, Optional

import pytest

from silnlp.nmt.llm_usage import LLMUsageStatistics
from silnlp.nmt.openrouter_models import OpenRouterListing, OpenRouterListings, OpenRouterModelList, OpenRouterRates


def _sol_pricing() -> dict:
    """GPT-6.1 Sol as OpenRouter listed it on 2026-11-03."""
    return {
        "prompt": "0.000002",
        "completion": "0.00001",
        "web_search": "0.01",
        "input_cache_read": "0.0000001",
        "input_cache_write": "0.0000025",
        "overrides": [
            {
                "min_prompt_tokens": 272000,
                "prompt": "0.000004",
                "completion": "0.000015",
                "input_cache_read": "0.0000002",
                "input_cache_write": "0.000005",
            }
        ],
    }


def _cost(usage: LLMUsageStatistics) -> Optional[float]:
    return usage.to_record()["cost"]


def test_uncached_prompt_tokens_are_priced_at_the_cache_write_rate_when_it_is_the_higher_one():
    usage = OpenRouterRates(_sol_pricing()).price(1000, 100)
    assert _cost(usage) == pytest.approx(1000 * 0.0000025 + 100 * 0.00001)


def test_a_cache_write_rate_below_the_prompt_rate_leaves_uncached_prompt_tokens_at_the_prompt_rate():
    rates = OpenRouterRates({"prompt": "0.000001", "completion": "0.000004", "input_cache_write": "0.00000004"})
    assert _cost(rates.price(1000, 100)) == pytest.approx(1000 * 0.000001 + 100 * 0.000004)


def test_without_a_cache_write_rate_uncached_prompt_tokens_are_priced_at_the_prompt_rate():
    rates = OpenRouterRates({"prompt": "0.000002", "completion": "0.00001"})
    assert _cost(rates.price(1000, 100)) == pytest.approx(1000 * 0.000002 + 100 * 0.00001)


def test_cached_prompt_tokens_are_priced_at_the_cache_read_rate():
    usage = OpenRouterRates(_sol_pricing()).price(1000, 100, 800)
    assert _cost(usage) == pytest.approx(200 * 0.0000025 + 800 * 0.0000001 + 100 * 0.00001)


def test_cached_prompt_tokens_leave_a_request_unpriced_when_no_cache_read_rate_is_listed():
    rates = OpenRouterRates({"prompt": "0.000002", "completion": "0.00001"})
    assert _cost(rates.price(1000, 100, 800)) is None


def test_a_prompt_that_reaches_the_long_prompt_threshold_is_priced_at_the_long_prompt_rates():
    usage = OpenRouterRates(_sol_pricing()).price(272000, 100)
    assert _cost(usage) == pytest.approx(272000 * 0.000005 + 100 * 0.000015)


def test_a_prompt_just_under_the_long_prompt_threshold_is_priced_at_the_base_rates():
    usage = OpenRouterRates(_sol_pricing()).price(271999, 100)
    assert _cost(usage) == pytest.approx(271999 * 0.0000025 + 100 * 0.00001)


def test_the_highest_threshold_a_prompt_reaches_sets_its_rates_whatever_order_they_are_listed_in():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.00001",
        "overrides": [
            {"min_prompt_tokens": 1000, "prompt": "0.000004"},
            {"min_prompt_tokens": 100, "prompt": "0.000003"},
        ],
    }
    assert _cost(OpenRouterRates(pricing).price(1000, 100)) == pytest.approx(1000 * 0.000004 + 100 * 0.00001)


def test_a_long_prompt_override_keeps_the_base_rates_it_does_not_list():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.00001",
        "overrides": [{"min_prompt_tokens": 500, "prompt": "0.000004"}],
    }
    assert _cost(OpenRouterRates(pricing).price(1000, 100)) == pytest.approx(1000 * 0.000004 + 100 * 0.00001)


def test_an_override_for_certain_hours_is_ignored():
    pricing = {
        "prompt": "0.000002",
        "completion": "0.00001",
        "overrides": [{"utc_start": 1500, "utc_end": 100, "prompt": "0.000001", "completion": "0.000005"}],
    }
    assert _cost(OpenRouterRates(pricing).price(1000, 100)) == pytest.approx(1000 * 0.000002 + 100 * 0.00001)


def test_a_long_prompt_override_that_also_depends_on_the_day_is_ignored():
    pricing = {
        "prompt": "0.000002",
        "completion": "0.00001",
        "overrides": [{"min_prompt_tokens": 500, "utc_days": ["saturday"], "prompt": "0.000001"}],
    }
    assert _cost(OpenRouterRates(pricing).price(1000, 100)) == pytest.approx(1000 * 0.000002 + 100 * 0.00001)


def test_listed_rates_are_fixed():
    assert OpenRouterRates(_sol_pricing()).is_fixed()


def test_a_router_listing_a_price_of_minus_one_has_no_fixed_rates():
    assert not OpenRouterRates({"prompt": "-1", "completion": "-1"}).is_fixed()


def test_a_model_listing_no_completion_price_has_no_fixed_rates():
    assert not OpenRouterRates({"prompt": "0.000002"}).is_fixed()


class ListedModels(OpenRouterModelList):
    def __init__(self, models: List[dict]) -> None:
        super().__init__()
        self._models = models

    def fetch(self) -> List[dict]:
        return self._models


class UnreachableModelList(OpenRouterModelList):
    def fetch(self) -> List[dict]:
        raise ConnectionError("openrouter.ai is unreachable")


def _listed_sol() -> List[dict]:
    return [
        {"id": "openai/gpt-6.1-terra", "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
        {"id": "openai/gpt-6.1-sol", "canonical_slug": "openai/gpt-6.1-sol-20261001", "pricing": _sol_pricing()},
    ]


def _rates_listed_for(model_id: str) -> Optional[OpenRouterRates]:
    listing = OpenRouterListings(ListedModels(_listed_sol())).find(model_id)
    assert listing is not None
    return listing.rates()


def test_a_model_is_priced_at_the_rates_listed_under_its_id():
    rates = _rates_listed_for("openai/gpt-6.1-sol")
    assert rates is not None
    assert rates.price(1000, 100) == OpenRouterRates(_sol_pricing()).price(1000, 100)


def test_a_model_is_priced_at_the_rates_listed_under_its_canonical_slug():
    rates = _rates_listed_for("openai/gpt-6.1-sol-20261001")
    assert rates is not None
    assert rates.price(1000, 100) == OpenRouterRates(_sol_pricing()).price(1000, 100)


def test_a_model_openrouter_does_not_list_is_not_found_and_a_warning_says_so(caplog):
    assert OpenRouterListings(ListedModels(_listed_sol())).find("openai/gpt-7") is None
    assert "OpenRouter's model list has no 'openai/gpt-7'" in caplog.text


def test_a_model_list_that_cannot_be_fetched_finds_nothing_and_a_warning_says_why(caplog):
    assert OpenRouterListings(UnreachableModelList()).find("openai/gpt-6.1-sol") is None
    assert "Could not fetch OpenRouter's model list" in caplog.text
    assert "openrouter.ai is unreachable" in caplog.text


def test_a_router_has_no_rates_and_a_warning_says_so(caplog):
    listing = OpenRouterListing("openrouter/auto", {"pricing": {"prompt": "-1", "completion": "-1"}})
    assert listing.rates() is None
    assert "OpenRouter lists no fixed price for 'openrouter/auto'" in caplog.text


def test_a_model_listed_as_supporting_reasoning_can_reason():
    listing = OpenRouterListing("openai/gpt-5", {"supported_parameters": ["max_tokens", "reasoning"]})
    assert listing.can_reason()


def test_a_model_listed_without_reasoning_among_its_parameters_cannot_reason():
    assert not OpenRouterListing("openai/gpt-4o", {"supported_parameters": ["max_tokens", "temperature"]}).can_reason()


def test_a_model_listed_without_any_supported_parameters_cannot_reason():
    assert not OpenRouterListing("openai/gpt-4o", {}).can_reason()
