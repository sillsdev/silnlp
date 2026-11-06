import logging
from typing import Dict, List, Optional, Set, Tuple

import requests

from .llm_usage import LLMUsageStatistics

LOGGER = logging.getLogger(__name__)


class OpenRouterModelList:
    """OpenRouter publishes the models it serves, with their prices, for anyone to read without an API key."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self._timeout_seconds = timeout_seconds

    def fetch(self) -> List[dict]:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()["data"]


class OpenRouterRates:
    """One model's listed prices per token, including the higher rates that some models charge for long prompts."""

    _PRICES = ("prompt", "completion", "input_cache_read", "input_cache_write")

    def __init__(self, pricing: dict) -> None:
        self._base = self._parse(pricing)
        # Only overrides for long prompts apply, since others, such as off-peak rates, depend on when a run happens.
        self._long_prompt_tiers: List[Tuple[int, Dict[str, float]]] = sorted(
            (
                (int(override["min_prompt_tokens"]), {**self._base, **self._parse(override)})
                for override in pricing.get("overrides", [])
                if self._conditions_of(override) == {"min_prompt_tokens"}
            ),
            key=lambda tier: tier[0],
        )

    def is_fixed(self) -> bool:
        # Routers list a price of -1, since what they charge depends on the model each request is routed to.
        return all(self._base.get(name, -1.0) >= 0 for name in ("prompt", "completion"))

    def price(self, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0) -> LLMUsageStatistics:
        rates = self._rates_for(prompt_tokens)
        if cached_prompt_tokens > 0 and "input_cache_read" not in rates:
            return LLMUsageStatistics.of_unpriced_request(prompt_tokens, completion_tokens)
        cost = (
            (prompt_tokens - cached_prompt_tokens) * self._uncached_prompt_rate(rates)
            + cached_prompt_tokens * rates.get("input_cache_read", 0.0)
            + completion_tokens * rates["completion"]
        )
        return LLMUsageStatistics.of_priced_request(prompt_tokens, completion_tokens, cost)

    def _parse(self, pricing: dict) -> Dict[str, float]:
        return {name: float(pricing[name]) for name in self._PRICES if name in pricing}

    def _conditions_of(self, override: dict) -> Set[str]:
        # OpenRouter lists every price as a string, so a value of any other type is a condition.
        return {name for name, value in override.items() if not isinstance(value, str)}

    def _rates_for(self, prompt_tokens: int) -> Dict[str, float]:
        rates = self._base
        for min_prompt_tokens, tier_rates in self._long_prompt_tiers:
            if prompt_tokens >= min_prompt_tokens:
                rates = tier_rates
        return rates

    def _uncached_prompt_rate(self, rates: Dict[str, float]) -> float:
        # Automatic caching bills an uncached prompt at the cache-write price where that exceeds the prompt price.
        return max(rates["prompt"], rates.get("input_cache_write", 0.0))


class OpenRouterListing:
    def __init__(self, model_id: str, entry: dict) -> None:
        self._model_id = model_id
        self._entry = entry

    def rates(self) -> Optional[OpenRouterRates]:
        """None, after a warning, when OpenRouter lists no fixed price for the model."""
        rates = OpenRouterRates(self._entry.get("pricing", {}))
        if not rates.is_fixed():
            LOGGER.warning("OpenRouter lists no fixed price for '%s', so it can't be priced.", self._model_id)
            return None
        return rates

    def can_reason(self) -> bool:
        return "reasoning" in self._entry.get("supported_parameters", [])


class OpenRouterListings:
    def __init__(self, model_list: OpenRouterModelList) -> None:
        self._model_list = model_list

    def find(self, model_id: str) -> Optional[OpenRouterListing]:
        """None, after a warning that says why, when OpenRouter's model list can't be fetched or lacks the model."""
        try:
            models = self._model_list.fetch()
        except Exception as error:
            LOGGER.warning("Could not fetch OpenRouter's model list to look up '%s': %s", model_id, error)
            return None
        listed = next((model for model in models if model_id in (model.get("id"), model.get("canonical_slug"))), None)
        if listed is None:
            LOGGER.warning("OpenRouter's model list has no '%s', so it can't be priced.", model_id)
            return None
        return OpenRouterListing(model_id, listed)
