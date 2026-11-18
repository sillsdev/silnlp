import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .example_retrieval import LengthRatio
from .llm_usage import RequestUsage
from .prompt_caching import (
    PrefixPromptCachingPolicy,
    PromptCachingPolicy,
    PromptTokenCount,
    SharedSystemMessageCacheStatus,
)
from .reasoning_estimate import NoReasoning, ReasoningEstimate
from .token_counter import TokenCounter
from .usage_summary_table import UsageSummaryTable

LOGGER = logging.getLogger(__name__)


class PriceList(ABC):
    @abstractmethod
    def price(self, prompt_token_count: PromptTokenCount, completion_tokens: int) -> RequestUsage:
        pass


class UnknownPrices(PriceList):
    def price(self, prompt_token_count: PromptTokenCount, completion_tokens: int) -> RequestUsage:
        return RequestUsage.of_unpriced_request(prompt_token_count.total, completion_tokens)


class LiteLLMPriceList(PriceList):
    def __init__(self, model: str, litellm: Any) -> None:
        self._model = model
        self._litellm = litellm

    def price(self, prompt_token_count: PromptTokenCount, completion_tokens: int) -> RequestUsage:
        try:
            model_info = self._litellm.get_model_info(self._model)
            # LiteLLM bills cached tokens at nothing, rather than at full price, when it has no cache price for a model.
            if prompt_token_count.read_from_cache > 0 and model_info.get("cache_read_input_token_cost") is None:
                return RequestUsage.of_unpriced_request(prompt_token_count.total, completion_tokens)
            prompt_cost, completion_cost = self._litellm.cost_per_token(
                model=self._model,
                prompt_tokens=prompt_token_count.total,
                completion_tokens=completion_tokens,
                cache_read_input_tokens=prompt_token_count.read_from_cache,
                cache_creation_input_tokens=self._tokens_written_at_a_premium(prompt_token_count, model_info),
            )
        except Exception:
            LOGGER.debug("LiteLLM has no price for '%s'.", self._model, exc_info=True)
            return RequestUsage.of_unpriced_request(prompt_token_count.total, completion_tokens)
        return RequestUsage.of_priced_request(
            prompt_token_count.total, completion_tokens, float(prompt_cost) + float(completion_cost)
        )

    def _tokens_written_at_a_premium(self, prompt_token_count: PromptTokenCount, model_info: dict) -> int:
        write_price = model_info.get("cache_creation_input_token_cost")
        # Left as ordinary input, since LiteLLM bills written tokens at nothing without a write price.
        if write_price is None or write_price <= model_info.get("input_cost_per_token", 0.0):
            return 0
        return prompt_token_count.written_to_cache


@dataclass(frozen=True)
class RequestEstimate:
    usage: RequestUsage
    usage_if_cached: RequestUsage
    reasoning_tokens: int


class RequestCostEstimator:
    """How a model would bill a request: the reasoning it adds to the reply, and its cache and prices for the prompt."""

    def __init__(self, prices: PriceList, reasoning: ReasoningEstimate, caching_policy: PromptCachingPolicy) -> None:
        self._prices = prices
        self._reasoning = reasoning
        self._caching_policy = caching_policy

    @classmethod
    def of_unknown_model(cls) -> "RequestCostEstimator":
        return cls(UnknownPrices(), NoReasoning(), PrefixPromptCachingPolicy())

    def estimate(
        self, prompt_tokens: int, reply_tokens: int, cache_status: SharedSystemMessageCacheStatus
    ) -> RequestEstimate:
        reasoning_tokens = self._reasoning.tokens_for(reply_tokens)
        completion_tokens = reply_tokens + reasoning_tokens
        prompt_token_count = self._caching_policy.split_prompt(prompt_tokens, cache_status)
        prompt_token_count_if_cached = self._caching_policy.split_prompt_if_cached(prompt_tokens, cache_status)
        usage = self._prices.price(prompt_token_count, completion_tokens)
        usage_if_cached = (
            usage
            if prompt_token_count_if_cached == prompt_token_count
            else self._prices.price(prompt_token_count_if_cached, completion_tokens)
        )
        return RequestEstimate(usage, usage_if_cached, reasoning_tokens)

    def describe_reasoning(self, reasoning_tokens: int) -> str:
        return self._reasoning.describe_total(reasoning_tokens)


class EstimateTokenCounter:
    """Counts tokens as the model would, and fails rather than let an estimate rest on a guess."""

    def __init__(self, token_counter: TokenCounter) -> None:
        self._token_counter = token_counter

    def count(self, text: str) -> int:
        return self._required(self._token_counter.count_tokens(text))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        return self._required(self._token_counter.count_message_tokens(messages))

    def _required(self, tokens: Optional[int]) -> int:
        if tokens is None:
            raise RuntimeError("A dry run has to count the model's tokens, and they could not be counted.")
        return tokens


class SharedSystemMessageCacheTracker:
    """Counts the requests that share each system message, to say whether a request finds it already cached."""

    def __init__(self, tokens: EstimateTokenCounter, concurrency: int) -> None:
        self._tokens = tokens
        self._concurrency = concurrency
        self._token_counts: Dict[str, int] = {}
        self._request_counts: Dict[str, int] = {}
        # Requests are estimated from several threads.
        self._lock = threading.Lock()

    def next_request(self, messages: List[Dict[str, str]]) -> SharedSystemMessageCacheStatus:
        system_message = messages[0]["content"] if messages[0]["role"] == "system" else ""
        with self._lock:
            if system_message not in self._token_counts:
                self._token_counts[system_message] = self._tokens.count(system_message) if system_message else 0
            self._request_counts[system_message] = self._request_counts.get(system_message, 0) + 1
            # The first requests go out together, and none can read what the others are still writing to the cache.
            already_cached = self._request_counts[system_message] > self._concurrency
            return SharedSystemMessageCacheStatus(self._token_counts[system_message], already_cached)


class EstimateTotals:
    """Totals a dry run's estimates so the usage summary can note how much is reasoning and what caching would save."""

    def __init__(self) -> None:
        self._usage = RequestUsage()
        self._usage_if_cached = RequestUsage()
        self._reasoning_tokens = 0
        # Requests are estimated from several threads.
        self._lock = threading.Lock()

    def add(self, estimate: RequestEstimate) -> None:
        with self._lock:
            self._usage = self._usage + estimate.usage
            self._usage_if_cached = self._usage_if_cached + estimate.usage_if_cached
            self._reasoning_tokens += estimate.reasoning_tokens

    def add_notes_to(self, table: UsageSummaryTable, cost_estimator: RequestCostEstimator) -> None:
        with self._lock:
            if self._usage.is_empty():
                return
            reasoning = cost_estimator.describe_reasoning(self._reasoning_tokens)
            if reasoning:
                table.add_note("Completion tokens", f"including {reasoning}")
            if self._usage_if_cached.costs_less_than(self._usage):
                table.add_note(
                    "Cost",
                    f"{self._usage_if_cached.describe_cost()} if the provider caches the system message the requests "
                    "share",
                )


class UsageEstimator:
    """Estimates what each request of a dry run would use, from the tokens of its prompt and of its likely reply."""

    def __init__(
        self,
        token_counter: TokenCounter,
        length_ratio: LengthRatio,
        cost_estimator: RequestCostEstimator,
        concurrency: int,
    ) -> None:
        self._tokens = EstimateTokenCounter(token_counter)
        self._length_ratio = length_ratio
        self._cost_estimator = cost_estimator
        self._cache_tracker = SharedSystemMessageCacheTracker(self._tokens, concurrency)
        self._totals = EstimateTotals()

    def estimate(self, messages: List[Dict[str, str]], texts: Sequence[str]) -> RequestUsage:
        estimate = self._cost_estimator.estimate(
            self._tokens.count_messages(messages),
            self._reply_tokens(texts),
            self._cache_tracker.next_request(messages),
        )
        self._totals.add(estimate)
        return estimate.usage

    def add_notes_to(self, table: UsageSummaryTable) -> None:
        self._totals.add_notes_to(table, self._cost_estimator)

    def _reply_tokens(self, texts: Sequence[str]) -> int:
        tokens = self._length_ratio.scale(self._tokens.count("\n".join(texts)))
        if len(texts) > 1:
            # A batched reply also numbers its lines.
            tokens += self._tokens.count("\n".join(f"{number}." for number in range(1, len(texts) + 1)))
        return tokens
