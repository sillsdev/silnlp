from typing import Any, Dict, Optional

from .reasoning_estimate import ReasoningEstimate


class LLMUsageStatistics:
    """Tokens and cost summed over requests, where a request to a model with no price is counted, not taken as free."""

    def __init__(self) -> None:
        self._requests = 0
        self._unpriced_requests = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        # Summed over the priced requests alone.
        self._cost = 0.0

    @classmethod
    def of_priced_request(cls, prompt_tokens: int, completion_tokens: int, cost: float) -> "LLMUsageStatistics":
        usage = cls()
        usage._requests = 1
        usage._prompt_tokens = prompt_tokens
        usage._completion_tokens = completion_tokens
        usage._cost = cost
        return usage

    @classmethod
    def of_unpriced_request(cls, prompt_tokens: int, completion_tokens: int) -> "LLMUsageStatistics":
        usage = cls()
        usage._requests = 1
        usage._unpriced_requests = 1
        usage._prompt_tokens = prompt_tokens
        usage._completion_tokens = completion_tokens
        return usage

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "LLMUsageStatistics":
        prompt_tokens = int(record["prompt_tokens"])
        completion_tokens = int(record["completion_tokens"])
        if record["cost"] is None:
            return cls.of_unpriced_request(prompt_tokens, completion_tokens)
        return cls.of_priced_request(prompt_tokens, completion_tokens, float(record["cost"]))

    def to_record(self) -> Dict[str, Any]:
        cost: Optional[float] = None if self._unpriced_requests > 0 else self._cost
        return {"prompt_tokens": self._prompt_tokens, "completion_tokens": self._completion_tokens, "cost": cost}

    def __add__(self, other: "LLMUsageStatistics") -> "LLMUsageStatistics":
        total = LLMUsageStatistics()
        total._include(self)
        total._include(other)
        return total

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)

    def including_retried_attempt(self, discarded: "LLMUsageStatistics") -> "LLMUsageStatistics":
        """Still one request, which is unpriced if either attempt was."""
        prompt_tokens = self._prompt_tokens + discarded._prompt_tokens
        completion_tokens = self._completion_tokens + discarded._completion_tokens
        if self._unpriced_requests > 0 or discarded._unpriced_requests > 0:
            return LLMUsageStatistics.of_unpriced_request(prompt_tokens, completion_tokens)
        return LLMUsageStatistics.of_priced_request(prompt_tokens, completion_tokens, self._cost + discarded._cost)

    def describe(self) -> str:
        summary = (
            f"{self._requests:,} requests, {self._prompt_tokens:,} prompt + "
            f"{self._completion_tokens:,} completion tokens"
        )
        if self._unpriced_requests == 0:
            return f"{summary}, ${self._cost:.4f}"
        if self._unpriced_requests == self._requests:
            return f"{summary}; cost unavailable (no pricing for this model)"
        return f"{summary}, ${self._cost:.4f} excluding {self._unpriced_requests:,} unpriced requests"

    def _include(self, other: "LLMUsageStatistics") -> None:
        self._requests += other._requests
        self._unpriced_requests += other._unpriced_requests
        self._prompt_tokens += other._prompt_tokens
        self._completion_tokens += other._completion_tokens
        self._cost += other._cost


class EstimatedLLMUsageStatistics(LLMUsageStatistics):
    """Usage estimated without sending anything, which also says how much is reasoning and what caching would save."""

    def __init__(
        self,
        usage: LLMUsageStatistics,
        usage_if_cached: LLMUsageStatistics,
        reasoning_tokens: int,
        reasoning: ReasoningEstimate,
    ) -> None:
        super().__init__()
        self._include(usage)
        self._usage_if_cached = usage_if_cached
        self._reasoning_tokens = reasoning_tokens
        self._reasoning = reasoning

    def __add__(self, other: LLMUsageStatistics) -> "EstimatedLLMUsageStatistics":
        estimate = (
            other
            if isinstance(other, EstimatedLLMUsageStatistics)
            else EstimatedLLMUsageStatistics(other, other, 0, self._reasoning)
        )
        return EstimatedLLMUsageStatistics(
            super().__add__(estimate),
            self._usage_if_cached + estimate._usage_if_cached,
            self._reasoning_tokens + estimate._reasoning_tokens,
            self._reasoning,
        )

    # Python tries a subclass's __radd__ first, so a sum that starts from no usage still ends as an estimate.
    __radd__ = __add__

    def describe(self) -> str:
        return super().describe() + self._reasoning.describe_total(self._reasoning_tokens) + self._describe_caching()

    def _describe_caching(self) -> str:
        cached = self._usage_if_cached
        if cached._unpriced_requests > 0 or cached._cost >= self._cost:
            return ""
        return f"; ${cached._cost:.4f} if the provider caches the system message the requests share"
