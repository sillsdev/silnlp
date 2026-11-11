import threading
from typing import Any, Dict, Optional

from .usage_summary_table import UsageSummaryTable


class RequestUsage:
    """Tokens and cost summed over requests, where a request to a model with no price is counted, not taken as free."""

    def __init__(self) -> None:
        self._requests = 0
        self._unpriced_requests = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        # Summed over the priced requests alone.
        self._cost = 0.0

    @classmethod
    def of_priced_request(cls, prompt_tokens: int, completion_tokens: int, cost: float) -> "RequestUsage":
        usage = cls()
        usage._requests = 1
        usage._prompt_tokens = prompt_tokens
        usage._completion_tokens = completion_tokens
        usage._cost = cost
        return usage

    @classmethod
    def of_unpriced_request(cls, prompt_tokens: int, completion_tokens: int) -> "RequestUsage":
        usage = cls()
        usage._requests = 1
        usage._unpriced_requests = 1
        usage._prompt_tokens = prompt_tokens
        usage._completion_tokens = completion_tokens
        return usage

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "RequestUsage":
        prompt_tokens = int(record["prompt_tokens"])
        completion_tokens = int(record["completion_tokens"])
        if record["cost"] is None:
            return cls.of_unpriced_request(prompt_tokens, completion_tokens)
        return cls.of_priced_request(prompt_tokens, completion_tokens, float(record["cost"]))

    def to_record(self) -> Dict[str, Any]:
        cost: Optional[float] = None if self._unpriced_requests > 0 else self._cost
        return {"prompt_tokens": self._prompt_tokens, "completion_tokens": self._completion_tokens, "cost": cost}

    def __add__(self, other: "RequestUsage") -> "RequestUsage":
        total = RequestUsage()
        for usage in (self, other):
            total._requests += usage._requests
            total._unpriced_requests += usage._unpriced_requests
            total._prompt_tokens += usage._prompt_tokens
            total._completion_tokens += usage._completion_tokens
            total._cost += usage._cost
        return total

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)

    def is_empty(self) -> bool:
        return self._requests == 0

    def costs_less_than(self, other: "RequestUsage") -> bool:
        return self._unpriced_requests == 0 and self._cost < other._cost

    def including_retried_attempt(self, discarded: "RequestUsage") -> "RequestUsage":
        """Still one request, which is unpriced if either attempt was."""
        prompt_tokens = self._prompt_tokens + discarded._prompt_tokens
        completion_tokens = self._completion_tokens + discarded._completion_tokens
        if self._unpriced_requests > 0 or discarded._unpriced_requests > 0:
            return RequestUsage.of_unpriced_request(prompt_tokens, completion_tokens)
        return RequestUsage.of_priced_request(prompt_tokens, completion_tokens, self._cost + discarded._cost)

    def describe(self) -> str:
        summary = (
            f"{self._requests:,} requests, {self._prompt_tokens:,} prompt + "
            f"{self._completion_tokens:,} completion tokens"
        )
        if self._unpriced_requests == 0:
            return f"{summary}, {self.describe_cost()}"
        if self._unpriced_requests == self._requests:
            return f"{summary}; cost unavailable (no pricing for this model)"
        return f"{summary}, {self.describe_cost()} excluding {self._unpriced_requests:,} unpriced requests"

    def describe_cost(self) -> str:
        return f"${self._cost:.4f}"

    def add_requests_row_to(self, table: UsageSummaryTable) -> None:
        table.add_row("Requests", f"{self._requests:,}")

    def add_tokens_and_cost_rows_to(self, table: UsageSummaryTable) -> None:
        table.add_row("Prompt tokens", f"{self._prompt_tokens:,}")
        table.add_row("Completion tokens", f"{self._completion_tokens:,}")
        if 0 < self._unpriced_requests == self._requests:
            table.add_row("Cost", "unavailable", "no pricing for this model")
        elif self._unpriced_requests > 0:
            table.add_row("Cost", self.describe_cost(), f"excluding {self._unpriced_requests:,} unpriced requests")
        else:
            table.add_row("Cost", self.describe_cost())


class RunUsage:
    """The requests a translation run sent, and the saved replies it reused and translations it copied instead."""

    def __init__(self) -> None:
        self._request_usage = RequestUsage()
        self._reused_replies = 0
        self._copied_translations = 0
        # Requests are made from several threads.
        self._lock = threading.Lock()

    def add_request(self, request_usage: RequestUsage) -> None:
        with self._lock:
            self._request_usage = self._request_usage + request_usage

    def add_reused_reply(self) -> None:
        with self._lock:
            self._reused_replies += 1

    def add_copied_translation(self) -> None:
        with self._lock:
            self._copied_translations += 1

    def include(self, other: "RunUsage") -> None:
        with self._lock:
            self._request_usage = self._request_usage + other._request_usage
            self._reused_replies += other._reused_replies
            self._copied_translations += other._copied_translations

    def describe(self) -> str:
        with self._lock:
            summary = self._request_usage.describe()
            if self._reused_replies > 0:
                summary = f"{summary}; replies reused from an earlier run: {self._reused_replies:,}"
            if self._copied_translations > 0:
                summary = f"{summary}; translations copied from the training corpus: {self._copied_translations:,}"
            return summary

    def add_rows_to(self, table: UsageSummaryTable) -> None:
        with self._lock:
            self._request_usage.add_requests_row_to(table)
            table.add_row(
                "Saved replies reused",
                f"{self._reused_replies:,}",
                (
                    "no requests; disable with --force-infer or infer.reuse_saved_replies: false"
                    if self._reused_replies
                    else ""
                ),
            )
            table.add_row(
                "Translations copied",
                f"{self._copied_translations:,}",
                (
                    "from training data, no requests; disable with infer.copy_exact_matches: false"
                    if self._copied_translations
                    else ""
                ),
            )
            self._request_usage.add_tokens_and_cost_rows_to(table)
