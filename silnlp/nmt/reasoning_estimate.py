import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


class ReasoningEstimate(ABC):
    """The output tokens a request spends on reasoning, which are billed like its reply but never seen."""

    def __init__(self) -> None:
        self._announced = False
        # Requests are estimated from several threads.
        self._lock = threading.Lock()

    def tokens_for(self, reply_tokens: int) -> int:
        with self._lock:
            if not self._announced:
                self._announced = True
                self._announce()
        return self._estimate_tokens(reply_tokens)

    def describe_total(self, reasoning_tokens: int) -> str:
        return ""

    def _announce(self) -> None:
        pass

    @abstractmethod
    def _estimate_tokens(self, reply_tokens: int) -> int:
        pass


class NoReasoning(ReasoningEstimate):
    def _estimate_tokens(self, reply_tokens: int) -> int:
        return 0


class UnknownReasoning(NoReasoning):
    def __init__(self, model: str) -> None:
        super().__init__()
        self._model = model

    def _announce(self) -> None:
        LOGGER.warning(
            "%s can spend output tokens on reasoning, which the estimate leaves out, so a real run may cost more.",
            self._model,
        )


class LimitedReasoning(ReasoningEstimate):
    """Reasoning and the reply share a request's max_new_tokens, so reasoning gets at most what the reply leaves."""

    def __init__(self, max_new_tokens: int) -> None:
        super().__init__()
        self._max_new_tokens = max_new_tokens
        self._warned_of_cut_off_replies = False

    def _estimate_tokens(self, reply_tokens: int) -> int:
        expected = self._expected_tokens(reply_tokens)
        room = max(0, self._max_new_tokens - reply_tokens)
        if expected > room:
            self._warn_of_cut_off_replies()
        return min(expected, room)

    @abstractmethod
    def _expected_tokens(self, reply_tokens: int) -> int:
        pass

    def _warn_of_cut_off_replies(self) -> None:
        with self._lock:
            if self._warned_of_cut_off_replies:
                return
            self._warned_of_cut_off_replies = True
        LOGGER.warning(
            "Dry run: reasoning could use up all %s of infer.max_new_tokens before some replies are complete, "
            "which leaves them cut short or blank in a real run. Raise max_new_tokens or lower infer_batch_size.",
            f"{self._max_new_tokens:,}",
        )


class ReasoningBudget(LimitedReasoning):
    def __init__(self, model: str, max_new_tokens: int, budget: int) -> None:
        super().__init__(max_new_tokens)
        self._model = model
        self._budget = budget

    def _announce(self) -> None:
        LOGGER.info(
            "Dry run: each request to %s is estimated to spend its whole thinking budget of %s tokens, which is an "
            "upper bound.",
            self._model,
            f"{self._budget:,}",
        )

    def describe_total(self, reasoning_tokens: int) -> str:
        return f"up to {reasoning_tokens:,} reasoning tokens, so the cost is an upper bound"

    def _expected_tokens(self, reply_tokens: int) -> int:
        return self._budget


class MeasuredReasoning(LimitedReasoning):
    def __init__(self, model: str, max_new_tokens: int, tokens_per_reply_token: float, measured_for: str) -> None:
        super().__init__(max_new_tokens)
        self._model = model
        self._tokens_per_reply_token = tokens_per_reply_token
        self._measured_for = measured_for

    def _announce(self) -> None:
        LOGGER.info(
            "Dry run: %s is estimated to reason for %s tokens per reply token, as measured for %s.",
            self._model,
            f"{self._tokens_per_reply_token:g}",
            self._measured_for,
        )

    def describe_total(self, reasoning_tokens: int) -> str:
        return f"about {reasoning_tokens:,} reasoning tokens"

    def _expected_tokens(self, reply_tokens: int) -> int:
        return round(reply_tokens * self._tokens_per_reply_token)


class GptReasoningRates:
    """Rates measured on GPT-6.1 Sol, which stand in for every GPT model since a rough estimate beats none."""

    # Keyed by reasoning effort, where None is the provider's default.
    _TOKENS_PER_REPLY_TOKEN: Dict[Optional[str], float] = {
        None: 0.54,
        "none": 0.0,
        "minimal": 0.03,
        "low": 0.06,
        "medium": 0.28,
        "high": 5.7,
        "xhigh": 12.0,
    }

    def estimate_for(
        self, model: str, litellm_options: Dict[str, Any], max_new_tokens: int
    ) -> Optional[ReasoningEstimate]:
        if not model.split("/")[-1].lower().startswith("gpt-"):
            return None
        effort = self._effort(litellm_options)
        if effort not in self._TOKENS_PER_REPLY_TOKEN:
            return None
        setting = "the default reasoning effort" if effort is None else f"reasoning effort '{effort}'"
        return MeasuredReasoning(
            model, max_new_tokens, self._TOKENS_PER_REPLY_TOKEN[effort], f"GPT-6.1 Sol at {setting}"
        )

    def _effort(self, litellm_options: Dict[str, Any]) -> Optional[str]:
        reasoning = litellm_options.get("reasoning")
        if "reasoning_effort" not in litellm_options and isinstance(reasoning, dict):
            # OpenRouter's own form of the setting.
            return reasoning.get("effort")
        return litellm_options.get("reasoning_effort")
