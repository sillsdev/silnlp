import logging
import math
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


class ReasoningToTheLimit(LimitedReasoning):
    def __init__(self, model: str, max_new_tokens: int, measured_for: str) -> None:
        super().__init__(max_new_tokens)
        self._model = model
        self._measured_for = measured_for

    def _announce(self) -> None:
        LOGGER.warning(
            "Dry run: %s is estimated to reason until infer.max_new_tokens runs out, which leaves replies cut short or "
            "blank in a real run; a lower reasoning effort avoids this. Measured with max_new_tokens at 4,096 for %s.",
            self._model,
            self._measured_for,
        )

    def describe_total(self, reasoning_tokens: int) -> str:
        return f"up to {reasoning_tokens:,} reasoning tokens, so the cost is an upper bound"

    def _expected_tokens(self, reply_tokens: int) -> int:
        # No more than the reply leaves, since the announcement has already warned of cut-off replies.
        return max(0, self._max_new_tokens - reply_tokens)


class ReasoningRates(ABC):
    """How much a provider's models reason, so that a dry run can bill it like the reply it never shows."""

    def __init__(self, model: str) -> None:
        self._model = model

    @classmethod
    def for_model(cls, model: str) -> "ReasoningRates":
        name = model.lower()
        if name.split("/")[-1].startswith("gpt-"):
            return OpenAIReasoningRates(model)
        if "claude" in name:
            return AnthropicReasoningRates(model)
        if "gemini" in name:
            return GoogleReasoningRates(model)
        return UnmeasuredReasoningRates(model)

    def estimate_for(self, litellm_options: Dict[str, Any], max_new_tokens: int) -> ReasoningEstimate:
        budget = self._budget(litellm_options)
        if budget is not None:
            return self._estimate_for_budget(budget, max_new_tokens)
        return self._estimate_for_effort(self._effort(litellm_options), max_new_tokens)

    def _estimate_for_budget(self, budget: int, max_new_tokens: int) -> ReasoningEstimate:
        return ReasoningBudget(self._model, max_new_tokens, budget)

    @abstractmethod
    def _estimate_for_effort(self, effort: Optional[str], max_new_tokens: int) -> ReasoningEstimate:
        pass

    def _effort(self, litellm_options: Dict[str, Any]) -> Optional[str]:
        if "reasoning_effort" in litellm_options:
            return litellm_options["reasoning_effort"]
        return self._openrouter_settings(litellm_options).get("effort")

    def _budget(self, litellm_options: Dict[str, Any]) -> Optional[int]:
        if "reasoning_effort" in litellm_options:
            return None
        budget = self._openrouter_settings(litellm_options).get("max_tokens")
        return budget if isinstance(budget, int) else None

    def _openrouter_settings(self, litellm_options: Dict[str, Any]) -> Dict[str, Any]:
        reasoning = litellm_options.get("reasoning")
        return reasoning if isinstance(reasoning, dict) else {}


class UnmeasuredReasoningRates(ReasoningRates):
    def _estimate_for_effort(self, effort: Optional[str], max_new_tokens: int) -> ReasoningEstimate:
        return UnknownReasoning(self._model)


class MeasuredReasoningRates(ReasoningRates):
    """Experimentally measured reasoning rates, which a dry run uses to estimate.."""

    # Keyed by reasoning effort, where None is the provider's default; math.inf marks an effort that ran out of tokens.
    _TOKENS_PER_REPLY_TOKEN: Dict[Optional[str], float]

    def _estimate_for_effort(self, effort: Optional[str], max_new_tokens: int) -> ReasoningEstimate:
        if effort not in self._TOKENS_PER_REPLY_TOKEN:
            return UnknownReasoning(self._model)
        rate = self._TOKENS_PER_REPLY_TOKEN[effort]
        setting = "the default reasoning effort" if effort is None else f"reasoning effort '{effort}'"
        measured_for = self._measured_for(setting)
        if math.isinf(rate):
            return ReasoningToTheLimit(self._model, max_new_tokens, measured_for)
        return MeasuredReasoning(self._model, max_new_tokens, rate, measured_for)

    @abstractmethod
    def _measured_for(self, setting: str) -> str:
        pass


class OpenAIReasoningRates(MeasuredReasoningRates):

    _TOKENS_PER_REPLY_TOKEN = {
        None: 0.54,
        "none": 0.0,
        "minimal": 0.03,
        "low": 0.06,
        "medium": 0.28,
        "high": 5.7,
        "xhigh": 12.0,
    }

    def _estimate_for_budget(self, budget: int, max_new_tokens: int) -> ReasoningEstimate:
        # OpenRouter turns a budget into an effort for these models, and every budget tried reasoned like 'medium'.
        return self._estimate_for_effort("medium", max_new_tokens)

    def _measured_for(self, setting: str) -> str:
        return f"GPT-6.1 Sol at {setting}"


class AnthropicReasoningRates(MeasuredReasoningRates):

    _TOKENS_PER_REPLY_TOKEN = {
        None: 2.7,
        "none": 0.0,
        "minimal": 0.0,
        "low": 0.0,
        "medium": 3.8,
        "high": 16.0,
        "xhigh": 65.0,
    }

    def _measured_for(self, setting: str) -> str:
        return f"Claude Haiku 5.5 and Sonnet 5.5 at {setting}, whichever reasoned more"


class GoogleReasoningRates(MeasuredReasoningRates):

    # LiteLLM asks Gemini for its low thinking level at none and minimal, and for its high one at medium on Flash.
    _TOKENS_PER_REPLY_TOKEN = {
        None: math.inf,
        "none": 4.2,
        "minimal": 4.2,
        "low": 4.2,
        "medium": math.inf,
        "high": math.inf,
    }

    def _measured_for(self, setting: str) -> str:
        return f"Gemini 3.7 Flash, 3.5 Flash Lite and 3.1 Pro at {setting}, whichever reasoned most"


class LiteLLMReasoningEstimator:
    """Picks how a dry run estimates a model's reasoning, from what LiteLLM makes of the run's settings."""

    def __init__(self, model: str, litellm_options: Dict[str, Any], max_new_tokens: int, litellm: Any) -> None:
        self._model = model
        self._litellm_options = litellm_options
        self._max_new_tokens = max_new_tokens
        self._litellm = litellm

    def estimate(self, listed_as_reasoning: bool = False) -> ReasoningEstimate:
        budget = self._budget()
        if budget is not None:
            return ReasoningBudget(self._model, self._max_new_tokens, budget)
        if not (listed_as_reasoning or self._can_reason()):
            return NoReasoning()
        return ReasoningRates.for_model(self._model).estimate_for(self._litellm_options, self._max_new_tokens)

    def _can_reason(self) -> bool:
        if any(option in self._litellm_options for option in ("reasoning_effort", "reasoning", "thinking")):
            return True
        try:
            return bool(self._litellm.supports_reasoning(model=self._model))
        except Exception:
            return False

    def _budget(self) -> Optional[int]:
        settings = {
            name: value for name, value in self._litellm_options.items() if name in ("reasoning_effort", "thinking")
        }
        if not settings:
            return None
        try:
            # LiteLLM's own translation of the settings, so that its table of budgets stays the only one.
            model, provider, _, _ = self._litellm.get_llm_provider(self._model)
            params = self._litellm.get_optional_params(
                model=model, custom_llm_provider=provider, max_tokens=self._max_new_tokens, **settings
            )
        except Exception:
            LOGGER.debug("LiteLLM could not translate the reasoning settings for '%s'.", self._model, exc_info=True)
            return None
        thinking = params.get("thinking") or params.get("thinkingConfig") or {}
        budget = thinking.get("budget_tokens", thinking.get("thinkingBudget"))
        # Gemini takes a negative budget to mean that the model decides.
        return budget if isinstance(budget, int) and budget >= 0 else None
