import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptTokenCount:
    """A request's count of prompt tokens, split by how a provider's prompt cache bills them."""

    total: int
    read_from_cache: int = 0
    written_to_cache: int = 0


@dataclass(frozen=True)
class SharedSystemMessageCacheStatus:
    """The system message a run's requests share, which an earlier request may already have cached."""

    num_tokens: int
    already_cached: bool


class PromptCachingPolicy(ABC):
    """How a provider caches the system message that a run's requests share."""

    @classmethod
    def for_model(cls, model: str) -> "PromptCachingPolicy":
        return AnthropicPromptCachingPolicy(model) if "claude" in model.lower() else PrefixPromptCachingPolicy()

    def mark(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages

    @abstractmethod
    def split_prompt(self, prompt_tokens: int, cache_status: SharedSystemMessageCacheStatus) -> PromptTokenCount:
        """The split an estimate relies on, which counts only the caching that requests ask for."""

    @abstractmethod
    def split_prompt_if_cached(
        self, prompt_tokens: int, cache_status: SharedSystemMessageCacheStatus
    ) -> PromptTokenCount:
        pass


class PrefixPromptCachingPolicy(PromptCachingPolicy):
    """A provider that may cache a repeated prefix unasked, and may bill every token it writes at a premium."""

    # Providers that cache prompts automatically leave a shorter prefix uncached.
    _MIN_CACHED_TOKENS = 1024

    def split_prompt(self, prompt_tokens: int, cache_status: SharedSystemMessageCacheStatus) -> PromptTokenCount:
        return PromptTokenCount(prompt_tokens, written_to_cache=prompt_tokens)

    def split_prompt_if_cached(
        self, prompt_tokens: int, cache_status: SharedSystemMessageCacheStatus
    ) -> PromptTokenCount:
        read = (
            cache_status.num_tokens
            if cache_status.already_cached and cache_status.num_tokens >= self._MIN_CACHED_TOKENS
            else 0
        )
        return PromptTokenCount(prompt_tokens, read_from_cache=read, written_to_cache=prompt_tokens - read)


class AnthropicPromptCachingPolicy(PromptCachingPolicy):
    """Anthropic's models cache a prompt only up to a block marked for caching, and only if it is long enough."""

    # From Anthropic's prompt caching docs; a model none of these match is taken to be as new as those needing 512.
    _MIN_TOKENS = (
        (re.compile(r"opus-4-[56]\b|haiku-4-5\b"), 4096),
        (re.compile(r"opus-4-7\b|3-5-haiku|haiku-3-5|3-haiku|mythos-preview"), 2048),
        (re.compile(r"opus-4-[18]\b|opus-4(?!-\d\b)|sonnet-4|sonnet-5(?!-\d\b)|claude-3"), 1024),
    )
    _NEWEST_MIN_TOKENS = 512

    def __init__(self, model: str) -> None:
        self._model = model
        # OpenRouter writes versions with a dot where Anthropic writes a dash.
        self._min_tokens = self._min_tokens_for(model.lower().replace(".", "-"))
        self._announced = False
        # Requests are estimated from several threads.
        self._lock = threading.Lock()

    def mark(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._marked(message) if message["role"] == "system" else message for message in messages]

    def split_prompt(self, prompt_tokens: int, cache_status: SharedSystemMessageCacheStatus) -> PromptTokenCount:
        if cache_status.num_tokens < self._min_tokens:
            return PromptTokenCount(prompt_tokens)
        self._announce()
        if cache_status.already_cached:
            return PromptTokenCount(prompt_tokens, read_from_cache=cache_status.num_tokens)
        return PromptTokenCount(prompt_tokens, written_to_cache=cache_status.num_tokens)

    def split_prompt_if_cached(
        self, prompt_tokens: int, cache_status: SharedSystemMessageCacheStatus
    ) -> PromptTokenCount:
        return self.split_prompt(prompt_tokens, cache_status)

    def _marked(self, message: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **message,
            "content": [{"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}],
        }

    def _min_tokens_for(self, model: str) -> int:
        return next((tokens for pattern, tokens in self._MIN_TOKENS if pattern.search(model)), self._NEWEST_MIN_TOKENS)

    def _announce(self) -> None:
        with self._lock:
            if self._announced:
                return
            self._announced = True
        LOGGER.info(
            "Dry run: requests to %s mark the system message they share for caching, so it is estimated at the "
            "cache-write price until an earlier request has cached it, and at the cache-read price after.",
            self._model,
        )
