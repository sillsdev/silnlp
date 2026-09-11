import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from silnlp.nmt.remote_llm_config import (
    Completion,
    CompletionClient,
    CompletionClientFactory,
    RemoteLLMConfig,
    TokenLogprob,
)

# Matches the numbered source segments that the batch prompt asks the model to translate.
_NUMBERED_SEGMENT = re.compile(r"^\s*(\d{1,4})\.\s+(.*)$")


@dataclass
class CompletionStats:
    requests: List[List[Dict[str, str]]] = field(default_factory=list)

    @property
    def num_requests(self) -> int:
        return len(self.requests)


class MockCompletionClient(CompletionClient):
    """Answers translation requests locally, in the format the prompt asks for.

    The "translations" are just marked-up source text: the smoke test asserts structural
    properties (one output line per input segment, files in the right places), not translation
    quality, and a real call would need a provider API key and cost money.
    """

    def __init__(self, stats: CompletionStats) -> None:
        self._stats = stats

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        self._stats.requests.append(messages)
        segments = self._numbered_segments(messages[-1]["content"])
        if len(segments) == 0:
            # The single-segment prompt is unnumbered and expects a bare translation.
            text = "translated"
        else:
            text = "\n".join(f"{number}. translated {segment}" for number, segment in segments)
        if not logprobs:
            return Completion(text)
        return Completion(text, [TokenLogprob(token, -0.1) for token in text.split()])

    def supports_logprobs(self) -> bool:
        return True

    @staticmethod
    def _numbered_segments(user_message: str) -> List[tuple]:
        # The segments to translate are the numbered lines in the final block of the prompt,
        # after the instruction; retrieved examples are rendered without numbering.
        segments = []
        for line in user_message.splitlines():
            match = _NUMBERED_SEGMENT.match(line)
            if match is not None:
                segments.append((int(match.group(1)), match.group(2)))
        return segments


class MockCompletionClientFactory(CompletionClientFactory):
    def __init__(self, stats: Optional[CompletionStats] = None) -> None:
        self._stats = stats or CompletionStats()

    @property
    def stats(self) -> CompletionStats:
        return self._stats

    def create(self, config: RemoteLLMConfig) -> CompletionClient:
        return MockCompletionClient(self._stats)
