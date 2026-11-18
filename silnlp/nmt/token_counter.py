from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class TokenCounter(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> Optional[int]:
        pass

    # None means the tokens could not be counted
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> Optional[int]:
        total = 0
        for message in messages:
            tokens = self.count_tokens(message["content"])
            if tokens is None:
                return None
            total += tokens
        return total
