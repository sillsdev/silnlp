from abc import ABC, abstractmethod
from typing import Optional


class TokenCounter(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> Optional[int]:
        pass
