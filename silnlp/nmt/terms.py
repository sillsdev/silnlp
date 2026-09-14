import logging
from typing import List, Optional, Set, Union

LOGGER = logging.getLogger(__name__)


class TermCategories:
    def __init__(self, configured: Optional[Union[str, List[str]]]) -> None:
        if isinstance(configured, str):
            configured = [category.strip() for category in configured.split(",")]
        self._categories = configured

    def excludes_everything(self) -> bool:
        return self._categories is not None and len(self._categories) == 0

    def includes(self, category: str) -> bool:
        return self._categories is None or category in self._categories

    def as_set(self) -> Optional[Set[str]]:
        return None if self._categories is None else set(self._categories)


class GlossLanguages:
    """The languages Paratext term lists carry glosses for."""

    _ISOS = ["fr", "en", "id", "es", "pt"]

    def includes(self, iso: Union[bool, str, None]) -> bool:
        return iso in self._ISOS

    def any_in(self, isos: Set[str]) -> bool:
        return len(isos.intersection(self._ISOS)) > 0

    def first_in(self, isos: Set[str]) -> Optional[str]:
        matches = list(isos.intersection(self._ISOS))
        return matches[0] if matches else None

    def describe(self) -> str:
        return ", ".join(self._ISOS)


class GlossLanguage:
    def __init__(self, include_glosses: Union[bool, str], source_isos: Set[str], target_isos: Set[str]) -> None:
        self._requested = include_glosses
        self._source_isos = source_isos
        self._target_isos = target_isos
        self._languages = GlossLanguages()
        self._iso = self._resolve()

    def is_available(self) -> bool:
        return self._iso is not None

    def iso(self) -> Optional[str]:
        return self._iso

    def can_serve_as_target(self) -> bool:
        return self._iso is not None and self._iso in self._target_isos

    def can_serve_as_source(self) -> bool:
        return self._iso is not None and (self._iso in self._source_isos or self._iso == self._requested)

    def _resolve(self) -> Optional[str]:
        if not self._requested:
            return None
        requested = str(self._requested).lower()
        if requested == "true":
            return self._first_supported()
        if not self._languages.includes(requested):
            LOGGER.warning(
                f"Gloss language code, {requested}, does not match the supported gloss language codes: "
                f"{self._languages.describe()}."
            )
            return None
        return requested

    def _first_supported(self) -> Optional[str]:
        match = self._languages.first_in(self._source_isos) or self._languages.first_in(self._target_isos)
        if match is None:
            LOGGER.warning(
                "Glosses could not be included. No source or target language matches any of the supported gloss "
                f"language codes: {self._languages.describe()}."
            )
        return match
