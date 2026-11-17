import re
from typing import Dict, List, Optional


class FolderNaming:
    def __init__(self, overrides: Dict[str, Dict[str, str]]):
        self._overrides = overrides

    def country_name(self, country: str) -> str:
        return self._override_for("countries", country) or country

    def language_name(self, language: str) -> str:
        return self._override_for("languages", language) or language

    def country_segment(self, country: str) -> str:
        return self._segment("countries", country)

    def language_segment(self, language: str) -> str:
        return self._segment("languages", language)

    def previous_segment(self, name: str) -> str:
        # Folders from earlier runs use this form, and the warning needs it to find them.
        return "_".join(word.capitalize() for word in self._words(name))

    def _segment(self, kind: str, name: str) -> str:
        override = self._override_for(kind, name)
        return self._join(override if override is not None else self._rearranged(name))

    def _rearranged(self, name: str) -> str:
        without_brackets = re.sub(r"\s*\([^)]*\)", "", name)
        # Languages are generally spoken and a standard form is the default, so neither word distinguishes a name.
        without_labels = re.sub(r"\s*\b(Spoken|Standard)\b", "", without_brackets)
        if "," not in without_labels:
            return without_labels
        base, qualifier = (part.strip() for part in without_labels.split(",", 1))
        return f"{qualifier} {base}" if qualifier else base

    def _join(self, name: str) -> str:
        return re.sub("_+", "_", "_".join(self._words(name)))

    def _words(self, name: str) -> List[str]:
        return name.replace(",", "").replace("-", " ").split()

    def _override_for(self, kind: str, name: str) -> Optional[str]:
        override = self._overrides[kind].get(name)
        return override if isinstance(override, str) and override.strip() else None
