from typing import List, Tuple


class UsageSummaryTable:
    """Values are right-aligned in one column, so that their sizes can be compared at a glance."""

    def __init__(self, title: str) -> None:
        self._title = title
        self._rows: List[Tuple[str, str, str]] = []

    def add_row(self, label: str, value: str, note: str = "") -> None:
        self._rows.append((label, value, note))

    def render(self) -> str:
        label_width = max((len(label) for label, _, _ in self._rows), default=0)
        value_width = max((len(value) for _, value, _ in self._rows), default=0)
        lines = [self._title]
        for label, value, note in self._rows:
            row = f"    {label:<{label_width}}  {value:>{value_width}}"
            lines.append(f"{row}  {note}" if note else row)
        return "\n".join(lines)
