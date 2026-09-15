import ast
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "silnlp"

# Bases that contribute no attributes of their own, so a class inheriting only from them is still decidable.
EMPTY_BASES = {"ABC", "Generic", "Protocol", "object"}


class ClassIndex:
    """Every class defined under a source root, so what a class inherits can be resolved by name."""

    def __init__(self, source_root: Path) -> None:
        self._classes: List[Tuple[Path, ast.ClassDef]] = []
        self._by_name: Dict[str, List[ast.ClassDef]] = {}
        for path in sorted(source_root.rglob("*.py")):
            with warnings.catch_warnings():
                # Reading every file surfaces each invalid escape sequence in the repository.
                warnings.simplefilter("ignore", DeprecationWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._classes.append((path.relative_to(source_root.parent), node))
                    self._by_name.setdefault(node.name, []).append(node)

    def unresolved_references(self) -> List[str]:
        unresolved: Set[Tuple[str, int, str]] = set()
        for path, node in self._classes:
            available = self._available_names(node)
            if available is None:
                continue
            for attribute in self._self_attributes(node, ast.Load):
                if attribute.attr not in available and not attribute.attr.startswith("__"):
                    unresolved.add((str(path), attribute.lineno, attribute.attr))
        return [f"{path}:{line} {name}" for path, line, name in sorted(unresolved)]

    def _available_names(self, node: ast.ClassDef) -> Optional[Set[str]]:
        """The names self can refer to, or None when a base outside the source root makes that unknowable."""
        names = self._names_defined_by(node)
        pending = [self._base_name(base) for base in node.bases]
        seen: Set[str] = set()
        while pending:
            base = pending.pop()
            if base in EMPTY_BASES or base in seen:
                continue
            if base is None or base not in self._by_name:
                return None
            seen.add(base)
            for definition in self._by_name[base]:
                names |= self._names_defined_by(definition)
                pending.extend(self._base_name(inherited) for inherited in definition.bases)
        return names

    def _base_name(self, base: ast.expr) -> Optional[str]:
        if isinstance(base, ast.Subscript):
            return self._base_name(base.value)
        return getattr(base, "id", None) or getattr(base, "attr", None)

    def _names_defined_by(self, node: ast.ClassDef) -> Set[str]:
        names: Set[str] = set()
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(child.name)
            elif isinstance(child, ast.Assign):
                names.update(target.id for target in child.targets if isinstance(target, ast.Name))
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                names.add(child.target.id)
        for attribute in self._self_attributes(node, ast.Store):
            names.add(attribute.attr)
        return names

    def _self_attributes(self, node: ast.ClassDef, context: type) -> Iterator[ast.Attribute]:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(child.ctx, context):
                target = child.value
                if isinstance(target, ast.Name) and target.id == "self":
                    yield child


def test_every_self_reference_resolves_to_something_the_class_defines():
    # Deleting a method whose callers remain leaves a failure that surfaces only on an uncovered path.
    assert ClassIndex(SOURCE_ROOT).unresolved_references() == []
