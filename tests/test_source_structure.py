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


class ModuleIndex:
    """Every name each module defines or re-exports, resolved without importing anything."""

    def __init__(self, source_root: Path) -> None:
        self._root = source_root
        self._names: Dict[str, Set[str]] = {}
        for path in sorted(source_root.rglob("*.py")):
            self._names[self._module_of(path)] = self._names_defined_by(path)

    def unresolved_imports(self) -> List[str]:
        problems = []
        for path in sorted(self._root.rglob("*.py")):
            module = self._module_of(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                target = self._resolved_module(module, node)
                if target not in self._names:
                    continue
                for alias in node.names:
                    if alias.name != "*" and alias.name not in self._names[target]:
                        location = path.relative_to(self._root.parent)
                        problems.append(f"{location}:{node.lineno} {alias.name} is not defined in {target}")
        return sorted(problems)

    def _module_of(self, path: Path) -> str:
        parts = path.relative_to(self._root.parent).with_suffix("").parts
        return ".".join(parts[:-1]) if parts[-1] == "__init__" else ".".join(parts)

    def _resolved_module(self, module: str, node: ast.ImportFrom) -> str:
        if node.level == 0:
            return node.module or ""
        base = module.split(".")[: -node.level]
        return ".".join(base + ([node.module] if node.module else []))

    def _names_defined_by(self, path: Path) -> Set[str]:
        names: Set[str] = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.Import):
                names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in node.names)
        return names


# Already broken on master: extract_terms_list exists nowhere, and get_scripture_path is a method on
# SilNlpEnv rather than a function. Listed so the check still has teeth for everything else.
IMPORTS_BROKEN_ON_MASTER = [
    "silnlp/common/extract_terms_list.py:4 extract_terms_list is not defined in silnlp.common.paratext",
    "silnlp/smt/preprocess.py:9 get_scripture_path is not defined in silnlp.common.corpus",
]


def test_every_import_names_something_its_module_defines():
    # Dropping a re-export breaks its importers at load time, which no test of behaviour reaches.
    assert ModuleIndex(SOURCE_ROOT).unresolved_imports() == IMPORTS_BROKEN_ON_MASTER


def test_every_self_reference_resolves_to_something_the_class_defines():
    # Deleting a method whose callers remain leaves a failure that surfaces only on an uncovered path.
    assert ClassIndex(SOURCE_ROOT).unresolved_references() == []
