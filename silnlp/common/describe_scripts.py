import ast
import json
import runpy
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

from .argument_parser import SilnlpArgumentParser


class ScriptModules:
    def __init__(self, package_dir: Path) -> None:
        self._package_dir = package_dir

    def names(self) -> List[str]:
        return [self._module_name(path) for path in sorted(self._package_dir.rglob("*.py")) if self._is_script(path)]

    def _module_name(self, path: Path) -> str:
        return ".".join(path.relative_to(self._package_dir.parent).with_suffix("").parts)

    def _is_script(self, path: Path) -> bool:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            # Listing the module lets the describe step report the syntax error instead of hiding the script.
            return f"{SilnlpArgumentParser.__name__}(" in source
        return any(self._constructs_parser(node) for node in ast.walk(tree))

    def _constructs_parser(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name):
            return node.func.id == SilnlpArgumentParser.__name__
        return isinstance(node.func, ast.Attribute) and node.func.attr == SilnlpArgumentParser.__name__


class ScriptDescriber:
    def describe(self, module: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "description.json"
            try:
                exit_status = self._run(module, output_path)
            except Exception as e:
                return {"module": module, "error": f"{type(e).__name__}: {e}"}
            if not output_path.is_file():
                return {"module": module, "error": f"Exited with status {exit_status} before describing its arguments"}
            return {"module": module, "parser": json.loads(output_path.read_text(encoding="utf-8"))}

    def _run(self, module: str, output_path: Path) -> Any:
        original_argv = sys.argv
        sys.argv = [module, "--describe-json", str(output_path)]
        try:
            # Anything the module prints would corrupt the JSON lines this command writes to stdout.
            with redirect_stdout(sys.stderr):
                runpy.run_module(module, run_name="__main__", alter_sys=True)
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = original_argv
        return 0


def main() -> None:
    parser = SilnlpArgumentParser(
        description="Describe the command-line arguments of silnlp scripts, one JSON object per line"
    )
    parser.add_argument("modules", nargs="*", help="Modules to describe. Describes every script if omitted.")
    parser.add_argument("--list", action="store_true", help="Print the script modules as a JSON list instead")
    args = parser.parse_args()

    modules = args.modules
    if args.list or len(modules) == 0:
        modules = ScriptModules(Path(__file__).resolve().parent.parent).names()
    if args.list:
        print(json.dumps(modules))
        return

    describer = ScriptDescriber()
    for module in modules:
        print(json.dumps(describer.describe(module)), flush=True)


if __name__ == "__main__":
    main()
