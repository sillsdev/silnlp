import json
import subprocess
import sys
from pathlib import Path

from silnlp.common.describe_scripts import ScriptDescriber, ScriptModules


def fixture_module(name: str) -> str:
    return f"tests.unit_tests.script_fixtures.{name}"


def write_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_lists_modules_that_build_a_silnlp_argument_parser(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    write_module(package_dir / "by_name.py", "parser = SilnlpArgumentParser()\n")
    write_module(package_dir / "sub" / "by_attribute.py", "parser = argument_parser.SilnlpArgumentParser()\n")
    write_module(package_dir / "mentions_only.py", '# SilnlpArgumentParser()\nNAME = "SilnlpArgumentParser()"\n')
    write_module(package_dir / "defines.py", "class SilnlpArgumentParser(ArgumentParser):\n    pass\n")
    write_module(package_dir / "plain.py", "parser = argparse.ArgumentParser()\n")
    write_module(package_dir / "broken.py", "parser = SilnlpArgumentParser(\n")

    names = ScriptModules(package_dir).names()

    assert names == ["pkg.broken", "pkg.by_name", "pkg.sub.by_attribute"]


def test_describes_a_script_without_running_it() -> None:
    result = ScriptDescriber().describe(fixture_module("describable_script"))

    assert result["module"] == fixture_module("describable_script")
    assert result["parser"]["description"] == "Counts the verses in a corpus"
    assert [argument["name"] for argument in result["parser"]["arguments"]] == ["corpus", "--by-book"]


def test_reports_a_script_that_fails_to_import() -> None:
    result = ScriptDescriber().describe(fixture_module("broken_import_script"))

    assert result["error"].startswith("ImportError: cannot import name 'ParserThatDoesNotExist'")


def test_reports_a_module_that_never_parses_arguments() -> None:
    result = ScriptDescriber().describe(fixture_module("no_parser_script"))

    assert result["error"] == "Exited with status 0 before describing its arguments"


def test_reports_a_missing_module() -> None:
    result = ScriptDescriber().describe(fixture_module("does_not_exist"))

    assert result["error"].startswith("ImportError: No module named")


def test_command_writes_one_json_line_per_module_despite_script_output() -> None:
    modules = [fixture_module("noisy_script"), fixture_module("no_parser_script")]

    result = subprocess.run(
        [sys.executable, "-m", "silnlp.common.describe_scripts", *modules],
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert [line["module"] for line in lines] == modules
    assert lines[0]["parser"]["description"] == "Prints while it is imported"
    assert "Loading models..." in result.stderr


def test_command_lists_silnlp_scripts() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "silnlp.common.describe_scripts", "--list"],
        capture_output=True,
        text=True,
        check=True,
    )

    names = json.loads(result.stdout)
    assert "silnlp.common.describe_scripts" in names
    assert "silnlp.nmt.experiment" in names
    assert "silnlp.common.argument_parser" not in names
