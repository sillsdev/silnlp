import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from silnlp.common.argument_parser import SilnlpArgumentParser


def describe(parser: SilnlpArgumentParser, tmp_path: Path) -> Dict[str, Any]:
    output_path = tmp_path / "description.json"
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--describe-json", str(output_path)])
    assert exit_info.value.code == 0
    return json.loads(output_path.read_text(encoding="utf-8"))


def argument(description: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(argument for argument in description["arguments"] if argument["name"] == name)


def test_describes_parser(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser(prog="experiment.py", description="Run an experiment", epilog="See the wiki")

    description = describe(parser, tmp_path)

    assert description["format"] == 1
    assert description["prog"] == "experiment.py"
    assert description["description"] == "Run an experiment"
    assert description["epilog"] == "See the wiki"


def test_describing_ignores_missing_required_arguments(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--queue", required=True, help="Queue")

    description = describe(parser, tmp_path)

    assert argument(description, "experiment")["required"]
    assert argument(description, "--queue")["required"]


def test_describes_positional_argument(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("linregress_file", nargs="?", default=None, help="Optional file")

    description = describe(parser, tmp_path)

    assert argument(description, "experiment") == {
        "name": "experiment",
        "flags": [],
        "dest": "experiment",
        "kind": "store",
        "value_type": "string",
        "nargs": None,
        "required": True,
        "default": None,
        "const": None,
        "choices": None,
        "metavar": None,
        "help": "Experiment name",
        "group": "positional arguments",
    }
    assert not argument(description, "linregress_file")["required"]


def test_names_option_by_its_first_long_flag(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("-n", "--num-devices", "--devices", type=int, default=1, help="Devices")
    parser.add_argument("-q", help="Quiet")

    description = describe(parser, tmp_path)

    assert argument(description, "--num-devices")["flags"] == ["-n", "--num-devices", "--devices"]
    assert argument(description, "-q")["dest"] == "q"


def test_describes_value_types(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("--count", type=int, default=3, help="Count")
    parser.add_argument("--rate", type=float, default=0.5, help="Rate")
    parser.add_argument("--path", type=Path, default=Path("MT/experiments"), help="Path")
    parser.add_argument("--name", type=str.lower, help="Name")
    parser.add_argument("--plain", help="Plain")

    description = describe(parser, tmp_path)

    assert argument(description, "--count")["value_type"] == "integer"
    assert argument(description, "--count")["default"] == 3
    assert argument(description, "--rate")["value_type"] == "number"
    assert argument(description, "--path")["value_type"] == "path"
    assert argument(description, "--path")["default"] == "MT/experiments"
    assert argument(description, "--name")["value_type"] == "string"
    assert argument(description, "--plain")["value_type"] == "string"


def test_describes_action_kinds(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Stats")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false", help="Mixed precision")
    parser.add_argument("--fast", dest="speed", action="store_const", const="fast", help="Fast")
    parser.add_argument("--book", action="append", help="Book")
    parser.add_argument("--marker", action="append_const", const="x", help="Marker")
    parser.add_argument("--scorer", action="extend", nargs="+", help="Scorer")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity")
    parser.add_argument("--color", action=argparse.BooleanOptionalAction, default=True, help="Color")

    description = describe(parser, tmp_path)

    kinds = {argument["name"]: argument["kind"] for argument in description["arguments"]}
    assert kinds == {
        "--stats": "store_true",
        "--no-mixed-precision": "store_false",
        "--fast": "store_const",
        "--book": "append",
        "--marker": "append_const",
        "--scorer": "extend",
        "--verbose": "count",
        "--color": "boolean_optional",
    }
    assert argument(description, "--stats")["default"] is False
    assert argument(description, "--stats")["nargs"] == 0
    assert argument(description, "--fast")["const"] == "fast"
    assert argument(description, "--color")["flags"] == ["--color", "--no-color"]


def test_describes_choices_and_list_defaults(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("--scorers", nargs="*", choices=["chrf3", "bleu"], default=["bleu"], help="Scorers")
    parser.add_argument("--metrics", nargs="+", choices={"ter", "bleu"}, default={"ter", "bleu"}, help="Metrics")
    parser.add_argument("--level", choices=range(1, 4), type=int, help="Level")

    description = describe(parser, tmp_path)

    assert argument(description, "--scorers")["choices"] == ["chrf3", "bleu"]
    assert argument(description, "--scorers")["default"] == ["bleu"]
    assert argument(description, "--scorers")["nargs"] == "*"
    assert argument(description, "--metrics")["choices"] == ["bleu", "ter"]
    assert argument(description, "--metrics")["default"] == ["bleu", "ter"]
    assert argument(description, "--level")["choices"] == [1, 2, 3]


def test_describes_optional_value_with_const(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("--copy-from", nargs="?", const=Path("Downloads"), default=None, type=Path, help="Copy")

    description = describe(parser, tmp_path)

    assert argument(description, "--copy-from")["nargs"] == "?"
    assert argument(description, "--copy-from")["const"] == "Downloads"


def test_expands_help_as_help_output_does(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--retries", type=int, default=5, help="Retries")
    parser.add_argument("--mode", choices=["a", "b"], default="a", help="One of %(choices)s")

    description = describe(parser, tmp_path)

    assert argument(description, "--retries")["help"] == "Retries (default: 5)"
    assert argument(description, "--mode")["help"] == "One of a, b (default: a)"


def test_omits_help_and_hidden_options(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("--secret", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version="1.0")
    parser.add_argument("--shown", help="Shown")

    description = describe(parser, tmp_path)

    assert [argument["name"] for argument in description["arguments"]] == ["--shown"]


def test_describes_argument_groups(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("--plain", help="Plain")
    advanced = parser.add_argument_group("Advanced")
    advanced.add_argument("--seed", type=int, help="Seed")

    description = describe(parser, tmp_path)

    assert argument(description, "--plain")["group"] == "options"
    assert argument(description, "--seed")["group"] == "Advanced"


def test_describes_mutually_exclusive_groups(tmp_path: Path) -> None:
    parser = SilnlpArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--no-test", action="store_true", help="No test")
    group.add_argument("--test100", action="store_true", help="Test 100")

    description = describe(parser, tmp_path)

    assert description["mutually_exclusive_groups"] == [{"required": True, "arguments": ["--no-test", "--test100"]}]


def test_parent_parser_can_be_shared(tmp_path: Path) -> None:
    parent = SilnlpArgumentParser(add_help=False)
    parent.add_argument("--shared", help="Shared")
    child = SilnlpArgumentParser(parents=[parent])

    description = describe(child, tmp_path)

    assert argument(description, "--shared")["help"] == "Shared"


def test_parses_normally_without_describe_json() -> None:
    parser = SilnlpArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args(["MyExperiment", "--stats"])

    assert vars(args) == {"experiment": "MyExperiment", "stats": True}


def test_describe_json_dash_writes_to_stdout() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "silnlp.common.describe_scripts", "--describe-json", "-"],
        capture_output=True,
        text=True,
        check=True,
    )

    description = json.loads(result.stdout)
    assert argument(description, "--list")["kind"] == "store_true"

