import argparse
import json
import sys
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Sequence


class _JsonValue:
    def __init__(self, value: Any) -> None:
        self._value = value

    def to_json(self) -> Any:
        if self._value is None or isinstance(self._value, (str, bool, int, float)):
            return self._value
        if isinstance(self._value, PurePath):
            return str(self._value)
        if isinstance(self._value, (set, frozenset)):
            return sorted((_JsonValue(item).to_json() for item in self._value), key=str)
        if isinstance(self._value, (list, tuple)):
            return [_JsonValue(item).to_json() for item in self._value]
        return str(self._value)


class ArgumentDescription:
    # Subclasses come before their bases, since a store_true action is also a store_const action.
    _KINDS = (
        (argparse.BooleanOptionalAction, "boolean_optional"),
        (argparse._StoreTrueAction, "store_true"),
        (argparse._StoreFalseAction, "store_false"),
        (argparse._StoreConstAction, "store_const"),
        (argparse._ExtendAction, "extend"),
        (argparse._AppendConstAction, "append_const"),
        (argparse._AppendAction, "append"),
        (argparse._CountAction, "count"),
        (argparse._StoreAction, "store"),
    )

    def __init__(self, action: argparse.Action, formatter: argparse.HelpFormatter, group_title: Optional[str]) -> None:
        self._action = action
        self._formatter = formatter
        self._group_title = group_title

    def is_hidden(self) -> bool:
        return self._action.help == argparse.SUPPRESS or isinstance(
            self._action, (argparse._HelpAction, argparse._VersionAction)
        )

    def name(self) -> str:
        long_flags = [flag for flag in self._action.option_strings if flag.startswith("--")]
        if len(long_flags) > 0:
            return long_flags[0]
        if len(self._action.option_strings) > 0:
            return self._action.option_strings[0]
        return self._action.dest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "flags": list(self._action.option_strings),
            "dest": self._action.dest,
            "kind": self._kind(),
            "value_type": self._value_type(),
            "nargs": self._action.nargs,
            "required": self._action.required,
            "default": (
                None if self._action.default is argparse.SUPPRESS else _JsonValue(self._action.default).to_json()
            ),
            "const": _JsonValue(self._action.const).to_json(),
            "choices": self._choices(),
            "metavar": _JsonValue(self._action.metavar).to_json(),
            "help": None if self._action.help is None else self._formatter._expand_help(self._action),
            "group": self._group_title,
        }

    def _choices(self) -> Optional[List[Any]]:
        choices = self._action.choices
        if choices is None:
            return None
        if not isinstance(choices, (set, frozenset)):
            choices = list(choices)
        return _JsonValue(choices).to_json()

    def _kind(self) -> str:
        for action_class, kind in self._KINDS:
            if isinstance(self._action, action_class):
                return kind
        return "other"

    def _value_type(self) -> str:
        if self._action.type is int:
            return "integer"
        if self._action.type is float:
            return "number"
        if isinstance(self._action.type, type) and issubclass(self._action.type, PurePath):
            return "path"
        return "string"


class ParserDescription:
    """Reads argparse internals, since argparse offers no public way to list a parser's arguments."""

    _FORMAT = 1

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self._parser = parser

    def to_json(self) -> str:
        arguments = [argument for argument in self._arguments() if not argument.is_hidden()]
        return json.dumps(
            {
                "format": self._FORMAT,
                "prog": self._parser.prog,
                "description": self._parser.description,
                "epilog": self._parser.epilog,
                "arguments": [argument.to_dict() for argument in arguments],
                "mutually_exclusive_groups": self._mutually_exclusive_groups(),
            },
            indent=2,
        )

    def _arguments(self) -> List[ArgumentDescription]:
        formatter = self._parser._get_formatter()
        group_titles = {
            id(action): group.title for group in self._parser._action_groups for action in group._group_actions
        }
        return [
            ArgumentDescription(action, formatter, group_titles.get(id(action))) for action in self._parser._actions
        ]

    def _mutually_exclusive_groups(self) -> List[Dict[str, Any]]:
        formatter = self._parser._get_formatter()
        groups = []
        for group in self._parser._mutually_exclusive_groups:
            members = [ArgumentDescription(action, formatter, None) for action in group._group_actions]
            groups.append(
                {
                    "required": group.required,
                    "arguments": [member.name() for member in members if not member.is_hidden()],
                }
            )
        return groups


class _DescribeJsonAction(argparse.Action):
    def __init__(self, option_strings: Sequence[str], dest: str, **kwargs: Any) -> None:
        super().__init__(
            option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, metavar="PATH", help=argparse.SUPPRESS
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        description = ParserDescription(parser).to_json()
        if values == "-":
            sys.stdout.write(description + "\n")
        else:
            Path(values).write_text(description + "\n", encoding="utf-8")
        parser.exit()


class SilnlpArgumentParser(argparse.ArgumentParser):
    """Lets tools such as a web front end read a script's options by running it with --describe-json PATH."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Parent parsers are built without help, and a second --describe-json would conflict in the child.
        if self.add_help:
            self.add_argument("--describe-json", action=_DescribeJsonAction)
