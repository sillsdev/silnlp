from silnlp.common.argument_parser import ParserThatDoesNotExist  # type: ignore[attr-defined]


def main() -> None:
    ParserThatDoesNotExist().parse_args()


if __name__ == "__main__":
    main()
