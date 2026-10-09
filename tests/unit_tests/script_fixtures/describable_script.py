from silnlp.common.argument_parser import SilnlpArgumentParser


def main() -> None:
    parser = SilnlpArgumentParser(description="Counts the verses in a corpus")
    parser.add_argument("corpus", help="Corpus to count")
    parser.add_argument("--by-book", action="store_true", help="Count each book separately")
    parser.parse_args()
    raise AssertionError("Describing a script must stop before the script does any work")


if __name__ == "__main__":
    main()
