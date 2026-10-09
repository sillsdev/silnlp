from silnlp.common.argument_parser import SilnlpArgumentParser

print("Loading models...")


def main() -> None:
    parser = SilnlpArgumentParser(description="Prints while it is imported")
    parser.parse_args()


if __name__ == "__main__":
    main()
