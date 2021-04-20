import argparse

from .environment import MT_TERMS_DIR
from .paratext import extract_terms_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts a Paratext Biblical Terms list")
    parser.add_argument("list", type=str, help="Biblical Terms list")
    args = parser.parse_args()

    MT_TERMS_DIR.mkdir(exist_ok=True, parents=True)

    extract_terms_list(args.list)


if __name__ == "__main__":
    main()
