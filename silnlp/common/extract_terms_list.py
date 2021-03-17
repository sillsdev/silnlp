import argparse
import os

from .environment import PT_PREPROCESSED_DIR
from .paratext import extract_terms_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts a Paratext Biblical Terms list")
    parser.add_argument("list", type=str, help="Biblical Terms list")
    args = parser.parse_args()

    terms_dir = os.path.join(PT_PREPROCESSED_DIR, "terms")
    os.makedirs(terms_dir, exist_ok=True)

    extract_terms_list(args.list)


if __name__ == "__main__":
    main()
