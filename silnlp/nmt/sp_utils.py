from typing import Iterable


def decode_sp(line: str) -> str:
    #    return line.replace(" ", "").replace("\u2581", " ").lstrip()
    # Unicode 2581 (Lower One Eighth Block) used by SentencePiece for start-of-word indicator
    # Unicode 2580 (Upper Half Block) used for morpheme segment indicator
    return line.replace(" ", "").replace("\u2581\u2580", "").replace("\u2581", " ").lstrip()


def decode_sp_lines(lines: Iterable[str]) -> Iterable[str]:
    return map(decode_sp, lines)
