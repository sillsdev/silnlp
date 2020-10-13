# from __future__ import annotations
from typing import Iterable

from nlp.common.canon import book_id_to_number, book_number_to_id

VERSE_RANGE_SEPARATOR = "-"
VERSE_SEQUENCE_INDICATOR = ","


def is_verse_parseable(verse: str) -> bool:
    return (
        len(verse) != 0
        and verse[0].isdigit()
        and verse[-1] != VERSE_RANGE_SEPARATOR
        and verse[-1] != VERSE_SEQUENCE_INDICATOR
    )


def get_verse_num(verse: str) -> int:
    if str == "":
        return -1

    v_num = 0
    for i in range(len(verse)):
        ch = verse[i]
        if not ch.isdigit():
            if i == 0:
                return -1
            break

        v_num = (v_num * 10) + int(ch)
        if v_num > 999:
            return -1
    return v_num


class VerseRef:
    def __init__(self, book: str, chapter: str, verse: str) -> None:
        book_num = book_id_to_number(book)
        if not chapter.isdigit():
            raise ValueError("The chapter is invalid.")
        chapter_num = int(chapter)
        if chapter_num < 0:
            raise ValueError("The chapter is invalid.")
        if not is_verse_parseable(verse):
            raise ValueError("The verse is invalid.")

        self.book_num = book_num
        self.chapter_num = int(chapter)
        self.verse = verse
        self.verse_num = get_verse_num(verse)
        self.has_multiple = verse.find(VERSE_RANGE_SEPARATOR) != -1 or verse.find(VERSE_SEQUENCE_INDICATOR) != -1

    @classmethod
    def from_string(cls, verse_str: str) -> "VerseRef":
        b_cv = verse_str.strip().split(" ")
        if len(b_cv) != 2:
            raise ValueError("The verse reference is invalid.")

        c_v = b_cv[1].split(":")
        if len(c_v) != 2:
            raise ValueError("The verse reference is invalid.")
        return VerseRef(b_cv[0], c_v[0], c_v[1])

    @classmethod
    def from_range(cls, start: "VerseRef", end: "VerseRef") -> "VerseRef":
        if start.book_num != end.book_num or start.chapter_num != end.chapter_num:
            raise ValueError("The start and end verses are not in the same chapter.")
        if start.has_multiple:
            raise ValueError("This start verse contains multiple verses.")
        if end.has_multiple:
            raise ValueError("This end verse contains multiple verses.")

        return VerseRef(start.book, start.chapter, f"{start.verse_num}-{end.verse_num}")

    @property
    def book(self) -> str:
        return book_number_to_id(self.book_num)

    @property
    def chapter(self) -> str:
        return "" if self.chapter_num < 0 else str(self.chapter_num)

    def all_verses(self) -> Iterable["VerseRef"]:
        parts = self.verse.split(VERSE_SEQUENCE_INDICATOR)
        for part in parts:
            pieces = part.split(VERSE_RANGE_SEPARATOR)
            start_verse = VerseRef(self.book, self.chapter, pieces[0])
            yield start_verse

            if len(pieces) > 1:
                last_verse = VerseRef(self.book, self.chapter, pieces[1])
                for verse_num in range(start_verse.verse_num + 1, last_verse.verse_num):
                    yield VerseRef(self.book, self.chapter, str(verse_num))
                yield last_verse

    def simplify(self) -> "VerseRef":
        return VerseRef(self.book, self.chapter, str(self.verse_num))

    def __str__(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"

    def __repr__(self) -> str:
        return str(self)
