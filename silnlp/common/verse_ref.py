from typing import Iterable, Union

from ..common.canon import book_id_to_number, book_number_to_id

_VERSE_RANGE_SEPARATOR = "-"
_VERSE_SEQUENCE_INDICATOR = ","
_CHAPTER_DIGIT_SHIFTER = 1000
_BOOK_DIGIT_SHIFTER = _CHAPTER_DIGIT_SHIFTER * _CHAPTER_DIGIT_SHIFTER
_BCV_MAX_VALUE = _CHAPTER_DIGIT_SHIFTER


def is_verse_parseable(verse: str) -> bool:
    return (
        len(verse) != 0
        and verse[0].isdigit()
        and verse[-1] != _VERSE_RANGE_SEPARATOR
        and verse[-1] != _VERSE_SEQUENCE_INDICATOR
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


def get_bbbcccvvv(book_num: int, chapter_num: int, verse_num: int) -> int:
    return (
        (book_num % _BCV_MAX_VALUE) * _BOOK_DIGIT_SHIFTER
        + ((chapter_num % _BCV_MAX_VALUE) * _CHAPTER_DIGIT_SHIFTER if chapter_num >= 0 else 0)
        + (verse_num % _BCV_MAX_VALUE if verse_num >= 0 else 0)
    )


class VerseRef:
    def __init__(self, book: Union[str, int], chapter: Union[str, int], verse: Union[str, int]) -> None:
        if isinstance(book, str):
            self.book_num = book_id_to_number(book)
        else:
            self.book_num = book

        if isinstance(chapter, str):
            if not chapter.isdigit():
                raise ValueError("The chapter is invalid.")
            chapter_num = int(chapter)
            if chapter_num < 0:
                raise ValueError("The chapter is invalid.")
            self.chapter_num = chapter_num
        else:
            self.chapter_num = chapter

        if isinstance(verse, str):
            if not is_verse_parseable(verse):
                raise ValueError("The verse is invalid.")
            self.verse = verse
            self.verse_num = get_verse_num(verse)
            self.has_multiple = verse.find(_VERSE_RANGE_SEPARATOR) != -1 or verse.find(_VERSE_SEQUENCE_INDICATOR) != -1
        else:
            self.verse = str(verse)
            self.verse_num = verse

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

    @classmethod
    def from_bbbcccvvv(cls, bbbcccvvv: int) -> "VerseRef":
        book = bbbcccvvv // 1000000
        chapter = bbbcccvvv % 1000000 // 1000
        verse = bbbcccvvv % 1000
        return VerseRef(book, chapter, verse)

    @property
    def book(self) -> str:
        return book_number_to_id(self.book_num)

    @property
    def chapter(self) -> str:
        return "" if self.chapter_num < 0 else str(self.chapter_num)

    @property
    def bbbcccvvv(self) -> int:
        return get_bbbcccvvv(self.book_num, self.chapter_num, self.verse_num)

    def all_verses(self) -> Iterable["VerseRef"]:
        parts = self.verse.split(_VERSE_SEQUENCE_INDICATOR)
        for part in parts:
            pieces = part.split(_VERSE_RANGE_SEPARATOR)
            start_verse = VerseRef(self.book, self.chapter, pieces[0])
            yield start_verse

            if len(pieces) > 1:
                last_verse = VerseRef(self.book, self.chapter, pieces[1])
                for verse_num in range(start_verse.verse_num + 1, last_verse.verse_num):
                    yield VerseRef(self.book, self.chapter, str(verse_num))
                yield last_verse

    def simplify(self) -> "VerseRef":
        return VerseRef(self.book, self.chapter, str(self.verse_num))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VerseRef):
            raise NotImplementedError

        return (
            self.book_num == other.book_num
            and self.chapter_num == other.chapter_num
            and self.verse_num == other.verse_num
            and self.verse == other.verse
        )

    def __hash__(self) -> int:
        if self.verse is not None:
            return self.bbbcccvvv ^ hash(self.verse)
        return self.bbbcccvvv

    def __str__(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"

    def __repr__(self) -> str:
        return str(self)
