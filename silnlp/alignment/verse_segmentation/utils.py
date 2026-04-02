import regex

unicode_letter_regex = regex.compile("\\p{L}")


def contains_letter(token: str) -> bool:
    return unicode_letter_regex.search(token) is not None
