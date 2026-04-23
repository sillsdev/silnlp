import regex

unicode_letter_regex = regex.compile("\\p{L}")


def contains_letter(token: str) -> bool:
    return unicode_letter_regex.search(token) is not None


def starts_with_capital_letter(token: str) -> bool:
    return len(token) > 0 and token[0].isupper()
