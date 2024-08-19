from typing import Hashable

import streamlit as st
from consts import BOOKS_ABBREVS
from cryptography.fernet import Fernet


def check_required(f: Hashable, *args, func=None):
    if func is None:
        if not all([arg is not None and len(arg) > 0 for arg in args]):
            st.session_state.errors[f] = "A required field was left blank"
            st.rerun()
    elif not all([func(arg) for arg in args]):
        st.session_state.errors[f] = "A required field was left blank"
        st.rerun()
    if f in st.session_state.errors:
        del st.session_state.errors[f]


def check_error(f: Hashable):
    if "errors" in st.session_state:
        if f in st.session_state.errors:
            st.error(st.session_state.errors[f])
    else:
        st.session_state.errors = {}


def set_success(f: Hashable, message: str):
    if "success_messages" not in st.session_state:
        st.session_state.success_messages = {}
    st.session_state.success_messages[f] = message


def check_success(f: Hashable):
    if "success_messages" in st.session_state:
        if f in st.session_state.success_messages:
            st.success(st.session_state.success_messages[f])
            st.toast(st.session_state.success_messages[f])
            del st.session_state.success_messages[f]
    else:
        st.session_state.success_messages = {}


def simplify_books(books: "list[str]"):
    if set(BOOKS_ABBREVS[BOOKS_ABBREVS.index("GEN") : BOOKS_ABBREVS.index("MAL") + 1]).issubset(books):
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index("GEN") : BOOKS_ABBREVS.index("MAL") + 1]:
            books.remove(b)
        books.append("OT")
    if set(BOOKS_ABBREVS[BOOKS_ABBREVS.index("MAT") : BOOKS_ABBREVS.index("REV") + 1]).issubset(books):
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index("MAT") : BOOKS_ABBREVS.index("REV") + 1]:
            books.remove(b)
        books.append("NT")
    if set(BOOKS_ABBREVS[BOOKS_ABBREVS.index("REV") + 1 :]).issubset(books):
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index("REV") + 1 :]:
            books.remove(b)
        books.append("DT")
    return set(books)


def expand_books(books: "list[str]"):
    if "OT" in books:
        books.remove("OT")
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index("GEN") : BOOKS_ABBREVS.index("MAL") + 1]:
            books.append(b)
    if "NT" in books:
        books.remove("NT")
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index("MAT") : BOOKS_ABBREVS.index("REV") + 1]:
            books.append(b)
    if "DT" in books:
        books.remove("DT")
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index("REV") + 1 :]:
            books.append(b)
    return set(books)


_MY_KEY = Fernet.generate_key()


def encrypt(to_encrypt: bytes) -> bytes:
    return Fernet(key=_MY_KEY).encrypt(to_encrypt)


def decrypt(to_decrypt: bytes) -> bytes:
    return Fernet(key=_MY_KEY).decrypt(to_decrypt)
