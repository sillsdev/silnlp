import streamlit as st
from consts import BOOKS_ABBREVS

def check_required(f, *args, func = None):
    if func is None:
        if not all([arg is not None and len(arg) > 0 for arg in args]):
            st.session_state.errors[f] = 'A required field was left blank'
            st.rerun()
    elif (not all([func(arg) for arg in args])):
        st.session_state.errors[f] = 'A required field was left blank'
        st.rerun()
    if f in st.session_state.errors:
        del st.session_state.errors[f]

def check_error(f):
    if 'errors' in st.session_state:
        if f in st.session_state.errors:
            st.error(st.session_state.errors[f])
    else:
        st.session_state.errors = {}

def simplify_books(books: 'list[str]'):
    if set(BOOKS_ABBREVS[BOOKS_ABBREVS.index('GEN'):BOOKS_ABBREVS.index('MAL')+1]).issubset(books):
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index('GEN'):BOOKS_ABBREVS.index('MAL')+1]:
            books.remove(b)
        books.append('OT')
    if set(BOOKS_ABBREVS[BOOKS_ABBREVS.index('MAT'):BOOKS_ABBREVS.index('REV')+1]).issubset(books):
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index('MAT'):BOOKS_ABBREVS.index('REV')+1]:
            books.remove(b)
        books.append('NT')
    if set(BOOKS_ABBREVS[BOOKS_ABBREVS.index('REV')+1:]).issubset(books):
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index('REV')+1:]:
            books.remove(b)
        books.append('DT')    
    return books

def expand_books(books: 'list[str]'):
    if 'OT' in books:
        books.remove('OT')
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index('GEN'):BOOKS_ABBREVS.index('MAL')+1]:
            books.append(b)
    if 'NT' in books:
        books.remove('NT')
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index('MAT'):BOOKS_ABBREVS.index('REV')+1]:
            books.append(b)
    if 'DT' in books:
        books.remove('DT')
        for b in BOOKS_ABBREVS[BOOKS_ABBREVS.index('REV')+1:]:
            books.append(b)
    return books