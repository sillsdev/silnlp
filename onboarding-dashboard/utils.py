import streamlit as st

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