import streamlit as st
from models import Investigation, Status
import sys
import os
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)
from clowder import functions
from clowder.environment import DuplicateInvestigationException

st.set_page_config(
    page_title="OnboardingDashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

def get_investigations() -> list:
    return list(map(lambda i: Investigation.from_clowder(i), functions.list_inv()))

if 'investigations' not in st.session_state:
    st.session_state.investigations = get_investigations()

for investigation in st.session_state.investigations:
    investigation: Investigation
    c = st.container()
    investigation.to_st(container=c)

with st.form(key='new_investigation', clear_on_submit=True):
    st.header("Create a new investigation")
    if 'error' in st.session_state:
        st.error(st.session_state.error)
    name = st.text_input("Name")
    if st.form_submit_button():
        if name is None or len(name) == 0:
            st.session_state.error = "Please fill in all required fields"
            st.rerun()
        try:
            with st.spinner("This may take a few minutes..."):
                functions.create(name) #TODO create-from-template
                st.session_state.investigations = get_investigations()
        except DuplicateInvestigationException:
            st.session_state.error = "An investigation with that name already exists"
            st.rerun()
        if 'error' in st.session_state:
            del st.session_state.error
        st.rerun()
        
