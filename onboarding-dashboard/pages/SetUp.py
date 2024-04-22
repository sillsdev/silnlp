import streamlit as st
from pydrive2.auth import GoogleAuth
import json
import sys
import os
parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from clowder import consts, functions
from utils import check_error, check_required

st.set_page_config(
    page_title="OnboardingDashboard",
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
gauth = GoogleAuth()
consts.get_env(gauth)
st.header('Set Up')
with st.form(key='set_up_form') as f:
    root = st.text_input('Link to investigations root folder', placeholder='https://drive.google.com/drive/u/0/folders/0000000000000000000')
    data_folder = st.text_input('Link to data folder',placeholder='https://drive.google.com/drive/u/0/folders/0000000000000000000')
    check_error('set_up')
    if st.form_submit_button('Set Up', type='primary'):
        check_required('set_up', root, data_folder, func=(lambda p: p is not None and p != '' and 'folders/' in p))
        functions.use_context(root.split('folders/')[1])
        functions.track(None)
        functions.use_data(data_folder.split('folders/')[1])
        st.session_state.set_up = True
        st.switch_page('Home.py')

if st.button('BYPASS FOR TESTING'):
    st.session_state.set_up = True
    st.switch_page('Home.py')