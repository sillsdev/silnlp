import os
import sys
from pathlib import Path

import streamlit as st

parent_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(parent_dir.parent))
from clowder import functions

if "investigation_to_delete" in st.session_state:
    st.write(
        f'<div style="text-align: center"> Are you sure you want to delete {st.session_state.investigation_to_delete.name}? </div>',
        unsafe_allow_html=True,
    )
    _, c1, c2, _ = st.columns([3, 1, 1, 3])
    with c1:
        if st.button("Yes"):
            try:
                functions.delete(st.session_state.investigation_to_delete.name)
                st.session_state.investigations.remove(st.session_state.investigation_to_delete)
                del st.session_state.investigation_to_delete
                st.switch_page("Home.py")
            except Exception as e:
                st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")

    with c2:
        if st.button("Cancel"):
            del st.session_state.investigation_to_delete
            st.switch_page("Home.py")
else:
    st.switch_page("Home.py")
