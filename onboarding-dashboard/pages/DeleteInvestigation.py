import streamlit as st
import sys
import os
from pathlib import Path
parent_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(parent_dir.parent))
from clowder import functions
if 'investigation_to_delete' in st.session_state:
    st.write(f"Are you sure you want to delete {st.session_state.investigation_to_delete.name}?")
    if st.button("Yes"):
        try:
            functions.delete(st.session_state.investigation_to_delete.name)
            st.session_state.investigations.remove(st.session_state.investigation_to_delete)
            del st.session_state.investigation_to_delete
            st.switch_page("Investigations.py")
        except Exception as e:
            print(e)
            st.error(e)
    elif st.button("Cancel"):
        del st.session_state.investigation_to_delete
        st.switch_page("Investigations.py")
else:
    st.switch_page("Investigations.py")

        