import zipfile
from io import BytesIO

import streamlit as st

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

if "set_up" not in st.session_state or st.session_state.set_up == False:
    st.switch_page("pages/LogIn.py")

import os
import sys

from models import Investigation

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)
from utils import check_error, check_required

from clowder import functions
from clowder.environment import DuplicateInvestigationException


def copy_resource_to_gdrive(r: BytesIO):
    if not zipfile.is_zipfile(r):
        return
    id = functions.ENV._create_gdrive_folder(r.name[:-4], functions.current_data())
    with zipfile.ZipFile(r) as f:
        for file in f.filelist:
            if "Notes" in file.filename:
                continue
            subid = id
            if "/" in file.filename:
                path_parts = file.filename.split("/")
                for part in path_parts:
                    subid = functions.ENV._create_gdrive_folder(part, subid)
            functions.ENV._write_gdrive_file_in_folder(subid, file.filename.split("/")[-1], f.read(file).decode())


def get_investigations() -> list:
    try:
        return list(map(lambda i: Investigation.from_clowder(i), functions.list_inv()))
    except Exception as e:
        st.error(f"Something went wrong while fetching investigation data. Please try again. Error: {e}")
        return []


@st.cache_data(show_spinner=False)
def get_resources():
    try:
        return list(map(lambda fn: fn[:-4], functions.list_resources()))
    except Exception as e:
        st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")
        return []


if "investigations" not in st.session_state:
    st.session_state.investigations = get_investigations()

investigation_tab, resource_tab, settings_tab, debug_tab = st.tabs(["Investigations", "Resources", "Settings", "Debug"])

with investigation_tab:
    st.header("Investigations")
    for investigation in st.session_state.investigations:
        investigation: Investigation
        c = st.container()
        investigation.to_st(container=c)

    with st.form(key="new_investigation", clear_on_submit=True):
        st.write("### Create a new investigation ###")
        if "error" in st.session_state:
            st.error(st.session_state.error)
        name = st.text_input("Name")
        if st.form_submit_button(type="primary"):
            if name is None or len(name) == 0:
                st.session_state.error = "Please fill in all required fields"
                st.rerun()
            try:
                with st.spinner("This may take a few minutes..."):
                    try:
                        functions.create(name)
                        st.session_state.investigations = get_investigations()
                    except Exception as e:
                        st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")
                        raise
            except DuplicateInvestigationException:
                st.session_state.error = "An investigation with that name already exists"
                st.rerun()
            if "error" in st.session_state:
                del st.session_state.error
            st.rerun()

with resource_tab:
    st.header("Resources")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(height=500):
            for resource in get_resources():
                with st.container(border=True):
                    st.write(resource)
    with c2:
        with st.form(key=f"add_resource"):
            resources = st.file_uploader("Resource", type="zip", accept_multiple_files=True)
            check_error("add_resource")
            if st.form_submit_button("Add", type="primary"):
                check_required("add_resource", resources)
                with st.spinner("This might take a few minutes..."):
                    for resource in resources:
                        with zipfile.ZipFile(resource) as f:
                            print("Copying", f.filename)
                            copy_resource_to_gdrive(resource)
                    try:
                        functions.use_data(functions.current_data())
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Something went wrong while adding resource data. Please try again. Error: {e}")

    with settings_tab:
        st.header("Change Set Up")
        with st.form(key="set_up_form") as f:
            root = st.text_input(
                "Link to investigations root folder",
                placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
                value=f"https://drive.google.com/drive/u/0/folders/{functions.current_context()}",
            )
            data_folder = st.text_input(
                "Link to data folder",
                placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
                value=f"https://drive.google.com/drive/u/0/folders/{functions.current_data()}"
                if functions.current_data() is not None
                else "",
            )
            check_error("set_up")
            if st.form_submit_button("Save Changes", type="primary"):
                check_required("set_up", root, func=(lambda p: p is not None and p != "" and "folders/" in p))
                with st.spinner("This might take a few minutes..."):
                    try:
                        functions.use_context(root.split("folders/")[1])
                        functions.track(None)
                        if data_folder is not None:
                            functions.use_data(data_folder.split("folders/")[1])
                    except Exception as e:
                        st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")
    
    with debug_tab:
        st.write(functions.ENV.meta.data)