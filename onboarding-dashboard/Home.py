import zipfile
from io import BytesIO
from time import sleep

import streamlit as st
from s3path import S3Path

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

if "set_up" not in st.session_state or not st.session_state.set_up:
    st.switch_page("pages/LogIn.py")

if st.session_state.google_auth is not None and st.session_state.google_auth.access_token_expired:
    # st.session_state.google_auth.Refresh() TODO
    st.session_state.set_up = False
    del st.session_state.google_auth
    st.switch_page("pages/LogIn.py")

import os
import subprocess
import sys

from models import Investigation

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

import pandas as pd
from utils import check_error, check_required, check_success, set_success

from clowder import functions
from clowder.environment import DuplicateInvestigationException
from silnlp.common.environment import SIL_NLP_ENV


def copy_resource_to_gdrive(r: BytesIO):
    if not zipfile.is_zipfile(r):
        return
    data_folder = functions.current_data(env=st.session_state.clowder_env)
    with functions._lock:
        functions.ENV = st.session_state.clowder_env
        id = functions.ENV._create_gdrive_folder(r.name[:-4], data_folder)
        with zipfile.ZipFile(r) as f:
            for file in f.filelist:
                if "Notes" in file.filename or "Print" in file.filename:
                    continue
                subid = id
                print(f"Copying {file.filename} to gdrive...")
                if "/" in file.filename:
                    path_parts = file.filename.split("/")
                    for part in path_parts:
                        try:
                            subid = functions.ENV._create_gdrive_folder(part, subid)
                        except:
                            print(f"Failed to copy {part} to gdrive")
                            sleep(5)
                            print(f"Retrying to copy {part} to gdrive")
                            subid = functions.ENV._create_gdrive_folder(part, subid)
                functions.ENV._write_gdrive_file_in_folder(
                    subid, file.filename.split("/")[-1], f.read(file).decode("utf-8", "ignore")
                )


def copy_resource_to_s3(r: BytesIO):
    if not zipfile.is_zipfile(r):
        return
    with zipfile.ZipFile(r) as f:
        for file in f.filelist:
            if "Notes" in file.filename or "Print" in file.filename:
                continue
            path = SIL_NLP_ENV.pt_projects_dir / f.filename[:-4] / file.filename
            print(f"Copying {file.filename} to s3 path {path}...")
            if not isinstance(path, S3Path) and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f.read(file).decode())


def get_investigations() -> list:
    try:
        functions.sync(investigation_name=None, env=st.session_state.clowder_env)
        return list(map(lambda i: Investigation.from_clowder(i), functions.list_inv(env=st.session_state.clowder_env)))
    except Exception as e:
        import traceback

        traceback.print_exc()
        st.error(f"Something went wrong while fetching investigation data. Please try again. Error: {e}")
        return []


@st.cache_data(show_spinner=False)
def get_resources(env):  # Pass in env to make cache per env
    try:
        fn_res = functions.list_resources(env=st.session_state.clowder_env)
        return list(map(lambda fn: fn[:-4], fn_res))
    except Exception as e:
        import traceback

        traceback.print_exc()
        st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")
        return []


if "investigations" not in st.session_state:
    st.session_state.investigations = get_investigations()

if os.environ.get("DEBUG_SFONBOARD", None) == "true":
    investigation_tab, resource_tab, settings_tab, debug_tab = st.tabs(
        ["Investigations", "Resources", "Settings", "Debug"]
    )
else:
    investigation_tab, resource_tab, settings_tab = st.tabs(["Investigations", "Resources", "Settings"])

with investigation_tab:
    st.header("Investigations")
    for investigation in st.session_state.investigations:
        investigation: Investigation
        c = st.container()
        investigation.to_st(container=c)

    with st.form(key="new_investigation", clear_on_submit=True):
        st.write("### Create a new investigation ###")
        check_error("create_investigation")
        name = st.text_input("Name")
        if st.form_submit_button(type="primary"):
            check_required("create_investigation", name)
            try:
                with st.spinner("This may take a few minutes..."):
                    try:
                        functions.create(name, env=st.session_state.clowder_env)
                        st.session_state.investigations = get_investigations()
                    except Exception as e:
                        st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")
                        raise
            except DuplicateInvestigationException:
                st.session_state.error = "An investigation with that name already exists"
                st.rerun()
            set_success("create_investigation", "Investigation successfully created!")
            st.rerun()
        check_success("create_investigation")

with resource_tab:
    st.header("Resources")
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        res = get_resources(str(st.session_state.clowder_env))
        data = {"Resource": res}
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=500)
    with c2:
        with st.form(key=f"add_resource", clear_on_submit=True):
            resources = st.file_uploader(
                "Resource",
                type="zip",
                accept_multiple_files=True,
                help="A Paratext project backup zip - i.e., a zip file with name <project_name>.zip with the following contents: \
                \n-- Book.usfm \
                \n--  AnotherBook.usfm \
                \n-- ... \
                \n-- Settings.xml \
                \nUpon success, a new text will be added to your resources with name <language_code>-<project_name> which can then be used in your investigations",
            )
            check_error("add_resource")
            if st.form_submit_button(
                "Add",
                type="primary",
            ):
                check_required("add_resource", resources)
                with st.spinner("This might take a few minutes..."):
                    if st.session_state.is_internal_user:
                        for resource in resources:
                            with zipfile.ZipFile(resource) as f:
                                copy_resource_to_s3(resource)
                                project = f.filename[:-4]
                                command = (
                                    f'{os.environ.get("PYTHON", "python")} -m silnlp.common.extract_corpora {project}'
                                )
                                print(f"Running {command}")
                                result = subprocess.run(
                                    command,
                                    shell=True,
                                    capture_output=True,
                                    text=True,
                                )
                                print("Result", result)
                                if result.stdout != "":
                                    print(result.stdout)
                                if result.returncode != 0:
                                    print(result.stderr)
                                    st.error(f"Something went wrong while adding resource data. Please try again.")
                                else:
                                    set_success("add_resource", "Resource(s) successfully uploaded!")
                                SIL_NLP_ENV.copy_pt_project_to_bucket(project)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        for resource in resources:
                            with zipfile.ZipFile(resource) as f:
                                copy_resource_to_gdrive(resource)
                        try:
                            functions.use_data(functions.current_data(env=st.session_state.clowder_env))
                            set_success("add_resource", "Resource(s) successfully uploaded!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Something went wrong while adding resource data. Please try again. Error: {e}")
            check_success("add_resource")

with settings_tab:
    st.header("Change Set Up")
    with st.form(key="set_up_form") as f:
        root = st.text_input(
            "Link to investigations root folder",
            placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
            value=f"https://drive.google.com/drive/u/0/folders/{functions.current_context(env=st.session_state.clowder_env)}",
        )
        data_folder = None
        if "is_internal_user" in st.session_state and not st.session_state.is_internal_user:
            data_folder = st.text_input(
                "Link to data folder",
                placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
                value=(
                    f"https://drive.google.com/drive/u/0/folders/{functions.current_data(env=st.session_state.clowder_env)}"
                    if functions.current_data(env=st.session_state.clowder_env) is not None
                    else ""
                ),
            )
        check_error("set_up")
        if st.form_submit_button("Save Changes", type="primary"):
            if data_folder is not None:
                check_required(
                    "set_up", root, data_folder, func=(lambda p: p is not None and p != "" and "folders/" in p)
                )
            else:
                check_required("set_up", root, func=(lambda p: p is not None and p != "" and "folders/" in p))
            with st.spinner("This might take a few minutes..."):
                try:
                    from clowder.environment import ClowderEnvironment

                    st.session_state.clowder_env = ClowderEnvironment(
                        auth=st.session_state.google_auth, context=root.split("folders/")[1]
                    )
                    if len(functions.list_inv(env=st.session_state.clowder_env)) == 0:
                        functions.track(None, env=st.session_state.clowder_env)
                    if data_folder is not None:
                        print(f"Using data folder {data_folder}")
                        functions.use_data(
                            data_folder.split("folders/")[1], env=st.session_state.clowder_env, refresh=True
                        )
                    set_success("set_up", "Successfully changed set-up!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")
            check_success("set_up")

    if os.environ.get("DEBUG_SFONBOARD", None) == "true":
        with debug_tab:
            st.write(
                st.session_state.clowder_env,
                st.session_state.clowder_env.meta.data,
                st.session_state.clowder_env.data_folder,
            )
