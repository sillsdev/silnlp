import zipfile
from io import BytesIO
from time import sleep
from typing import Callable

import streamlit as st
from importlib_resources.abc import Traversable
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

import ethnologue as eg
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


def walk_traversable(
    t: Traversable,
    on_dir: Callable[[Traversable, dict], bool],
    on_file: Callable[[Traversable, dict], bool],
    ctx: dict = {},
):
    """Walk through a traversable. Somewhat similar to the os.walk method, but uses callbacks for dirs and files.
    Initial argument must be a directory. The traversable item (dir or file) that corresponds to each callback as
    well as a copy of the ctx dict are provided to the callbacks. If callback returns False, children are not explored.
    """
    for x in t.iterdir():
        if x.is_dir():
            ctx_c = {k: v for k, v in ctx.items()}
            if on_dir(x, ctx_c):
                walk_traversable(x, on_dir, on_file, ctx_c)
        elif x.is_file():
            on_file(x, ctx)


def copy_resource_trav_to_gdrive(r: zipfile.Path):
    '''Copy a zipfile.Path resource to Google Drive. Path must be a "directory."'''
    if not r.is_dir():
        return
    data_folder = functions.current_data(env=st.session_state.clowder_env)

    def process_dir(t: Traversable, ctx: dict):
        """Process dir callback. Creates the corresponding google drive folder inside of the folder pointed to with ctx['subid']"""
        if "Notes" in t.name or "Print" in t.name:
            return False
        try:
            ctx["subid"] = functions.ENV._create_gdrive_folder(t.name, ctx["subid"])
        except:
            print(f"Failed to copy {t.name} to gdrive")
            sleep(5)
            print(f"Retrying to copy {t.name} to gdrive")
            ctx["subid"] = functions.ENV._create_gdrive_folder(t.name, ctx["subid"])
        return True

    def process_file(t: Traversable, ctx: dict):
        """Process file callback. Writes the corresponding file inside of the folder pointed to with ctx['subid']"""
        if "Notes" in t.name or "Print" in t.name:
            return False
        functions.ENV._write_gdrive_file_in_folder(ctx["subid"], t.name, t.read_bytes().decode("utf-8", "ignore"))

    with functions._lock:
        functions.ENV = st.session_state.clowder_env
        stem: str = str(r).split(".", maxsplit=1)[0] if r.name == "" else r.name
        id = functions.ENV._create_gdrive_folder(stem, data_folder)
        walk_traversable(r, process_dir, process_file, {"subid": id})


def copy_resource_trav_to_s3(r: zipfile.Path):
    '''Copy a zipfile.Path resource to S3 bucket. Path must be a "directory."'''
    if not r.is_dir():
        return

    stem: str = str(r).split(".", maxsplit=1)[0] if r.name == "" else r.name
    parent_path = SIL_NLP_ENV.pt_projects_dir / stem
    parent_path.mkdir(parents=True, exist_ok=True)

    def process_dir(t: Traversable, ctx: dict):
        """Process dir callback. Creates the corresponding S3 "folder" inside of the "folder" pointed to with ctx['parent_path']"""
        if "Notes" in t.name or "Print" in t.name:
            return False
        ctx["parent_path"] = ctx["parent_path"] / t.name
        ctx["parent_path"].mkdir(parents=True, exist_ok=True)
        return True

    def process_file(t: Traversable, ctx: dict):
        """Process file callback. Writes the corresponding file inside of the "folder" pointed to with ctx['parent_path']"""
        if "Notes" in t.name or "Print" in t.name:
            return False
        path = ctx["parent_path"] / t.name
        path.write_text(t.read_bytes().decode())

    walk_traversable(r, process_dir, process_file, {"parent_path": parent_path})


def get_investigations() -> list:
    try:
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
    investigation_tab, resource_tab, explore_tab, settings_tab, debug_tab = st.tabs(
        ["Investigations", "Resources", "Explore Language", "Settings", "Debug"]
    )
else:
    investigation_tab, resource_tab, explore_tab, settings_tab = st.tabs(
        ["Investigations", "Resources", "Explore Language", "Settings"]
    )

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


def glean_resources(z: zipfile.ZipFile):
    """Find nested resources inside of zipfile. Looks for Settings.xml files to
    determine whether a subfolder contains resources (but it doesn't
    support resources nested at different levels).

    Supported:
    - Resources directly in top level
    - Resources in folder in top level
    - Resources in folders in top level
    - Resources in folders contained in a single folder within a zip"""
    path = zipfile.Path(z)
    children = list(path.iterdir())
    if len(children) == 1 and children[0].is_dir():
        path = children[0]
        children = path.iterdir()
    child_dirs: list[zipfile.Path] = []
    for child in children:
        if child.is_dir():
            child_dirs.append(child)
        if child.name == "Settings.xml":
            return [path]
    result = []
    for path in child_dirs:
        for child in path.iterdir():
            if child.name == "Settings.xml":
                result.append(path)
    return result


def add_resource_internal(resource: zipfile.Path):
    """Do the processing required for adding resource to internal S3 bucket"""
    copy_resource_trav_to_s3(resource)
    project = str(resource).split(".", maxsplit=1)[0] if resource.name == "" else resource.name
    command = f'{os.environ.get("PYTHON", "python")} -m silnlp.common.extract_corpora {project}'
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
                    if "is_internal_user" in st.session_state and st.session_state.is_internal_user:
                        for zp in resources:
                            with zipfile.ZipFile(zp) as f:
                                for resource in glean_resources(f):
                                    add_resource_internal(resource)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        for zp in resources:
                            with zipfile.ZipFile(zp) as f:
                                for resource in glean_resources(f):
                                    copy_resource_trav_to_gdrive(resource)
                        try:
                            functions.use_data(functions.current_data(env=st.session_state.clowder_env))
                            set_success("add_resource", "Resource(s) successfully uploaded!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Something went wrong while adding resource data. Please try again. Error: {e}")
            check_success("add_resource")


def write_dict_columns(d: dict):
    c1, c2 = st.columns([0.5, 1])
    with c1:
        for key in d.keys():
            st.write(f"**{key}**")
    with c2:
        for val in d.values():
            st.write(val)


if "lang_code_btn_counter" not in st.session_state:
    st.session_state.lang_code_btn_counter = 0


def write_lang_codes(l: list, num_columns=5):
    columns = st.columns(num_columns)
    i = 0

    def callback(my_code):
        def _():
            st.session_state.language_code = my_code

        return _

    lang_code_button_counter = st.session_state.lang_code_btn_counter

    for column in columns:
        with column:
            for el in range(i, len(l), num_columns):
                st.button(f"`{l[el]}`", on_click=callback(l[el]), key=f"{i*7}{el*5}{lang_code_button_counter}{l[el]}")
        i += 1
        lang_code_button_counter += 1
        lang_code_button_counter %= 1000

    st.session_state.lang_code_btn_counter = lang_code_button_counter


with explore_tab:
    if "language_code" not in st.session_state:
        st.session_state.language_code = None
    with st.container(border=True):
        st.text_input("Language Code to Explore", key="language_code")
    if st.session_state.language_code:
        language_code = st.session_state.language_code
        with st.container(border=True):
            st.header(f"Language `{language_code}`")
            st.write(f"**Language Info (`{language_code}`):**")
            with st.container(border=True):
                write_dict_columns(eg.lang_info_dict(language_code))
            st.subheader(f"Explore `{language_code}` Classifications:")
            clasfs = eg.lang_classifications(language_code)
            clas = st.selectbox("Choose classification", clasfs)
            with st.expander(f"**Other languages with classification *{clas}***"):
                write_lang_codes(eg.find_class_langs(clas), 8)
            fam_c = eg.lang_country_family(language_code)
            st.subheader(f"`{language_code}` Language Family")
            st.write(fam_c)
            with st.expander(f"**Languages in Same Family (*{fam_c}*)**"):
                write_lang_codes(eg.find_country_family_langs(fam_c))
            st.subheader(f"Explore `{language_code}` Countries:")
            countries = {x["CountryCode"]: x for x in eg.lang_country_dicts(language_code)}
            country = st.selectbox("Choose country", countries.keys())
            st.write(f"**Language Country Info (`{language_code}`-`{country}`):**")
            with st.container(border=True):
                write_dict_columns(countries[country])

    with st.expander("DataFrames"):
        if st.button("Load Ethnolog DFs"):
            l_df = eg.language_df()
            lic_df = eg.language_in_country_df()
            licad_df = eg.language_in_country_add_data_df()
            lad_df = eg.language_add_data_df()
            lan_df = eg.language_alt_name_df()
            with st.popover("Language"):
                st.dataframe(l_df)
            with st.popover("In Country"):
                st.dataframe(lic_df)
            with st.popover("In Country AD"):
                st.dataframe(licad_df)
            with st.popover("AD"):
                st.dataframe(lad_df)
            with st.popover("Alternate Name"):
                st.dataframe(lan_df)

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
