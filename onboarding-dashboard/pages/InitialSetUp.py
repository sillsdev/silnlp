import os
import sys

import boto3
import streamlit as st
from streamlit_cookies_controller import CookieController

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from utils import check_error, check_required

st.set_page_config(page_title="OnboardingDashboard", initial_sidebar_state="collapsed")

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

if "cookie_controller" not in st.session_state:
    st.session_state.cookie_controller = CookieController()

if ("set_up" in st.session_state and st.session_state.set_up) or "google_auth" not in st.session_state:
    st.switch_page("Home.py")

bypass_auth = False
if os.environ.get("BYPASS_AUTH", "").lower() == "true":
    bypass_auth = st.session_state.auth_method == "BYPASS"

st.header("Set Up")
with st.form(key="set_up_form") as f:
    current_root = None
    try:
        current_root = st.session_state.cookie_controller.get("root")
    except:
        pass
    current_data_folder = None
    try:
        current_data_folder = st.session_state.cookie_controller.get("data_folder")
    except:
        pass
    root = st.text_input(
        "Link to investigations root folder",
        placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
        value=current_root,
    )
    is_external_user = not st.session_state.is_internal_user
    print("Is external user?", is_external_user)
    if not bypass_auth:
        if is_external_user:
            data_folder = st.text_input(
                "Link to data folder",
                placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
                value=current_data_folder,
            )
            refresh = st.checkbox(
                "Refresh resources",
                help="Check this box if new resources have been manually added to your resource gdrive",
            )
    check_error("set_up")
    if st.form_submit_button("Set Up", type="primary"):
        from clowder import functions

        if not bypass_auth and is_external_user:
            check_required("set_up", root, data_folder, func=(lambda p: p is not None and p != "" and "folders/" in p))
        else:
            check_required("set_up", root, func=(lambda p: p is not None and p != "" and "folders/" in p))
        with st.spinner("This might take a few minutes..."):
            from clowder.environment import ClowderEnvironment

            if bypass_auth or st.session_state.google_auth is None:
                st.session_state.clowder_env = ClowderEnvironment(context=root.split("folders/")[1].split("?")[0])
            else:
                st.session_state.clowder_env = ClowderEnvironment(
                    auth=st.session_state.google_auth, context=root.split("folders/")[1].split("?")[0]
                )
            functions.ENV = st.session_state.clowder_env
            boto3.resource("s3")  # start s3 connection during setup
            if len(functions.list_inv(env=st.session_state.clowder_env)) == 0:
                functions.track(None, env=st.session_state.clowder_env)
            if not bypass_auth and is_external_user:
                functions.use_data(data_folder.split("folders/")[1].split("?")[0], refresh)
            else:
                functions.unlink_data(env=st.session_state.clowder_env)
            try:
                st.session_state.cookie_controller.set("root", root)
                if not bypass_auth:
                    st.session_state.cookie_controller.set("data_folder", data_folder)
            except:
                pass
            st.session_state.set_up = True
            st.switch_page("Home.py")
