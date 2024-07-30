import os
import sys

import boto3
import google_auth_oauthlib.flow
import streamlit as st
from googleapiclient.discovery import build
from pydrive2.auth import GoogleAuth
from streamlit_cookies_controller import CookieController

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from utils import check_error, check_required

bypass_auth = False
if os.environ.get("BYPASS_AUTH", "").lower() == "true":
    bypass_auth = True

if not os.path.exists(os.environ.get("SIL_NLP_CACHE_EXPERIMENT_DIR", "~/.cache/silnlp")):
    os.makedirs(os.environ.get("SIL_NLP_CACHE_EXPERIMENT_DIR", "~/.cache/silnlp"))

from clowder import consts

consts.set_up_creds()

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

cookie_controller = CookieController()

if not bypass_auth:
    if not os.path.exists("client_secrets.json"):
        with open("client_secrets.json", "w") as f:
            f.write(os.environ.get("GOOGLE_CLIENT_SECRET", ""))


def auth_flow():
    if bypass_auth:
        st.session_state.google_auth = None
    else:
        gauth = GoogleAuth()
        gauth.settings["get_refresh_token"] = True
        auth_code = st.query_params.get("code")
        redirect_uri = os.environ.get("REDIRECT_URL", "http://localhost:8501/LogIn")
        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            "client_secrets.json",
            scopes=[
                "https://www.googleapis.com/auth/userinfo.email",
                "openid",
                "https://www.googleapis.com/auth/drive",
            ],
            redirect_uri=redirect_uri,
        )
        if auth_code:
            gauth.GetFlow()
            gauth.flow.redirect_uri = redirect_uri
            try:
                gauth.Authenticate(auth_code)
            except Exception as e:
                st.error(f"Something went wrong while authenticating. Please try again. Error: {e}")
                st.query_params.clear()
                st.page_link("pages/LogIn.py", label="Try again")
                return
            user_info_service = build(
                serviceName="oauth2",
                version="v2",
                credentials=gauth.credentials,
            )
            user_info = user_info_service.userinfo().get().execute()
            st.session_state.google_auth = gauth
            st.session_state.user_info = user_info
            st.rerun()
        else:
            st.title("Welcome")
            authorization_url, _ = flow.authorization_url(
                access_type="offline",
                include_granted_scopes="true",
            )
            st.page_link(page=authorization_url, label="Sign in with Google")


if "google_auth" not in st.session_state:
    auth_flow()
else:
    if bypass_auth:
        is_allowed_user = True
        st.session_state.is_internal_user = True
    else:
        internal_emails = os.environ.get("ONBOARDING_INTERNAL_USER_EMAILS", None)
        if internal_emails is None:
            internal_emails = []
        else:
            internal_emails = internal_emails.upper().split(";")

        external_emails = os.environ.get("ONBOARDING_EXTERNAL_USER_EMAILS", None)
        if external_emails is None:
            external_emails = []
        else:
            external_emails = external_emails.upper().split(";")

        is_internal_user = st.session_state.user_info.get("email", "X" * 100).upper() in internal_emails
        is_external_user = st.session_state.user_info.get("email", "X" * 100).upper() in external_emails
        st.session_state.is_internal_user = is_internal_user
        is_allowed_user = is_internal_user or is_external_user
    if is_allowed_user:
        st.header("Set Up")
        with st.form(key="set_up_form") as f:
            current_root = None
            try:
                current_root = cookie_controller.get("root")
            except:
                pass
            current_data_folder = None
            try:
                current_data_folder = cookie_controller.get("data_folder")
            except:
                pass
            root = st.text_input(
                "Link to investigations root folder",
                placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
                value=current_root,
            )
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
                    check_required(
                        "set_up", root, data_folder, func=(lambda p: p is not None and p != "" and "folders/" in p)
                    )
                else:
                    check_required("set_up", root, func=(lambda p: p is not None and p != "" and "folders/" in p))
                with st.spinner("This might take a few minutes..."):
                    from clowder.environment import ClowderEnvironment

                    if bypass_auth:
                        st.session_state.clowder_env = ClowderEnvironment(
                            context=root.split("folders/")[1].split("?")[0]
                        )
                    else:
                        st.session_state.clowder_env = ClowderEnvironment(
                            auth=st.session_state.google_auth, context=root.split("folders/")[1].split("?")[0]
                        )
                    functions.ENV = st.session_state.clowder_env
                    boto3.resource("s3")
                    if len(functions.list_inv(env=st.session_state.clowder_env)) == 0:
                        functions.track(None, env=st.session_state.clowder_env)
                    if not bypass_auth and is_external_user:
                        functions.use_data(data_folder.split("folders/")[1].split("?")[0], refresh)
                    else:
                        functions.unlink_data(env=st.session_state.clowder_env)
                    try:
                        cookie_controller.set("root", root)
                        if not bypass_auth:
                            cookie_controller.set("data_folder", data_folder)
                    except:
                        pass
                    st.session_state.set_up = True
                    st.switch_page("Home.py")

    else:
        st.title("You do not have access to this application.")
