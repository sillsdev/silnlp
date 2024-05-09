import os
import sys

import google_auth_oauthlib.flow
import streamlit as st
from googleapiclient.discovery import build
from pydrive2.auth import GoogleAuth

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from utils import check_error, check_required

from clowder import consts

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

if not os.path.exists("client_secrets.json"):
    with open("client_secrets.json", "w") as f:
        f.write(os.environ.get("GOOGLE_CLIENT_SECRET", ""))

def auth_flow():
    gauth = GoogleAuth()
    auth_code = st.query_params.get("code")
    redirect_uri = os.environ.get("REDIRECT_URL", "http://localhost:8501/LogIn")
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        "client_secrets.json",
        scopes=["https://www.googleapis.com/auth/userinfo.email", "openid", "https://www.googleapis.com/auth/drive"],
        redirect_uri=redirect_uri,
    )
    if auth_code and ('google_auth_code' not in st.session_state or auth_code != st.session_state.google_auth_code):
        gauth.GetFlow()
        gauth.flow.redirect_uri = redirect_uri
        gauth.Authenticate(auth_code)
        user_info_service = build(
            serviceName="oauth2",
            version="v2",
            credentials=gauth.credentials,
        )
        user_info = user_info_service.userinfo().get().execute()
        assert user_info.get("email"), "Email not found in infos"
        st.session_state.google_auth_code = auth_code
        st.session_state.user_info = user_info
        consts.get_env(gauth)
        st.rerun()
    else:
        _, c2, _ = st.columns(3)
        with c2:
            st.title("Welcome")
            authorization_url, _ = flow.authorization_url(
                access_type="offline",
                include_granted_scopes="true",
            )
            with st.container(border=True):
                st.page_link(page=authorization_url, label="Sign in with Google")


if "google_auth_code" not in st.session_state:
    auth_flow()
else:
    internal_emails = os.environ.get("ONBOARDING_INTERNAL_USER_EMAILS", None)
    if internal_emails is None:
        internal_emails = []
    else:
        internal_emails = internal_emails.split(";")

    external_emails = os.environ.get("ONBOARDING_EXTERNAL_USER_EMAILS", None)
    if external_emails is None:
        external_emails = []
    else:
        external_emails = external_emails.split(";")

    is_internal_user = st.session_state.user_info["email"] in internal_emails
    is_external_user = st.session_state.user_info["email"] in external_emails
    is_allowed_user = is_internal_user or is_external_user
    if is_allowed_user:
        st.header("Set Up")
        st.write("THIS",os.environ.get('AWS_ACCESS_KEY_ID'))
        with st.form(key="set_up_form") as f:
            root = st.text_input(
                "Link to investigations root folder",
                placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000",
            )
            if is_external_user:
                data_folder = st.text_input(
                    "Link to data folder", placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000"
                )
            check_error("set_up")
            if st.form_submit_button("Set Up", type="primary"):
                from clowder import functions

                if is_external_user:
                    check_required(
                        "set_up", root, data_folder, func=(lambda p: p is not None and p != "" and "folders/" in p)
                    )
                else:
                    check_required("set_up", root, func=(lambda p: p is not None and p != "" and "folders/" in p))
                with st.spinner("This might take a few minutes..."):
                    # try:
                    functions.use_context(root.split("folders/")[1])
                    if is_external_user:
                        functions.track(None)
                        functions.use_data(data_folder.split("folders/")[1])
                    else:
                        functions.track(None)
                        functions.unlink_data()
                    st.session_state.set_up = True
                    st.switch_page("Home.py")
                # except Exception as e:
                #     st.error(f"Something went wrong while attempting to run experiment. Please try again. Error: {e}")

    else:
        st.title("You do not have access to this application.")
