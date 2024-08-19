import os
import sys
from threading import Lock

import google_auth_oauthlib.flow
import streamlit as st
from googleapiclient.discovery import build
from pydrive2.auth import GoogleAuth
from pydrive2.files import ApiRequestError

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

if os.environ.get("DOWN_FOR_MAINTENANCE"):
    st.switch_page("pages/Down.py")

AUTH_OPTIONS = ["FULL", "RESTRICTED", "SERVICE"]
DEFAULT_AUTH = "FULL"

bypass_auth = False
if os.environ.get("BYPASS_AUTH", "").lower() == "true":
    AUTH_OPTIONS.insert(0, "BYPASS")
    DEFAULT_AUTH = "BYPASS"
    bypass_auth = True

import pickle

from utils import decrypt, encrypt

if st.query_params.get("state") is not None:
    try:
        enc_state = st.query_params.get("state")
        dec_state = decrypt(enc_state)
        state = pickle.loads(dec_state)
        if not state["FROM_O-D"]:
            raise Exception("Request came from different source...")
        st.session_state.auth_method = state["AUTH_METHOD"]
        print("Changed Auth Method to:", st.session_state.auth_method)
    except:
        st.query_params.clear()


if "auth_method" not in st.session_state:
    st.session_state.auth_method = DEFAULT_AUTH

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

if "auth_lock" not in st.session_state:
    st.session_state.auth_lock = Lock()

if not bypass_auth:
    if not os.path.exists("client_secrets.json"):
        with open("client_secrets.json", "w") as f:
            f.write(os.environ.get("GOOGLE_CLIENT_SECRET", ""))


def auth_flow_bypass(state=None):
    st.write("Bypassing Authorization...")
    if st.button("Continue To Setup"):
        st.session_state.google_auth = None
        st.rerun()


def auth_flow_restricted(state):
    gauth = GoogleAuth()
    gauth.settings["get_refresh_token"] = True
    gauth.settings["oauth_scope"] = ["https://www.googleapis.com/auth/drive.file"]
    # print(st.query_params)
    auth_code = st.query_params.get("code")
    st.query_params.clear()
    redirect_uri = os.environ.get("REDIRECT_URL", "http://localhost:8501/LogIn")
    # print("Flowing Auth: ", redirect_uri)
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        "client_secrets.json",
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "openid",
            "https://www.googleapis.com/auth/drive.file",
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
        st.write("SIL uses your Google account to manage authorization for the onboarding dashboard.")
        st.write(
            "At this authorization level, the onboarding dashboard will have access to modify or view the files within the folders that you give us in the pages following. If you choose this permission level, the onboarding dashboard will have the ability to access only the files owned by you in these folders. If you are the sole user of this tool in your organization, or you are an internal user, this permission level should work for you. If you would like to modify the permission level, toggle **Modify Google Drive Access** above to the on position, then move the slider below."
        )
        st.write("Otherwise ...")
        authorization_url, _ = flow.authorization_url(
            access_type="offline", include_granted_scopes="true", prompt="consent", state=state
        )
        st.link_button(url=authorization_url, label="Sign in with Google", type="primary")


def auth_flow_full(state):
    gauth = GoogleAuth()
    gauth.settings["get_refresh_token"] = True
    # print(st.query_params)
    auth_code = st.query_params.get("code")
    st.query_params.clear()
    redirect_uri = os.environ.get("REDIRECT_URL", "http://localhost:8501/LogIn")
    # print("Flowing Auth: ", redirect_uri)
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
        st.write("SIL uses your Google account to manage authorization for the onboarding dashboard.")
        st.write(
            "At this authorization level, the onboarding dashboard will have full access to all files in your Google drive. We will only modify or view the files within the folders that you give us in the pages following, but the onboarding dashboard will have the ability to access any file in your Google Drive. If you would like to modify the permission level, toggle **Modify Google Drive Access** above to the on position, then move the slider below."
        )
        st.write("Otherwise ...")
        authorization_url, _ = flow.authorization_url(
            access_type="offline", include_granted_scopes="true", prompt="consent", state=state
        )
        st.link_button(url=authorization_url, label="Sign in with Google", type="primary")


def auth_flow_service(state):

    @st.experimental_dialog("How to Give Access", width="large")
    def _access():
        st.write(
            """
        # Giving Access to Our Service Account
        > *Skip to [Checker](#checker)*
        1. Open the Google Drive web app and choose a folder.

        <img alt="Choose Folder Example" src="https://i.imgur.com/TeUzle8.png" width="300" />

        2. Right click the folder and open the **Share** menu

        <img alt="Right Click Example" src="https://i.imgur.com/xt4K2Fp.png" width="300"  />
        
        3. Paste the following email: `clowder@clowder-400318.iam.gserviceaccount.com` into the **Add People** section and confirm.
        
        <img alt="Paste Example" src="https://i.imgur.com/84blWPM.png" width="300"  />
        
        4. Give Clowder **Editor** permissions.

        <img alt="Permissions Example" src="https://i.imgur.com/N1xwqJr.png" width="300"  />

        # Check If We Have Access to A Folder <a name="checker"></a>
        """,
            unsafe_allow_html=True,
        )
        folder_url = st.text_input(
            "Folder Url", placeholder="https://drive.google.com/drive/u/0/folders/0000000000000000000"
        )
        if folder_url:
            try:
                from clowder.environment import ClowderEnvironment

                clowder_env = ClowderEnvironment()
                folder_id = folder_url.split("folders/")[1].split("?")[0]
                folder_name = "unknown"
                files_dict = {}
                with st.spinner("Checking metadata access"):
                    file_object = clowder_env._google_drive.CreateFile({"id": folder_id})
                    folder_name = file_object["title"]
                with st.spinner("Checking read access to folder"):
                    files_dict = clowder_env._dict_of_gdrive_files(folder_id)
                with st.spinner("Checking write access to folder"):
                    child_file_id = clowder_env._write_gdrive_file_in_folder(
                        folder_id, "TEST_CLOWDER_ACCESS", "TESTING!"
                    )
                    clowder_env._delete_gdrive_folder(child_file_id)
            except IndexError:
                st.error("Bad Google Drive folder url. Copy the url of the google drive folder after opening it.")
            except ApiRequestError as are:
                reason = are.GetField("reason")
                if reason == "notFound":
                    st.error(
                        "Could not find a Google Drive folder with this URL. Either there's a typo in the URL or it hasn't been shared with the service account."
                    )
                elif reason == "forbidden":
                    st.error(
                        "Did not get sufficient access to folder. Did you give the service account editor privileges?"
                    )
                elif reason == "invalid":
                    st.error("Was not able to do folder operations. Are you sure you linked to a folder?")
                else:
                    st.error(f"A strange error was encountered. Send the following to our support email. `{are}`")
            else:
                st.write("Success!")
                st.write(f"We can see the following files in {folder_name}")
                st.write_stream(f"* {x['title']}\n" for x in files_dict.values())

    gauth = GoogleAuth()
    gauth.settings["get_refresh_token"] = True
    gauth.settings["oauth_scope"] = ["https://www.googleapis.com/auth/userinfo.email", "openid"]
    # print(st.query_params)
    auth_code = st.query_params.get("code")
    st.query_params.clear()
    redirect_uri = os.environ.get("REDIRECT_URL", "http://localhost:8501/LogIn")
    # print("Flowing Auth: ", redirect_uri)
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        "client_secrets.json",
        scopes=["https://www.googleapis.com/auth/userinfo.email", "openid"],
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
        st.session_state.google_auth = None
        st.session_state.user_info = user_info
        st.rerun()
    else:
        st.write("SIL uses your Google account to manage authorization for the onboarding dashboard.")
        st.write(
            "At this authorization level, the onboarding dashboard will have no access to files in your Google drive, except for those that are shared with the service account <clowder@clowder-400318.iam.gserviceaccount.com>. We will only modify or view the files within the folders that you give us in the pages following. If you would like to modify the permission level, toggle **Modify Google Drive Access** above to the on position, then move the slider below."
        )
        st.write(
            "If you would like more information on how to give the Onboarding Dashboard access to your files, and check that we are able to access them, click below."
        )
        if st.button("More Information"):
            _access()
        st.write("Otherwise ...")
        authorization_url, _ = flow.authorization_url(
            access_type="offline", include_granted_scopes="true", prompt="consent", state=state
        )
        st.link_button(url=authorization_url, label="Sign in with Google", type="primary")


AUTH_FLOWS = {
    "BYPASS": auth_flow_bypass,
    "FULL": auth_flow_full,
    "RESTRICTED": auth_flow_restricted,
    "SERVICE": auth_flow_service,
}

with st.session_state.auth_lock:
    if "google_auth" not in st.session_state:
        st.title("Welcome to the Onboarding Dashboard")
        if st.toggle("Modify Google Drive access"):
            st.select_slider("Access Level", AUTH_OPTIONS, key="auth_method")
        state_obj = {"FROM_O-D": True, "AUTH_METHOD": st.session_state.auth_method}
        enc_state = encrypt(pickle.dumps(state_obj))
        AUTH_FLOWS[st.session_state.auth_method](enc_state)
    else:
        if bypass_auth and st.session_state.auth_method == "BYPASS":
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
            print(f"User {st.session_state.user_info.get('email', 'X' * 100)} is:")
            print(f"\tAn internal user: {is_internal_user}")
            print(f"\tAn external user: {is_external_user}")
            st.session_state.is_internal_user = is_internal_user
            is_allowed_user = is_internal_user or is_external_user
        if is_allowed_user:
            st.switch_page("pages/InitialSetUp.py")
        else:
            st.title("You do not have access to this application.")
