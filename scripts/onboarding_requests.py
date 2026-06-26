import argparse
import concurrent.futures
import logging
import os
import subprocess
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List

from requests import Response, post, get
from clearml import Task

LOGGER = logging.getLogger(__name__)


class OnboardingEnvironment:
    SF_AUTH_PWD: str = None
    SF_CLIENT_ID: str = None
    ONBOARDING_PATH: Path = None
    ONBOARDING_LOG_PATH: Path = None
    ONBOARDING_CLEANUP_PATH: Path = None
    ONBOARDING_REQUESTS_URL: str = None
    PROJECTS_URL: str = None
    SF_AUTH_URL: str = None
    ONBOARDING_REQUESTS_BUCKET_DIR: str = None
    REPO_PATH: Path = None
    SF_API_TOKEN: str = None
    UUID: str = None

    @classmethod
    def create_production_environment(cls):
        cls.ONBOARDING_REQUESTS_URL = "https://scriptureforge.org/command-api/onboarding-requests"
        cls.PROJECTS_URL = "https://scriptureforge.org/paratext-api/projects"
        cls.SF_AUTH_URL = "https://login.languagetechnology.org/oauth/token"
        cls.SF_AUTH_PWD = os.getenv("SF_AUTH_PWD")
        cls.SF_CLIENT_ID = os.getenv("SF_CLIENT_ID")
        cls.ONBOARDING_PATH = Path(os.getenv("ONBOARDING_PATH"))
        cls.setup_environment()

    @classmethod
    def create_qa_environment(cls):
        cls.ONBOARDING_REQUESTS_URL = "https://qa.scriptureforge.org/command-api/onboarding-requests"
        cls.PROJECTS_URL = "https://qa.scriptureforge.org/paratext-api/projects"
        cls.SF_AUTH_URL = "https://dev-sillsdev.auth0.com/oauth/token"
        cls.SF_AUTH_PWD = os.getenv("SF_QA_AUTH_PWD")
        cls.SF_CLIENT_ID = os.getenv("SF_QA_CLIENT_ID")
        cls.ONBOARDING_PATH = Path(os.getenv("QA_ONBOARDING_PATH"))
        cls.setup_environment()

    @classmethod
    def setup_environment(cls):
        cls.ONBOARDING_PATH.mkdir(parents=True, exist_ok=True)
        cls.ONBOARDING_LOG_PATH = cls.ONBOARDING_PATH / "onboarded_projects.log"
        cls.ONBOARDING_CLEANUP_PATH = cls.ONBOARDING_PATH / "paths_to_delete.txt"
        cls.ONBOARDING_LOG_PATH.touch()
        cls.ONBOARDING_CLEANUP_PATH.touch()
        cls.ONBOARDING_REQUESTS_BUCKET_DIR = "MT/experiments/_OnboardingRequests"
        cls.REPO_PATH = Path(os.getenv("REPO_PATH"))
        os.makedirs(cls.ONBOARDING_PATH, exist_ok=True)

        cls.SF_AUTH_USER = os.getenv("SF_AUTH_USER")
        cls.UUID = str(uuid.uuid4())
        payload = {
            "username": cls.SF_AUTH_USER,
            "password": cls.SF_AUTH_PWD,
            "grant_type": "http://auth0.com/oauth/grant-type/password-realm",
            "client_id": cls.SF_CLIENT_ID,
            "realm": "Username-Password-Authentication",
            "audience": "https://scriptureforge.org/",
            "scope": "openid profile email sf_data offline_access",
        }

        req = post(
            cls.SF_AUTH_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        cls.SF_API_TOKEN = req.json().get("access_token")


class OnboardingProject:
    def __init__(
        self,
        SF_id: str = "",
        paratext_id: str = "",
        short_name: str = "",
        language_name: str = "",
        language_code: str = "",
    ):
        self.SF_id: str = SF_id
        self.paratext_id: str = paratext_id
        self.short_name: str = short_name
        self.language_name: str = language_name
        self.language_code: str = language_code
        self.set_project_metadata()

    def set_project_metadata(self):
        metadata = (
            send_request(
                RequestType.GET,
                OnboardingEnvironment.ONBOARDING_REQUESTS_URL,
                "getProjectMetadata",
                {"scriptureForgeId": self.SF_id} if self.SF_id else {"paratextId": self.paratext_id},
            )
            .json()
            .get("result", {})
        )
        self.SF_id = metadata.get("id")
        self.paratext_id = metadata.get("paratextId")
        self.short_name = metadata.get("shortName")

    def download_project(self, download_path: Path) -> bool:
        project_url = f"{OnboardingEnvironment.PROJECTS_URL}/{self.SF_id}/download"
        response = send_request(RequestType.GET, project_url, "getProjectDownloadLink", {"paratextId": self.SF_id})
        if response.status_code != 200:
            return False
        if self.paratext_id and len(self.paratext_id) == 16:
            self.short_name = f"{self.short_name}_Resource"
        with open(f"{download_path}/{self.short_name}.zip", "wb") as f:
            f.write(response.content)
        return True


class OnboardingRequest:
    def __init__(self, request_dict: dict):
        self.id: str = request_dict.get("id")
        form_data: dict = request_dict["submission"]["formData"]
        self.main_project: OnboardingProject = OnboardingProject(
            SF_id=request_dict["submission"]["projectId"],
            language_name=form_data.get("translationLanguageName", ""),
            language_code=form_data.get("translationLanguageIsoCode", ""),
        )
        self.partner_org: str = form_data.get("partnerOrganization")
        if self.partner_org != "none":
            display_message("This request is for a partner organization.", MessageType.INFO, self.id)
        self.completed_books: str = form_data.get("completedBooks")
        self.planned_books: str = form_data.get("nextBooksToDraft")
        self.draft_source: OnboardingProject = (
            OnboardingProject(paratext_id=form_data.get("draftingSourceProject"))
            if form_data.get("draftingSourceProject")
            else None
        )
        self.bt_project: OnboardingProject = (
            OnboardingProject(
                paratext_id=form_data.get("backTranslationProject"),
                language_name=form_data.get("backTranslationLanguageName", ""),
                language_code=form_data.get("backTranslationLanguageIsoCode", ""),
            )
            if form_data.get("backTranslationProject")
            else None
        )
        self.reference_projects: List[OnboardingProject] = []
        for key in ["sourceProjectA", "sourceProjectB", "sourceProjectC"]:

            if key in form_data and not form_data[key] in [
                p.paratext_id for p in [self.draft_source, self.bt_project] if p is not None
            ]:
                self.reference_projects.append(OnboardingProject(paratext_id=form_data[key]))

    def download_projects(self) -> None:
        download_path = OnboardingEnvironment.ONBOARDING_PATH / Path(f"{self.main_project.short_name}_Request")
        os.makedirs(download_path, exist_ok=True)
        if not self.main_project.download_project(download_path):
            raise Exception(f"Failed to download main project {self.main_project.short_name}")
        for project in [self.draft_source, self.bt_project] + self.reference_projects:
            if project is not None:
                try:
                    if not project.download_project(download_path):
                        display_message(
                            f"Failed to download project {project.short_name}", MessageType.WARNING, self.id
                        )
                except Exception as e:
                    display_message(
                        f"Failed to download project {project.short_name}. Error: {e}", MessageType.WARNING, self.id
                    )


class RequestType(Enum):
    POST = "POST"
    GET = "GET"


def send_request(type: RequestType, url: str, method: str, params: dict) -> Response:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OnboardingEnvironment.SF_API_TOKEN}"}
    json_data = {
        "body": {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": OnboardingEnvironment.UUID,
        }
    }
    if type == RequestType.POST:
        response = post(url, headers=headers, json=json_data)
    elif type == RequestType.GET:
        response = get(url, headers=headers, json=json_data)
    else:
        raise ValueError("Invalid request type. Must be 'GET' or 'POST'.")
    return response


def add_comment(request_id: str, comment: str):
    send_request(
        RequestType.POST,
        OnboardingEnvironment.ONBOARDING_REQUESTS_URL,
        "addComment",
        {"requestId": request_id, "commentText": comment},
    )


def get_onboarding_requests() -> List[dict]:
    all_requests = (
        send_request(RequestType.GET, OnboardingEnvironment.ONBOARDING_REQUESTS_URL, "getAllRequests", {})
        .json()
        .get("result", [])
    )
    return [request for request in all_requests if request["status"] == "new"] if all_requests else []


def append_datestamp(project_name: str) -> str:
    now = datetime.now()
    datestamp = now.strftime("%Y_%m_%d")
    return f"{project_name}_{datestamp}"


def rename_project(project_name: str, datestamp: bool) -> str:
    resource = False
    if project_name.endswith("_Resource"):
        resource = True
        project_name = project_name.replace("_Resource", "")
    if "-" in project_name:
        project_name = project_name.replace("-", "_")
    if datestamp and not resource:
        project_name = append_datestamp(project_name)
    return project_name


def process_request(request_dict: dict):
    try:
        display_message("Processing this onboarding request...", MessageType.INFO, request_dict["id"])
        with open(OnboardingEnvironment.ONBOARDING_LOG_PATH, "a") as f:
            f.write(f"{request_dict['id']}\n")

        request = OnboardingRequest(request_dict)
        request.download_projects()

        task_name = f"Auto Onboarding - {request.main_project.short_name}"
        args = [
            "/usr/bin/python3",
            f"{OnboardingEnvironment.REPO_PATH}/scripts/automate_onboard_project.py",
            f"{request.main_project.short_name}.zip",
        ]
        if request.draft_source:
            args.extend(["--draft-source", request.draft_source.short_name])
        if request.bt_project:
            args.extend(["--bt-project", request.bt_project.short_name])
        if request.reference_projects:
            args.extend(
                [
                    "--reference-projects",
                    *[ref_project.short_name for ref_project in request.reference_projects],
                ]
            )

        args.extend(
            [
                "--completed-books",
                *[str(book) for book in request.completed_books],
                "--planned-books",
                *[str(book) for book in request.planned_books],
                "--dir",
                f"{OnboardingEnvironment.ONBOARDING_PATH}/{request.main_project.short_name}_Request",
                "--task-name",
                task_name,
            ]
        )
        subprocess.run(args)
        task: Task = Task.get_task(project_name="Onboarding", task_name=task_name, tags=["silnlp-auto-onboarding"])
        with open(OnboardingEnvironment.ONBOARDING_LOG_PATH, "a") as f:
            f.write(f"{request.id}\n")
        with open(OnboardingEnvironment.ONBOARDING_CLEANUP_PATH, "a") as f:
            f.write(f"{OnboardingEnvironment.ONBOARDING_PATH}/{request.main_project.short_name}_Request\n")

        display_message(
            f"This request is being automatically onboarded.\nClearML task: {task_name}.\nLink: {task.get_output_log_web_page()}",
            MessageType.INFO,
            request.id,
        )
        adjusted_name = rename_project(request.main_project.short_name, True)
        display_message(
            f"Results will be stored in {OnboardingEnvironment.ONBOARDING_REQUESTS_BUCKET_DIR}/{adjusted_name}",
            MessageType.INFO,
            request.id,
        )
    except Exception as e:
        display_message(f"Error processing onboarding request {request.id}:\n{e}", MessageType.ERROR, request.id)
        return

    try:
        task.wait_for_status()
        display_message(
            "Automatic onboarding was successful. See ClearML task for details.", MessageType.INFO, request.id
        )
    except Exception as e:
        display_message(
            f"Automatic onboarding failed. See ClearML task for details.\n {e}", MessageType.ERROR, request.id
        )


class MessageType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def display_message(message: str, message_type: MessageType, request_id: str = None):
    if message_type == MessageType.INFO:
        LOGGER.info(message)
    elif message_type == MessageType.WARNING:
        LOGGER.warning(message)
        message = f"WARNING: {message}"
    elif message_type == MessageType.ERROR:
        LOGGER.error(message)
        message = f"ERROR: {message}"
    if request_id:
        print(message)
        # add_comment(request_id, message)


def main():
    parser = argparse.ArgumentParser(description="Process onboarding requests.")
    parser.add_argument("--qa", action="store_true", help="Run on SF QA")
    args = parser.parse_args()

    if args.qa:
        OnboardingEnvironment.create_qa_environment()
    else:
        OnboardingEnvironment.create_production_environment()
    onboarding_requests = get_onboarding_requests()
    if not onboarding_requests:
        display_message("No new onboarding requests found.", MessageType.INFO)
        return
    onboarded_projects = []

    with open(OnboardingEnvironment.ONBOARDING_LOG_PATH, "r") as f:
        onboarded_projects = f.read().splitlines()
        onboarded_projects = [project_id.strip() for project_id in onboarded_projects]

    requests_to_process = [request for request in onboarding_requests if request["id"] not in onboarded_projects]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_request, request) for request in requests_to_process]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                display_message(f"Error processing request:\n{e}", MessageType.ERROR)


if __name__ == "__main__":
    main()
